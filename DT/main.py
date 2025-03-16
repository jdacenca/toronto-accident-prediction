import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score
from preprocessing_pipeline import create_preprocessing_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories():
    """Create necessary directories for outputs"""
    dirs = ['insights/serialized_artifacts', 'insights/performance']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the data"""
    logging.info("Loading data...")
    df = pd.read_csv('data/TOTAL_KSI_6386614326836635957.csv')
    
    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline()
    
    logging.info("Preprocessing data...")
    # First preprocess the data
    X = pipeline.fit_transform(df)
    
    # Get the indices of rows that weren't dropped
    mask = df['ACCLASS'] != 'Property Damage O'
    df_processed = df[mask]
    
    # Create target variable
    y = (df_processed['ACCLASS'] == 'Fatal').astype(int)

    logging.info(f"Final dataset shape: {X.shape}")
    logging.info(f"Number of Fatal accidents: {y.sum()}")
    logging.info(f"Number of Non-Fatal accidents: {len(y) - y.sum()}")
    
    return X, y, pipeline

def calculate_feature_importance_features(X, y):
    """Determine important features using a simple decision tree"""
    logging.info("Determining important features...")
    
    # Create and scale features for importance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train a decision tree to get feature importance
    dt = DecisionTreeClassifier(random_state=48)
    dt.fit(X_scaled, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('insights/performance/feature_importance.png')
    plt.close()
    
    # Save feature importance to CSV
    feature_importance.to_csv('insights/performance/feature_importance.csv', index=False)

def perform_grid_search(X_train, y_train):
    """Perform grid search to find the best hyperparameters"""
    logging.info("Performing grid search for hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    dt = DecisionTreeClassifier(random_state=48)
    
    # Create grid search object
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Log results
    logging.info("Best parameters found:")
    for param, value in grid_search.best_params_.items():
        logging.info(f"{param}: {value}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Save grid search results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv('insights/grid_search_results.csv', index=False)
    
    return grid_search.best_estimator_, grid_search.best_params_

def visualize_tree(model, feature_names, max_depth=3):
    """Create and save a visualization of the decision tree"""
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
             feature_names=feature_names,
             class_names=['Non-Fatal', 'Fatal'],
             filled=True,
             rounded=True,
             max_depth=max_depth)
    plt.savefig('insights/performance/decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate_model(X, y):
    """Train and evaluate the decision tree model"""
    logging.info("Training and evaluating model...")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Calculate class weights
    n_samples = len(y)
    n_classes = len(np.unique(y))
    class_weights = dict(zip(
        np.unique(y),
        n_samples / (n_classes * np.bincount(y))
    ))
    logging.info(f"Using class weights: {class_weights}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=48,
        stratify=y
    )
    
    # # SMOTE for balancing 
    # smote = SMOTE(random_state=48)
    # X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    # logging.info(f"After SMOTE - Training samples shape: {X_train_balanced.shape}")
    
    # Enhanced parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', class_weights, None], 
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    dt = DecisionTreeClassifier(random_state=48)
    
    # Create grid search with both accuracy and f1 scoring
    scoring = {
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'accuracy': 'accuracy'
    }
    
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit='f1',  # Still optimize for F1 score
        n_jobs=-1,
        verbose=1
    )
    
    # Fit on balanced data
    logging.info("Performing grid search with balanced data...")
    grid_search.fit(X_train, y_train)
    
    # Log detailed results
    logging.info("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        logging.info(f"{param}: {value}")
    
    for metric in scoring.keys():
        score = grid_search.cv_results_[f'mean_test_{metric}']
        best_idx = grid_search.best_index_
        logging.info(f"Best {metric} score: {score[best_idx]:.3f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Create tree visualization
    visualize_tree(best_model, X.columns)
    
    # Make predictions and calculate metrics
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    
    # Generate and save detailed classification report
    report = classification_report(y_test, y_pred)
    logging.info("\nClassification Report:\n" + report)
    
    with open('insights/performance/classification_report.txt', 'w') as f:
        f.write("Model Performance Metrics:\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Accuracy Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Testing Accuracy:  {test_accuracy:.3f}\n\n")
        
        f.write("Best Parameters:\n")
        f.write("-" * 20 + "\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nCross-Validation Scores:\n")
        f.write("-" * 20 + "\n")
        for metric in scoring.keys():
            score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
            f.write(f"{metric}: {score:.3f}\n")
        
        f.write("\nDetailed Classification Report:\n")
        f.write("-" * 20 + "\n")
        f.write(report)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('insights/performance/confusion_matrix.png')
    plt.close()

    # Create ROC curve
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('insights/performance/roc_curve.png')
    plt.close()

    # Create Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve\n(Average Precision = {avg_precision:.2f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--',
                label=f'No Skill (Baseline = {sum(y_test)/len(y_test):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig('insights/performance/precision_recall_curve.png')
    plt.close()

    return best_model

def save_model(model, pipeline):
    """Save the model and artifacts"""
    logging.info("Saving model and artifacts...")
    
    # Save model
    joblib.dump(model, 'insights/serialized_artifacts/decision_tree_model.pkl')
    
    # Save pipeline
    joblib.dump(pipeline, 'insights/serialized_artifacts/preprocessing_pipeline.pkl')

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Setup
    setup_directories()
    
    # Load and preprocess data
    X, y, pipeline = load_and_preprocess_data()
    
    # Calculate feature importance
    calculate_feature_importance_features(X, y)
    
    # Train and evaluate model
    model = train_and_evaluate_model(X, y)
    
    # Save artifacts
    save_model(model, pipeline)
    
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Model built completed successfully! Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()