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
from sklearn.inspection import permutation_importance
import shap
from sklearn.ensemble import RandomForestClassifier

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
    # Create target variable
    y = (df['ACCLASS'] == 'Fatal').astype(int)

    logging.info(f"Final dataset shape: {X.shape}")
    logging.info(f"Number of Fatal accidents: {y.sum()}")
    logging.info(f"Number of Non-Fatal accidents: {len(y) - y.sum()}")
    
    return X, y, pipeline

def calculate_feature_importance_features(X, y):
    """Determine important features using multiple techniques"""
    logging.info(f"Determining important features using multiple techniques... Shape of X: {X.shape}")
    
    # Create and scale features for importance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    logging.info(f"X_scaled shape: {X_scaled.shape}, dtypes: {X_scaled.dtypes.iloc[0]}")
    
    # Train a decision tree to get feature importance
    dt = DecisionTreeClassifier(random_state=48)
    dt.fit(X_scaled, y)
    logging.info("Decision tree model trained successfully")
    
    # 1. Native Decision Tree feature importance
    logging.info("Calculating native feature importance...")
    native_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info(f"Top 20 important features (native): {native_importance['feature'].head(20).tolist()}")
    
    # Save native feature importance to CSV
    native_importance.to_csv('insights/performance/native_feature_importance.csv', index=False)
    
    # Plot native feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=native_importance.head(20), x='importance', y='feature')
    plt.title('Native Feature Importance (Decision Tree)')
    plt.tight_layout()
    plt.savefig('insights/performance/native_feature_importance.png')
    plt.close()
    logging.info("Native feature importance saved")
    
    # 2. Permutation Importance (more reliable than native importance)
    logging.info("Calculating permutation importance...")
    
    try:
        perm_importance = permutation_importance(dt, X_scaled, y, n_repeats=10, random_state=48)
        
        perm_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        logging.info(f"Top 20 important features (permutation): {perm_importance_df['feature'].head(20).tolist()}")
        
        # Save permutation importance to CSV
        perm_importance_df.to_csv('insights/performance/permutation_importance.csv', index=False)
        
        # Plot permutation importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=perm_importance_df.head(20), x='importance', y='feature')
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.savefig('insights/performance/permutation_importance.png')
        plt.close()
        logging.info("Permutation importance saved")
    except Exception as e:
        logging.error(f"Error during permutation importance calculation: {e}")
        perm_importance_df = native_importance.copy()  # Fallback to native importance
    
    # 3. SHAP Values for more detailed and accurate feature importance
    logging.info("Calculating SHAP values...")
    try:
        # Sample the data if it's too large
        sample_size = min(500, X_scaled.shape[0])  # Use at most 500 samples
        sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
        X_sample = X_scaled.iloc[sample_indices]
        logging.info(f"Sample for SHAP: {X_sample.shape}")
        
        # Convert to numpy array for compatibility with SHAP
        X_sample_values = X_sample.values
        feature_names = X_sample.columns.tolist()
        logging.info(f"X_sample_values shape: {X_sample_values.shape}")
        
        # Create the explainer with proper numpy array input
        logging.info("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(dt)
        logging.info("Explainer created, calculating SHAP values...")
        shap_values = explainer(X_sample_values)
        logging.info(f"SHAP values calculated, shape: {shap_values.shape}")
        
        # For binary classification, SHAP returns values for both classes
        # Use the values for the positive class (fatal accidents, class 1)
        # This is typically the second element in the values array (index 1)
        
        # Check if we have multi-dimensional SHAP values (for binary classification)
        if len(shap_values.shape) == 3:
            logging.info("Multi-dimensional SHAP values detected (binary classification)")
            # Use class 1 (index 1) for binary classification
            shap_values_for_plots = shap_values.values[:, :, 1]
            logging.info(f"Using SHAP values for positive class, shape: {shap_values_for_plots.shape}")
        else:
            # If it's just samples x features, use as is
            shap_values_for_plots = shap_values.values
            logging.info(f"Using direct SHAP values, shape: {shap_values_for_plots.shape}")
        
        # SHAP Summary Plot
        logging.info("Creating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_for_plots, X_sample_values, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('insights/performance/shap_summary.png')
        plt.close()
        
        # SHAP Bar Plot
        logging.info("Creating SHAP bar plot...")
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values_for_plots, X_sample_values, feature_names=feature_names, plot_type="bar", show=False)
        plt.title('SHAP Mean Absolute Feature Importance')
        plt.tight_layout()
        plt.savefig('insights/performance/shap_bar.png')
        plt.close()
        
        # SHAP Dependence Plot for top feature
        logging.info("Creating SHAP dependence plot for top feature...")
        # First get the feature with highest mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values_for_plots).mean(0)
        top_feature_idx = np.argmax(mean_abs_shap)
        top_feature = feature_names[top_feature_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            top_feature_idx, 
            shap_values_for_plots, 
            X_sample_values, 
            feature_names=feature_names,
            show=False
        )
        plt.title(f'SHAP Dependence Plot for {top_feature}')
        plt.tight_layout()
        plt.savefig(f'insights/performance/shap_dependence_{top_feature}.png')
        plt.close()
        
        # Calculate and save SHAP importance values
        # Create DataFrame with feature importance based on SHAP
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        logging.info(f"Top 20 important features (SHAP): {shap_importance['feature'].head(20).tolist()}")
        
        # Save SHAP importance to CSV
        shap_importance.to_csv('insights/performance/shap_importance.csv', index=False)
        logging.info("SHAP importance saved")
        
    except Exception as e:
        logging.error(f"Error calculating SHAP values: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        logging.warning("Continuing without SHAP analysis")
        shap_importance = None
    
    # 4. Random Forest MDI (Mean Decrease in Impurity)
    logging.info("Calculating Random Forest Mean Decrease in Impurity...")
    try:
        # Train a random forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=48, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # Get MDI feature importance
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info(f"Top 20 important features (RF MDI): {rf_importance['feature'].head(20).tolist()}")
        
        # Save RF MDI importance to CSV
        rf_importance.to_csv('insights/performance/rf_mdi_importance.csv', index=False)
        
        # Plot RF MDI importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=rf_importance.head(20), x='importance', y='feature')
        plt.title('Random Forest MDI Feature Importance')
        plt.tight_layout()
        plt.savefig('insights/performance/rf_mdi_importance.png')
        plt.close()
        logging.info("Random Forest MDI importance saved")
    except Exception as e:
        logging.error(f"Error calculating Random Forest MDI: {e}")
        rf_importance = native_importance.copy()  # Fallback to native importance

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