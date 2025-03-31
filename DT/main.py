"""Main script for training and evaluating the Decision Tree model."""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from utils.config import (
    DATA_DIR, SERIALIZED_DIR, PERFORMANCE_DIR,
    MODEL_PARAMS, SCORING_METRICS, RANDOM_STATE
)
from utils.visualization import plot_feature_importance, plot_importance_comparison
from utils.evaluation import evaluate_model, calculate_feature_importance, save_model_artifacts
from preprocessing.pipeline import create_preprocessing_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories() -> None:
    """Create necessary directories for outputs."""
    dirs = [SERIALIZED_DIR, PERFORMANCE_DIR]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data() -> tuple[pd.DataFrame, np.ndarray, Any]:
    """Load and preprocess the data.
    
    Returns:
        tuple: (X, y, pipeline) where X is the feature matrix, y is the target vector,
               and pipeline is the fitted preprocessing pipeline.
    """
    logging.info("Loading data...")
    df = pd.read_csv(DATA_DIR / 'TOTAL_KSI_6386614326836635957.csv')
    
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

def train_model(X: pd.DataFrame, y: np.ndarray) -> DecisionTreeClassifier:
    """Train the decision tree model using grid search.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        DecisionTreeClassifier: The best trained model
    """
    logging.info("Training and evaluating model...")
    
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
        X, y, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Create base model
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=MODEL_PARAMS,
        cv=5,
        scoring=SCORING_METRICS,
        refit='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit on training data
    logging.info("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    # Log detailed results
    logging.info("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        logging.info(f"{param}: {value}")
    
    for metric in SCORING_METRICS.keys():
        score = grid_search.cv_results_[f'mean_test_{metric}']
        best_idx = grid_search.best_index_
        logging.info(f"Best {metric} score: {score[best_idx]:.3f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    feature_names = X.columns.tolist()
    evaluate_model(best_model, X_test, y_test, feature_names)
    
    return best_model

def analyze_feature_importance(model: DecisionTreeClassifier, X: pd.DataFrame, y: np.ndarray) -> None:
    """Analyze and visualize feature importance using multiple methods."""
    logging.info("Calculating feature importance...")
    
    # Calculate feature importance using multiple methods
    native_importance, perm_importance_df, shap_importance = calculate_feature_importance(model, X, y)
    
    # Save feature importance to CSV files
    native_importance.to_csv(PERFORMANCE_DIR / 'native_feature_importance.csv', index=False)
    perm_importance_df.to_csv(PERFORMANCE_DIR / 'permutation_importance.csv', index=False)
    shap_importance.to_csv(PERFORMANCE_DIR / 'shap_importance.csv', index=False)
    
    # Plot feature importance
    plot_feature_importance(native_importance, 'Native Feature Importance (Decision Tree)', 'native_feature_importance.png')
    plot_feature_importance(perm_importance_df, 'Permutation Feature Importance', 'permutation_importance.png')
    plot_feature_importance(shap_importance, 'SHAP Feature Importance', 'shap_importance.png')
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Native': native_importance.set_index('feature')['importance'],
        'Permutation': perm_importance_df.set_index('feature')['importance'],
        'SHAP': shap_importance.set_index('feature')['importance']
    })
    
    # Normalize importances
    comparison_df = comparison_df.div(comparison_df.sum())
    
    # Plot comparison
    plot_importance_comparison(comparison_df)
    
    # Save top features to a separate file
    top_features = comparison_df.mean(axis=1).sort_values(ascending=False).index.tolist()
    with open(PERFORMANCE_DIR / 'top_features.txt', 'w') as f:
        f.write("Most Important Features (Averaged Across Methods):\n")
        for i, feature in enumerate(top_features, 1):
            f.write(f"{i}. {feature}\n")

def main() -> None:
    """Main execution function."""
    start_time = time.time()
    
    # Setup
    setup_directories()
    
    # Load and preprocess data
    X, y, pipeline = load_and_preprocess_data()

    # Train model
    model = train_model(X, y)
    
    # Analyze feature importance
    analyze_feature_importance(model, X, y)
    
    # Save artifacts
    save_model_artifacts(model, pipeline)
    
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Model built completed successfully! Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()