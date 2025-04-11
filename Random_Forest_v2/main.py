# main.py

"""Main script for training and evaluating a Random Forest model."""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils.config import (
    DATA_DIR, SERIALIZED_DIR, PERFORMANCE_DIR,
    MODEL_PARAMS, SCORING_METRICS, RANDOM_STATE
)
from utils.visualization import plot_feature_importance, plot_importance_comparison
from utils.evaluation import (
    evaluate_model,
    calculate_feature_importance,
    save_model_artifacts
)
from utils.pipeline import create_preprocessing_pipeline
from utils.sampling import apply_sampling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories() -> None:
    """Create necessary directories for outputs."""
    dirs = [SERIALIZED_DIR, PERFORMANCE_DIR]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data() -> tuple[pd.DataFrame, np.ndarray, Any]:
    """Load and preprocess the data."""
    logging.info("Loading data...")
    df = pd.read_csv(DATA_DIR / 'TOTAL_KSI.csv')

    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline()

    logging.info("Preprocessing data...")
    X = pipeline.fit_transform(df)
    y = (df['ACCLASS'] == 'FATAL').astype(int)

    logging.info(f"Final dataset shape: {X.shape}")
    logging.info(f"Number of Fatal accidents: {y.sum()}")
    logging.info(f"Number of Non-Fatal accidents: {len(y) - y.sum()}")

    return X, y, pipeline

def train_random_forest(X: pd.DataFrame, y: np.ndarray) -> RandomForestClassifier:
    """Train the Random Forest model using grid search."""
    logging.info("Training and evaluating Random Forest...")

    # Optional: Apply sampling technique to mitigate class imbalance
    # E.g. "smote_tomek" as in your original approach
    X_resampled, y_resampled = apply_sampling(X, y, method='smote_tomek')

    # Split train/test AFTER resampling
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_resampled
    )

    # Create base model
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')

    # Create grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=MODEL_PARAMS,
        cv=5,
        scoring=SCORING_METRICS,
        refit='f1',  # pick F1 as the main refit metric
        n_jobs=-1,
        verbose=1
    )

    logging.info("Performing grid search on Random Forest with resampled data...")
    grid_search.fit(X_train, y_train)

    # Log best parameters
    logging.info("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        logging.info(f"{param}: {value}")

    # Log best metrics for each scoring
    for metric in SCORING_METRICS.keys():
        best_idx = grid_search.best_index_
        score = grid_search.cv_results_[f'mean_test_{metric}'][best_idx]
        logging.info(f"Best {metric} score: {score:.3f}")

    best_model = grid_search.best_estimator_

    # Evaluate on test set
    feature_names = X.columns.tolist()  # if X is still a DataFrame-like object
    evaluate_model(best_model, X_test, y_test, feature_names)

    return best_model

def analyze_feature_importance(model: RandomForestClassifier, X: pd.DataFrame, y: np.ndarray) -> None:
    """Analyze and visualize feature importance using multiple methods."""
    logging.info("Calculating feature importance...")

    native_imp, perm_imp, shap_imp = calculate_feature_importance(model, X, y)

    # Save to CSV
    native_imp.to_csv(PERFORMANCE_DIR / 'native_feature_importance.csv', index=False)
    perm_imp.to_csv(PERFORMANCE_DIR / 'permutation_importance.csv', index=False)
    shap_imp.to_csv(PERFORMANCE_DIR / 'shap_importance.csv', index=False)

    # Plot them
    plot_feature_importance(native_imp, 'Native Feature Importance (Random Forest)', 'native_feature_importance.png')
    plot_feature_importance(perm_imp, 'Permutation Feature Importance', 'permutation_importance.png')
    plot_feature_importance(shap_imp, 'SHAP Feature Importance', 'shap_importance.png')

    # Compare them
    comparison_df = pd.DataFrame({
        'Native': native_imp.set_index('feature')['importance'],
        'Permutation': perm_imp.set_index('feature')['importance'],
        'SHAP': shap_imp.set_index('feature')['importance']
    })

    comparison_df = comparison_df.div(comparison_df.sum())  # normalize
    plot_importance_comparison(comparison_df)

    # Save top features
    top_features = comparison_df.mean(axis=1).sort_values(ascending=False).index.tolist()
    with open(PERFORMANCE_DIR / 'top_features.txt', 'w') as f:
        f.write("Most Important Features (Averaged Across Methods):\n")
        for i, feature in enumerate(top_features, 1):
            f.write(f"{i}. {feature}\n")

def main() -> None:
    """Main execution function."""
    start_time = time.time()
    setup_directories()

    X, y, pipeline = load_and_preprocess_data()
    model = train_random_forest(X, y)

    # Optional: re-check feature importance on the entire dataset
    analyze_feature_importance(model, X, y)

    # Save final artifacts
    from utils.evaluation import save_model_artifacts
    save_model_artifacts(model, pipeline, 'random_forest_model.pkl', 'preprocessing_pipeline.pkl')

    end_time = time.time()
    logging.info(f"Random Forest training completed. Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
