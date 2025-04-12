"""Model evaluation utilities"""

import pandas as pd
import numpy as np
from typing import Any, Tuple
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score
)
from sklearn.inspection import permutation_importance
import shap
import joblib
import logging

from tqdm import tqdm  # For the progress bar

from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)
from utils.config import PERFORMANCE_DIR, RANDOM_STATE, SERIALIZED_DIR

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str]
) -> dict[str, float]:

    logging.info("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'roc_auc': auc(*roc_curve(y_test, y_prob)[:2]),
        'average_precision': average_precision_score(y_test, y_prob)
    }

    # Generate and save visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)

    # Save metrics to file
    with open(PERFORMANCE_DIR / 'classification_report.txt', 'w') as f:
        f.write("Model Performance Metrics:\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.3f}\n")
        f.write(f"Average Precision: {metrics['average_precision']:.3f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 20 + "\n")
        f.write(metrics['classification_report'])

    return metrics

def calculate_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    shap_sample_size: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # 1) Native feature importance
    logging.info("Calculating native feature importance...")
    native_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 2) Permutation importance
    logging.info("Calculating permutation importance (this may take a while)...")
    perm_results = permutation_importance(model, X, y, n_repeats=10, random_state=RANDOM_STATE)
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_results.importances_mean
    }).sort_values('importance', ascending=False)

    # 3) SHAP values
    #    If the dataset is large, sample rows to reduce computation time.
    n_samples = len(X)
    if n_samples > shap_sample_size:
        logging.info(f"Dataset has {n_samples} rows; sampling {shap_sample_size} for SHAP.")
        X_sample = X.sample(n=shap_sample_size, random_state=RANDOM_STATE)
    else:
        logging.info("Using the full dataset for SHAP calculation.")
        X_sample = X

    logging.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Check if SHAP is 3D (for binary classification in shap 0.x)
    if shap_values.values.ndim == 3:
        shap_values_for_plots = shap_values.values[:, :, 1]
    else:
        shap_values_for_plots = shap_values.values

    # Create a single-line updating progress bar across the feature dimension
    logging.info("Computing mean absolute SHAP values with a progress bar...")
    n_features = shap_values_for_plots.shape[1]

    # Tweak bar format for a single-line display
    bar_format = (
        "{l_bar}{bar}| "
        "{n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    mean_abs_shap = []
    for j in tqdm(
        range(n_features),
        desc="SHAP Calculation",
        dynamic_ncols=True,
        ascii=True,
        ncols=80,
        bar_format=bar_format
    ):
        abs_shap_j = np.mean(np.abs(shap_values_for_plots[:, j]))
        mean_abs_shap.append(abs_shap_j)

    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    return native_importance, perm_importance_df, shap_importance

def save_model_artifacts(
    model: Any,
    pipeline: Any,
    model_filename: str = 'random_forest_model.pkl',
    pipeline_filename: str = 'preprocessing_pipeline.pkl'
) -> None:

    joblib.dump(model, SERIALIZED_DIR / model_filename)
    joblib.dump(pipeline, SERIALIZED_DIR / pipeline_filename)