"""Model evaluation utilities."""

import pandas as pd
import numpy as np
from typing import Any, Tuple
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_decision_tree
)
from utils.config import PERFORMANCE_DIR, RANDOM_STATE, SERIALIZED_DIR
from sklearn.inspection import permutation_importance
import shap

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Evaluate model performance and generate visualizations."""
    # Make predictions
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
    
    # Generate visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_decision_tree(model, feature_names)
    
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

def calculate_feature_importance(model: Any, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate feature importance using multiple methods."""
    
    # Native feature importance
    native_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=RANDOM_STATE)
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    if len(shap_values.shape) == 3:
        shap_values_for_plots = shap_values.values[:, :, 1]
    else:
        shap_values_for_plots = shap_values.values
    
    mean_abs_shap = np.abs(shap_values_for_plots).mean(0)
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    return native_importance, perm_importance_df, shap_importance

def save_model_artifacts(model: Any, pipeline: Any) -> None:
    """Save model and preprocessing pipeline."""
    import joblib
    
    joblib.dump(model, SERIALIZED_DIR / 'decision_tree_model.pkl')
    joblib.dump(pipeline, SERIALIZED_DIR / 'preprocessing_pipeline.pkl') 