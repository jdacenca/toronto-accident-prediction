"""Visualization utilities for model evaluation and feature importance."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from .config import PERFORMANCE_DIR

def plot_feature_importance(importance_df: pd.DataFrame, title: str, filename: str, top_n: int = 20) -> None:
    """Plot feature importance bar chart."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PERFORMANCE_DIR / filename)
    plt.close()

def plot_decision_tree(model: Any, feature_names: list[str], max_depth: int = 3) -> None:
    """Create and save a visualization of the decision tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
             feature_names=feature_names,
             class_names=['Non-Fatal', 'Fatal'],
             filled=True,
             rounded=True,
             max_depth=max_depth)
    plt.savefig(PERFORMANCE_DIR / 'decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    cm = pd.crosstab(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(PERFORMANCE_DIR / 'confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
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
    plt.savefig(PERFORMANCE_DIR / 'roc_curve.png')
    plt.close()

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = np.mean(precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve\n(Average Precision = {avg_precision:.2f})')
    plt.axhline(y=sum(y_true)/len(y_true), color='red', linestyle='--',
                label=f'No Skill (Baseline = {sum(y_true)/len(y_true):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(PERFORMANCE_DIR / 'precision_recall_curve.png')
    plt.close()

def plot_importance_comparison(comparison_df: pd.DataFrame) -> None:
    """Plot comparison of feature importance across different methods."""
    # Plot comparison of features
    comparison_df.plot(kind='bar', figsize=(14, 10))
    plt.title('Feature Importance Comparison Across Methods')
    plt.xlabel('Features')
    plt.ylabel('Normalized Importance')
    plt.legend(title='Method')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(PERFORMANCE_DIR / 'importance_comparison.png')
    plt.close()
    
    # Create a heatmap for better visualization
    plt.figure(figsize=(14, 12))
    sns.heatmap(comparison_df.head(20), cmap='viridis', annot=True, fmt='.2f')
    plt.title('Feature Importance Heatmap Across Methods')
    plt.tight_layout()
    plt.savefig(PERFORMANCE_DIR / 'importance_heatmap.png')
    plt.close() 