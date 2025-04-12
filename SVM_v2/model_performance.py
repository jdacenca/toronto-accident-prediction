import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve, classification_report


class ModelPerformance:
    """
    A class to evaluate and visualize the performance of a machine learning model.
    """

    def __init__(self, model, x_test, y_test):
        """
        Initializes the ModelPerformance class.

        Parameters:
        - model: Trained machine learning model.
        - x_test: Test features.
        - y_test: Test labels.
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def conf_matrix(self, save_path):
        """
        Generates and saves a confusion matrix heatmap.

        Parameters:
        - save_path: Path to save the confusion matrix plot.
        """
        cm = confusion_matrix(self.y_test, self.model.predict(self.x_test))
        cm_df = pd.DataFrame(cm, index=np.unique(self.y_test), columns=np.unique(self.y_test))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="viridis", cbar=True)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        #plt.show()

    def roc_cur(self, save_path):
        """
        Generates and saves a Receiver Operating Characteristic (ROC) curve.

        Parameters:
        - save_path: Path to save the ROC curve plot.
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.model.predict(self.x_test))

        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Random Guess")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.legend(loc="lower right", frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        #plt.show()

    # classification report heatmap
    def classification_report_heatmap(self, save_path):
        """
        Generates and saves a heatmap of the classification report.

        Parameters:
        - save_path: Path to save the classification report heatmap.
        """
        report = classification_report(self.y_test, self.model.predict(self.x_test), output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :].T

        plt.figure(figsize=(8, 6))
        sns.heatmap(report_df, annot=True, cmap="rocket", cbar=True)
        plt.tight_layout()
        plt.savefig(save_path)
        #plt.show()

