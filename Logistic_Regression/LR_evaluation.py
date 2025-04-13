from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint
from LR_preprocessor import LRPreprocessor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, auc,
    precision_recall_curve, roc_curve)

class LREvaluation:
    def __init__(self, model, preprocessor, X_test, y_test):
        self.model = model
        df_test = preprocessor.transform(pd.concat([X_test, y_test], axis=1))
        # self.X_test = X_test
        # self.y_test = y_test
        self.X_test, self.y_test = df_test.drop('ACCLASS', axis = 1), df_test['ACCLASS']

        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = self.model.predict_proba(self.X_test)[:,-1]

    def confusion_matrix(self, save_path):
        """
        Generates and plot the confusion matrix.

        Parameters:
        - save_path: Path to save the confusion matrix to.
        """
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.model.classes_)
        print(f"Confusion Matrix:")
        print(cm)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.model.classes_)
        disp.plot()
        plt.savefig(save_path)
        plt.show()

    def roc_auc(self, save_path):
        """
        Generates and saves a ROC curve.

        Parameters:
        - save_path: Path to save the Precision-Recall curve to.
        """
        # Compute ROC curve and ROC-AUC
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.show()

    def precision_recall_auc(self, save_path):
        """
        Generates and saves a Precision-recall curve, particularly useful for imbalanced classes. Note the use of probability to see the threshold.

        Parameters:
        - save_path: Path to save the Precision-Recall curve to.
        """
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        precision_auc = auc(recall, precision)

        # Plot Precision-Recall curve
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % precision_auc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(save_path)
        plt.show()

    def classification_report(self, save_path):
        """
        Generates and saves a heatmap of the classification report.

        Parameters:
        - save_path: Path to save the classification report heatmap.
        """
        report = classification_report(self.y_test, self.model.predict(self.X_test), output_dict=True)
        pprint(report)
        report_df = pd.DataFrame(report).iloc[:-1, :].T

        plt.figure(figsize=(8, 6))
        sns.heatmap(report_df, annot=True, cmap="coolwarm", cbar=False)
        plt.title("Classification Report", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


