import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

class ModelPerformance:

    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def conf_matrix(self, save_path):
        cm = confusion_matrix(self.y_test, self.model.predict(self.x_test))
        cm_df = pd.DataFrame(cm, index=np.unique(self.y_test), columns=np.unique(self.y_test))
        plt.figure(figsize=(6, 5))
        custom_cmap = sns.light_palette("#F6866A", as_cmap=True)
        sns.heatmap(cm_df, annot=True, fmt="d", cmap=custom_cmap)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.show()

    def plot_classification_report_radial(self, save_path):
        y_pred = self.model.predict(self.x_test)
        report_dict = classification_report(self.y_test, y_pred, output_dict=True, zero_division=1)  # Add zero_division=1 here
        comments = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        metrics = ['precision', 'recall', 'f1-score']

        # Data Preparation
        data = np.array([[report_dict[comment][metric] for metric in metrics] for comment in comments])
        data = np.concatenate((data, data[:, :1]), axis=1)  # Close the loop for radial chart
        labels = metrics + [metrics[0]]  # Close the loop for radial chart

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

        for i, comment in enumerate(comments):
            ax.plot(theta, data[i], label=f'Class: {comment}')
            ax.fill(theta, data[i], alpha=0.25)

        ax.set_xticks(theta)
        ax.set_xticklabels(labels)
        ax.set_title(f'{self.model} Classification Metrics (Radial)', fontsize=14)  # Use self.model here
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()