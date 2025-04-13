import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve, classification_report, precision_recall_curve, accuracy_score,average_precision_score
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def plot_lift_gain_curves(classifiers, X_test, y_test, save_folder):
    # Plot Lift & Gain curves only
    plt.figure(figsize=(12, 10))
    
    for classifier in classifiers:
        if hasattr(classifier, "predict_proba"):
            y_test_proba = classifier.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = classifier.predict(X_test)

        # Sort the probabilities in descending order
        sorted_indices = np.argsort(y_test_proba)[::-1]
        sorted_probs = y_test_proba[sorted_indices]
        sorted_true = y_test[sorted_indices]

        # Calculate Lift & Gain
        total_positive = np.sum(y_test == 1)
        total_negative = np.sum(y_test == 0)
        
        cumulative_positive = np.cumsum(sorted_true)  # True positives accumulated
        gain = cumulative_positive / total_positive  # Gain curve
        lift = gain / (np.arange(1, len(gain) + 1) / len(y_test))  # Lift curve
        
        # Label name for the classifier
        if "VotingClassifier" in str(classifier):
            voting_type = "hard" if classifier.voting == "hard" else "soft"
            label_name = f"Voting ({voting_type})"
        else:
            label_name = str(classifier).split('(')[0]

        # Plot Gain curve
        plt.plot(np.arange(1, len(gain) + 1) / len(y_test), gain, label=f"{label_name} Gain Curve")
        
        # Plot Lift curve
        plt.plot(np.arange(1, len(lift) + 1) / len(y_test), lift, label=f"{label_name} Lift Curve")
    
    # Add labels and title
    plt.xlabel("Percentile of Samples")
    plt.ylabel("Gain / Lift")
    plt.title("Combined Lift and Gain Curves")
    plt.legend(loc="lower left")
    plt.grid()
    
    # Save the combined plot
    plt.savefig(f"ensemble_performance/performance_metrics/combined_plot/{save_folder}/combined_lift_gain_curve.png")
    plt.show()


def plot_combined_performance_bar_plots(classifiers, X_train, y_train, X_test, y_test, save_folder):
    """
    Plots combined bar charts for training and testing accuracy, precision, recall, and F1 score of all classifiers.

    Parameters:
    - classifiers: List of classifiers to evaluate.
    - X_train, y_train: Training data.
    - X_test, y_test: Testing data.
    - save_folder: Folder to save the plots.
    """
    
    # Initialize lists to store metric results for each model
    metrics = {
        'Model': [],
        'Train Accuracy': [],
        'Test Accuracy': [],
        'Train Precision': [],
        'Test Precision': [],
        'Train Recall': [],
        'Test Recall': [],
        'Train F1': [],
        'Test F1': []
    }
    
    for classifier in classifiers:
        model_name = str(classifier).split('(')[0]
        
        # Training metrics
        y_train_pred = classifier.predict(X_train)
        metrics['Model'].append(model_name)
        metrics['Train Accuracy'].append(accuracy_score(y_train, y_train_pred))
        metrics['Train Precision'].append(precision_score(y_train, y_train_pred))
        metrics['Train Recall'].append(recall_score(y_train, y_train_pred))
        metrics['Train F1'].append(f1_score(y_train, y_train_pred))
        
        # Testing metrics
        y_test_pred = classifier.predict(X_test)
        metrics['Test Accuracy'].append(accuracy_score(y_test, y_test_pred))
        metrics['Test Precision'].append(precision_score(y_test, y_test_pred))
        metrics['Test Recall'].append(recall_score(y_test, y_test_pred))
        metrics['Test F1'].append(f1_score(y_test, y_test_pred))

    # Convert metrics dictionary to numpy array for plotting
    metric_names = ['Train Accuracy', 'Test Accuracy', 'Train Precision', 'Test Precision', 
                    'Train Recall', 'Test Recall', 'Train F1', 'Test F1']
    
    metric_values = np.array([metrics[name] for name in metric_names]).T
    
    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define positions for bars
    bar_width = 0.1
    index = np.arange(len(classifiers))  # Number of classifiers

    # Plot bars for each metric
    for i, metric_name in enumerate(metric_names):
        ax.bar(index + (i - 4) * bar_width, metric_values[:, i], bar_width, label=metric_name)

    # Add labels and title
    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison (Train vs Test)')
    ax.set_xticks(index)
    ax.set_xticklabels(metrics['Model'])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"ensemble_performance/performance_metrics/combined_plot/{save_folder}/combined_performance_bar_plot.png")
    # plt.show()


def plot_combined_precision_recall_curves(classifiers, X_test, y_test, save_folder):

    # Plot PR curve for testing data
    plt.figure(figsize=(10, 8))
    for classifier in classifiers:
        if hasattr(classifier, "predict_proba"):
            y_test_proba = classifier.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = classifier.predict(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        avg_precision = average_precision_score(y_test, y_test_proba)

        if "VotingClassifier" in str(classifier):
            voting_type = "hard" if classifier.voting == "hard" else "soft"
            label_name = f"Voting ({voting_type})"
        else:
            label_name = str(classifier).split('(')[0]

        plt.plot(recall, precision, label=f"{label_name} (AP = {avg_precision:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curve (Testing)")
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig(f"ensemble_performance/performance_metrics/combined_plot/{save_folder}/combined_pr_curve_testing.png")

def plot_combined_roc_curves(classifiers, X_train, y_train, X_test, y_test, save_folder):
    """
    Plots combined ROC curves for all classifiers on training and testing datasets.

    Parameters:
    - classifiers: List of classifiers to evaluate.
    - X_train, y_train: Training data.
    - X_test, y_test: Testing data.
    - save_folder: Folder to save the plots.
    """
    # Plot ROC for training data
    plt.figure(figsize=(10, 8))
    for classifier in classifiers:
        if hasattr(classifier, "predict_proba"):
            y_train_proba = classifier.predict_proba(X_train)[:, 1]
        else:
            y_train_proba = classifier.predict(X_train)
        fpr, tpr, _ = roc_curve(y_train, y_train_proba)
        roc_auc = auc(fpr, tpr)
        
        if "VotingClassifier" in str(classifier):
            voting_type = "hard" if classifier.voting == "hard" else "soft"
            label_name = f"Voting ({voting_type})"
        else:
            label_name = str(classifier).split('(')[0]

        plt.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.2f})")   

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.title("Combined ROC Curve (Training)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"ensemble_performance/performance_metrics/combined_plot/{save_folder}/combined_roc_curve_training.png")


    # Plot ROC for testing data
    plt.figure(figsize=(10, 8))
    for classifier in classifiers:
        if hasattr(classifier, "predict_proba"):
            y_test_proba = classifier.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = classifier.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)

        if "VotingClassifier" in str(classifier):
            voting_type = "hard" if classifier.voting == "hard" else "soft"
            label_name = f"Voting ({voting_type})"
        else:
            label_name = str(classifier).split('(')[0]

        plt.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.2f})")   
         
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.title("Combined ROC Curve (Testing)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"ensemble_performance/performance_metrics/combined_plot/{save_folder}/combined_roc_curve_testing.png")

class ModelPerformance:
    """
    A class to evaluate and visualize the performance of a machine learning model.
    """

    def __init__(self, model, X_train,y_train, x_test, y_test):
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
        self.x_train = X_train
        self.y_train = y_train

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

        if hasattr(self.model, "predict_proba"):
            y_scores = self.model.predict_proba(self.x_test)[:, 1]
            y_scores_train = self.model.predict_proba(self.x_train)[:, 1]
        else:
            y_scores = self.model.predict(self.x_test)
            y_scores_train = self.model.predict(self.x_train)
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)
        fpr_train, tpr_train, _ = roc_curve(self.y_train, y_scores_train)

        roc_auc = auc(fpr, tpr)
        roc_auc_train = auc(fpr_train, tpr_train)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_train, tpr_train, color="green", label=f"Train ROC Curve (AUC = {roc_auc_train:.2f})")
        plt.plot(fpr, tpr, color="darkorange", label=f"Test ROC Curve (AUC = {roc_auc:.2f})")
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

