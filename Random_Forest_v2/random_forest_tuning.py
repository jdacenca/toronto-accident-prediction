
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import logging
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from utils.sampling import apply_sampling
from utils.pipeline import create_preprocessing_pipeline
from utils.config import DATA_DIR, RANDOM_STATE

warnings.filterwarnings('ignore')
plt.style.use('default')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RandomForestTuning:
    def __init__(self, X, y, unseen_X=None, unseen_y=None):
        self.X = X
        self.y = y
        self.unseen_X = unseen_X
        self.unseen_y = unseen_y
        self.results = []
        self.unseen_results = []
        self.setup_directories()

    def setup_directories(self):
        dirs = ['insights/tuning', 'insights/unseen_testing']
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def prepare_data(self, sampling_strategy=None):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=self.y
        )

        if sampling_strategy:
            X_train, y_train = apply_sampling(X_train, y_train, sampling_strategy)

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5,  # x-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Train the model on the full training set
        model.fit(X_train, y_train)

        # Predictions
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        results = {
            'Model': model_name,
            'Train Acc.': cv_scores.mean() * 100,
            'Test Acc.': accuracy_score(y_test, y_test_pred) * 100,
            'Precision': precision_score(y_test, y_test_pred, zero_division=0),
            'Recall': recall_score(y_test, y_test_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_test_pred, zero_division=0),
            'Parameters': str(model.get_params())
        }

        self.results.append(results)

        # Logging
        logging.info(f"\nResults for {model_name}:")
        logging.info(f"Train Accuracy: {results['Train Acc.']:.2f}%")
        logging.info(f"Test Accuracy: {results['Test Acc.']:.2f}%")
        logging.info(f"Precision: {results['Precision']:.4f}")
        logging.info(f"Recall: {results['Recall']:.4f}")
        logging.info(f"F1-Score: {results['F1-Score']:.4f}")

        # Visualization directory
        viz_dir = Path(f"insights/tuning/{model_name}")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(viz_dir / 'roc_curve.png')
        plt.close()

        # Classification Report Visualization
        plt.figure(figsize=(8, 6))
        cr_dict = classification_report(y_test, y_test_pred, output_dict=True)
        cr_df = pd.DataFrame({
            'precision': [
                cr_dict['0']['precision'],
                cr_dict['1']['precision'],
                cr_dict['accuracy'],
                cr_dict['macro avg']['precision'],
                cr_dict['weighted avg']['precision']
            ],
            'recall': [
                cr_dict['0']['recall'],
                cr_dict['1']['recall'],
                cr_dict['accuracy'],
                cr_dict['macro avg']['recall'],
                cr_dict['weighted avg']['recall']
            ],
            'f1-score': [
                cr_dict['0']['f1-score'],
                cr_dict['1']['f1-score'],
                cr_dict['accuracy'],
                cr_dict['macro avg']['f1-score'],
                cr_dict['weighted avg']['f1-score']
            ]
        }, index=['0', '1', 'accuracy', 'macro avg', 'weighted avg'])

        sns.heatmap(cr_df.round(2), annot=True, cmap='RdPu', fmt='.2f', cbar=True)
        plt.title(f'Classification Report - {model_name}')
        plt.tight_layout()
        plt.savefig(viz_dir / 'classification_report.png')
        plt.close()

        # Save classification report as text
        report = classification_report(y_test, y_test_pred)
        with open(viz_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            f.write("\n\nModel Parameters:\n")
            f.write("-"*20 + "\n")
            f.write(str(model.get_params()))

        # Evaluate on unseen data
        if self.unseen_X is not None and self.unseen_y is not None:
            parts = model_name.split()
            sampling = parts[-1] if len(parts) > 1 else None

            self.evaluate_on_unseen_data(model, model_name, sampling)

        return model

    def evaluate_on_unseen_data(self, model, model_name, sampling_strategy=None):
        logging.info(f"\nEvaluating {model_name} on unseen data...")

        if sampling_strategy in ['smote', 'random_over', 'random_under', 'smote_tomek', 'smote_enn']:
            unseen_X, unseen_y = apply_sampling(self.unseen_X, self.unseen_y, sampling_strategy)
            logging.info(f"Applied {sampling_strategy} to unseen data")
        else:
            unseen_X, unseen_y = self.unseen_X, self.unseen_y

        y_unseen_pred = model.predict(unseen_X)
        y_unseen_prob = model.predict_proba(unseen_X)[:, 1]

        # Calculate metrics
        unseen_results = {
            'Model': model_name,
            'Sampling': sampling_strategy if sampling_strategy else 'None',
            'Accuracy': accuracy_score(unseen_y, y_unseen_pred) * 100,
            'Precision': precision_score(unseen_y, y_unseen_pred, zero_division=0),
            'Recall': recall_score(unseen_y, y_unseen_pred, zero_division=0),
            'F1-Score': f1_score(unseen_y, y_unseen_pred, zero_division=0)
        }
        self.unseen_results.append(unseen_results)

        # Logging
        logging.info(f"Unseen Data Results for {model_name}:")
        logging.info(f"Accuracy: {unseen_results['Accuracy']:.2f}%")
        logging.info(f"Precision: {unseen_results['Precision']:.4f}")
        logging.info(f"Recall: {unseen_results['Recall']:.4f}")
        logging.info(f"F1-Score: {unseen_results['F1-Score']:.4f}")

        # Visualization
        viz_dir = Path(f"insights/unseen_testing/{model_name}")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Save classification report
        report = classification_report(unseen_y, y_unseen_pred)
        with open(viz_dir / 'unseen_classification_report.txt', 'w') as f:
            f.write(f"Unseen Data Classification Report for {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Sampling Strategy: {sampling_strategy if sampling_strategy else 'None'}\n\n")
            f.write(report)

        # Confusion Matrix for unseen data
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(unseen_y, y_unseen_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Unseen Data Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(viz_dir / 'unseen_confusion_matrix.png')
        plt.close()

    def run_comparison(self):
        sampling_strategies = [None, 'smote', 'random_over', 'random_under', 'smote_tomek', 'smote_enn']

        for sampling in sampling_strategies:
            logging.info(f"\nTesting with {sampling if sampling else 'no'} sampling")

            X_train, X_test, y_train, y_test = self.prepare_data(sampling)

            # Basic RandomForestClassifier
            rf_basic = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            )
            self.evaluate_model(rf_basic, X_train, X_test, y_train, y_test,
                                (f"basic {sampling if sampling else ''}").rstrip())


    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('insights/tuning/results.csv', index=False)
        logging.info("\nResults saved to insights/tuning/results.csv")

        markdown_table = "# Random Forest Tuning Results\n\n"
        markdown_table += "| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |\n"
        markdown_table += "|-------|------------|-----------|-----------|--------|----------|----------|\n"

        for result in self.results:
            parts = result['Model'].split()
            sampling = parts[-1] if len(parts) > 1 else 'None'
            markdown_table += f"| {result['Model']} | "
            markdown_table += f"{result['Train Acc.']:.2f} | "
            markdown_table += f"{result['Test Acc.']:.2f} | "
            markdown_table += f"{result['Precision']:.4f} | "
            markdown_table += f"{result['Recall']:.4f} | "
            markdown_table += f"{result['F1-Score']:.4f} | "
            markdown_table += f"{sampling} |\n"

        with open('insights/tuning/results.md', 'w') as f:
            f.write(markdown_table)
        logging.info("Markdown results saved to insights/tuning/results.md")

        if self.unseen_results:
            unseen_df = pd.DataFrame(self.unseen_results)
            unseen_df.to_csv('insights/unseen_testing/unseen_results.csv', index=False)
            logging.info("\nUnseen data results saved to insights/unseen_testing/unseen_results.csv")

            # Unseen data markdown
            md_unseen = "# Random Forest Performance on Unseen Data\n\n"
            md_unseen += "| Model | Sampling | Accuracy | Precision | Recall | F1-Score |\n"
            md_unseen += "|-------|----------|----------|-----------|--------|----------|\n"

            for result in self.unseen_results:
                md_unseen += f"| {result['Model']} | "
                md_unseen += f"{result['Sampling']} | "
                md_unseen += f"{result['Accuracy']:.2f} | "
                md_unseen += f"{result['Precision']:.4f} | "
                md_unseen += f"{result['Recall']:.4f} | "
                md_unseen += f"{result['F1-Score']:.4f} |\n"

            with open('insights/unseen_testing/unseen_results.md', 'w') as f:
                f.write(md_unseen)

    def plot_unseen_comparison(self):
        if not self.unseen_results:
            return

        unseen_df = pd.DataFrame(self.unseen_results)
        viz_dir = Path('insights/unseen_testing/comparison')
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Create bar plots for each metric
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            plt.figure(figsize=(14, 8))
            sns.barplot(x='Model', y=metric, hue='Sampling', data=unseen_df)
            plt.title(f'{metric} Comparison on Unseen Data')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Sampling Strategy')
            plt.tight_layout()
            plt.savefig(viz_dir / f'unseen_{metric.lower()}_comparison.png')
            plt.close()

        # Heatmap for F1-Score
        pivot_df = unseen_df.pivot(index='Model', columns='Sampling', values='F1-Score')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'F1-Score'})
        plt.title('F1-Score by Model and Sampling Strategy on Unseen Data')
        plt.tight_layout()
        plt.savefig(viz_dir / 'unseen_f1_heatmap.png')
        plt.close()

        logging.info(f"Unseen data comparison visualizations saved to {viz_dir}")

def main():
    # Load raw data
    data_path = DATA_DIR / 'TOTAL_KSI_6386614326836635957.csv'
    df = pd.read_csv(data_path)

    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline()

    # Prepare features (X) and target (y)
    X = pipeline.fit_transform(df)
    y = (df['ACCLASS'] == 'FATAL').astype(int)

    # Reserve some unseen data if desired
    unseen_X = X[-10:]
    unseen_y = y[-10:]
    X = X[:-10]
    y = y[:-10]

    # Initialize tuner
    tuner = RandomForestTuning(X, y, unseen_X, unseen_y)

    # Run sampling strategy comparisons
    tuner.run_comparison()

    # Plot unseen results
    tuner.plot_unseen_comparison()

    # Save results
    tuner.save_results()

if __name__ == "__main__":
    main()
