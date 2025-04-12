"""Script for tuning decision tree hyperparameters."""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import logging
from pathlib import Path
import warnings
from utils.sampling import apply_sampling, separate_unseen_data
from utils.pipeline import create_preprocessing_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from utils.hyperparameter_tuning import HyperparameterTuning
from utils.config import DATA_DIR, RANDOM_STATE, TARGET

warnings.filterwarnings('ignore')
plt.style.use('default')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyperparameterTuning:
    def __init__(self, X_train, y_train, X_unseen, y_unseen):
        """Initialize with data"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_unseen = X_unseen
        self.y_unseen = y_unseen
        self.results = []
        self.unseen_results = []
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['insights/tuning', 'insights/unseen_testing']
        for dir_path in dirs:
            Path(f'{dir_path}').mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, sampling_strategy=None):
        """Prepare data with optional sampling strategy"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, 
            test_size=0.2, 
            random_state=RANDOM_STATE,
            stratify=self.y_train
        )
                
        # Apply sampling strategy to training data
        if sampling_strategy:
            X_train, y_train = apply_sampling(X_train, y_train, sampling_strategy)
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate a model and store results"""
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Train the model on the full training set
        model.fit(X_train, y_train)
        
        # Get predictions and probabilities
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
        
        # Store results
        self.results.append(results)
        
        # Log results
        logging.info(f"\nResults for {model_name}:")
        logging.info(f"Train Accuracy: {results['Train Acc.']:.2f}%")
        logging.info(f"Test Accuracy: {results['Test Acc.']:.2f}%")
        logging.info(f"Precision: {results['Precision']:.4f}")
        logging.info(f"Recall: {results['Recall']:.4f}")
        logging.info(f"F1-Score: {results['F1-Score']:.4f}")
        
        # Create visualizations directory if it doesn't exist
        viz_dir = Path(f'insights/tuning/{model_name}')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/confusion_matrix.png')
        plt.close()
        
        # 2. Plot and save ROC curve
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
        plt.savefig(f'{viz_dir}/roc_curve.png')
        plt.close()
        
        # 3. Create and save classification report visualization
        plt.figure(figsize=(8, 6))
        
        # Get classification report as dict
        cr_dict = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Convert to DataFrame for easier plotting
        cr_df = pd.DataFrame({
            'precision': [cr_dict['0']['precision'], cr_dict['1']['precision'],
                        cr_dict['accuracy'], cr_dict['macro avg']['precision'],
                        cr_dict['weighted avg']['precision']],
            'recall': [cr_dict['0']['recall'], cr_dict['1']['recall'],
                      cr_dict['accuracy'], cr_dict['macro avg']['recall'],
                      cr_dict['weighted avg']['recall']],
            'f1-score': [cr_dict['0']['f1-score'], cr_dict['1']['f1-score'],
                        cr_dict['accuracy'], cr_dict['macro avg']['f1-score'],
                        cr_dict['weighted avg']['f1-score']]
        }, index=['0', '1', 'accuracy', 'macro avg', 'weighted avg'])
        
        # Create heatmap
        sns.heatmap(cr_df.round(2), annot=True, cmap='RdPu', fmt='.2f', cbar=True)
        plt.title(f'Classification Report - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/classification_report.png')
        plt.close()
        
        # 4. Save classification report as text
        report = classification_report(y_test, y_test_pred)
        with open(f'{viz_dir}/classification_report.txt', 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            f.write("\n\nModel Parameters:\n")
            f.write("-"*20 + "\n")
            f.write(str(model.get_params()))
        
        # If unseen data is available, evaluate on it too
        if self.X_unseen is not None and self.y_unseen is not None:
            # Extract sampling strategy (if any) from model name
            parts = model_name.split(' ')
            sampling = None if len(parts) == 1 else parts[-1]
            
            # Evaluate on unseen data with the same sampling strategy
            self.evaluate_on_unseen_data(model, model_name, sampling)
            
        return model
        
    def evaluate_on_unseen_data(self, model, model_name, sampling_strategy=None):
        """Evaluate trained model on completely unseen data with optional sampling"""
        logging.info(f"\nEvaluating {model_name} on unseen data...")
        
        # Apply same sampling strategy to unseen data if specified
        if sampling_strategy in ['oversampling', 'undersampling', 'SMOTE']:
            X_unseen, y_unseen = self.apply_sampling(self.X_unseen, self.y_unseen, sampling_strategy)
            logging.info(f"Applied {sampling_strategy} to unseen data")
            logging.info(f"Unseen data after {sampling_strategy}: {len(y_unseen)} samples")
            logging.info(f"Fatal: {sum(y_unseen == 1)}, Non-Fatal: {sum(y_unseen == 0)}")
        else:
            X_unseen, y_unseen = self.X_unseen, self.y_unseen
        
        # Get predictions and probabilities on unseen data
        y_unseen_pred = model.predict(X_unseen)
        y_unseen_prob = model.predict_proba(X_unseen)[:, 1]
        
        # Calculate metrics
        unseen_results = {
            'Model': model_name,
            'Sampling': sampling_strategy if sampling_strategy else 'None',
            'Accuracy': accuracy_score(y_unseen, y_unseen_pred) * 100,
            'Precision': precision_score(y_unseen, y_unseen_pred, zero_division=0),
            'Recall': recall_score(y_unseen, y_unseen_pred, zero_division=0),
            'F1-Score': f1_score(y_unseen, y_unseen_pred, zero_division=0)
        }
        
        # Store results
        self.unseen_results.append(unseen_results)
        
        # Log results
        logging.info(f"Unseen Data Results for {model_name}:")
        logging.info(f"Accuracy: {unseen_results['Accuracy']:.2f}%")
        logging.info(f"Precision: {unseen_results['Precision']:.4f}")
        logging.info(f"Recall: {unseen_results['Recall']:.4f}")
        logging.info(f"F1-Score: {unseen_results['F1-Score']:.4f}")
        
        # Create directory for unseen results if it doesn't exist
        viz_dir = Path(f'insights/unseen_testing/{model_name}')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create and save classification report for unseen data
        report = classification_report(y_unseen, y_unseen_pred)
        with open(f'{viz_dir}/unseen_classification_report.txt', 'w') as f:
            f.write(f"Unseen Data Classification Report for {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Sampling Strategy: {sampling_strategy if sampling_strategy else 'None'}\n\n")
            f.write(report)
            
        # Plot and save confusion matrix for unseen data
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_unseen, y_unseen_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Unseen Data Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/unseen_confusion_matrix.png')
        plt.close()
    
    def run_comparison(self):
        """Run comparison with different configurations"""
        sampling_strategies = [None, 'smote', 'random_over', 'random_under', 'smote_tomek', 'smote_enn']
        
        for sampling in sampling_strategies:
            logging.info(f"\nTesting with {sampling if sampling else 'no'} sampling")
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(sampling)
            
            # Basic Decision Tree
            dt_basic = DecisionTreeClassifier(
                class_weight='balanced', 
                min_samples_split=2, 
                random_state=RANDOM_STATE)
            self.evaluate_model(
                dt_basic, X_train, X_test, y_train, y_test,
                (f"basic {sampling if sampling else ''}").rstrip()
            )
            
            # Decision Tree with Gini
            dt_gini = DecisionTreeClassifier(
                class_weight='balanced',
                criterion='gini',
                min_samples_split=2,
                random_state=RANDOM_STATE
            )
            self.evaluate_model(
                dt_gini, X_train, X_test, y_train, y_test,
                (f"gini {sampling if sampling else ''}").rstrip()
            )
            
            # Decision Tree with Entropy
            dt_entropy = DecisionTreeClassifier(
                class_weight='balanced',
                criterion='entropy',
                min_samples_split=2,
                random_state=RANDOM_STATE
            )
            self.evaluate_model(
                dt_entropy, X_train, X_test, y_train, y_test,
                (f"entropy {sampling if sampling else ''}").rstrip()
            )
    
    def save_results(self):
        """Save results to CSV and markdown"""
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save to CSV
        results_df.to_csv(f'insights/tuning/results.csv', index=False)
        logging.info(f"\nResults saved to insights/tuning/results.csv")

        # Create a formatted markdown table
        markdown_table = f"# Decision Tree Tuning Results \n\n"
        markdown_table += "| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |\n"
        markdown_table += "|-------|------------|-----------|-----------|---------|-----------|----------|\n"
        
        for result in self.results:
            model_parts = result['Model'].split(' ')
            sampling = model_parts[-1] if len(model_parts) > 1 else 'original'
            markdown_table += f"| {result['Model']} | "
            markdown_table += f"{result['Train Acc.']:.2f} | "
            markdown_table += f"{result['Test Acc.']:.2f} | "
            markdown_table += f"{result['Precision']:.4f} | "
            markdown_table += f"{result['Recall']:.4f} | "
            markdown_table += f"{result['F1-Score']:.4f} | "
            markdown_table += f"{sampling} |\n"
        
        # Save markdown table
        with open(f'insights/tuning/results.md', 'w') as f:
            f.write(markdown_table)
        
        logging.info(f"Markdown results saved to insights/tuning/results.md")
        
        # Save unseen data results if available
        if self.unseen_results:
            # Convert unseen results to DataFrame
            unseen_df = pd.DataFrame(self.unseen_results)
            
            # Save to CSV
            unseen_df.to_csv(f'insights/unseen_testing/unseen_results.csv', index=False)
            logging.info(f"\nUnseen data results saved to insights/unseen_testing/unseen_results.csv")

            # Create a formatted markdown table for unseen results
            markdown_table = f"# Decision Tree Performance on Unseen Data \n\n"
            markdown_table += "| Model | Sampling | Accuracy | Precision | Recall | F1-Score |\n"
            markdown_table += "|-------|----------|----------|-----------|--------|----------|\n"
            
            for result in self.unseen_results:
                markdown_table += f"| {result['Model']} | "
                markdown_table += f"{result['Sampling']} | "
                markdown_table += f"{result['Accuracy']:.2f} | "
                markdown_table += f"{result['Precision']:.4f} | "
                markdown_table += f"{result['Recall']:.4f} | "
                markdown_table += f"{result['F1-Score']:.4f} |\n"
            
            # Save markdown table
            with open(f'insights/unseen_testing/unseen_results.md', 'w') as f:
                f.write(markdown_table)
    
    def plot_unseen_comparison(self):
        """Create visual comparison of all models on unseen data"""
        if not self.unseen_results:
            return
            
        # Convert to DataFrame
        unseen_df = pd.DataFrame(self.unseen_results)
        
        # Create directory
        viz_dir = Path('insights/unseen_testing/comparison')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bar plots for each metric
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            plt.figure(figsize=(14, 8))
            
            # Create grouped bar chart
            ax = sns.barplot(x='Model', y=metric, hue='Sampling', data=unseen_df)
            
            plt.title(f'{metric} Comparison on Unseen Data')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Sampling Strategy')
            plt.tight_layout()
            
            plt.savefig(f'{viz_dir}/unseen_{metric.lower()}_comparison.png')
            plt.close()
        
        # Create heatmap for F1-score
        pivot_df = unseen_df.pivot(index='Model', columns='Sampling', values='F1-Score')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'F1-Score'})
        plt.title('F1-Score by Model and Sampling Strategy on Unseen Data')
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/unseen_f1_heatmap.png')
        plt.close()
        
        logging.info(f"Unseen data comparison visualizations saved to {viz_dir}")

def main():
    """Main execution function."""
    # Load and preprocess data
    data_path = DATA_DIR / 'TOTAL_KSI_6386614326836635957.csv'
    df = pd.read_csv(data_path)
    
    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline()
    
    # First preprocess the data
    processed_df = pipeline.fit_transform(df)
    # Seperate features and target variable
    X = processed_df.drop(columns=[TARGET])
    y = processed_df[TARGET] 

    X_train, y_train, X_unseen, y_unseen = separate_unseen_data(X, y)
    
    # Initialize hyperparameter tuning
    tuner = HyperparameterTuning(X_train, y_train, X_unseen, y_unseen)
    
    # Run comparison
    tuner.run_comparison()
    
    # Create visual comparisons of unseen data results
    tuner.plot_unseen_comparison()
    
    # Save all results
    tuner.save_results()

if __name__ == "__main__":
    main()