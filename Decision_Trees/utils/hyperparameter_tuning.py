"""Hyperparameter tuning utilities for decision tree models."""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from .config import INSIGHTS_DIR, RANDOM_STATE

warnings.filterwarnings('ignore')
plt.style.use('default')

class HyperparameterTuning:
    """Class for tuning decision tree hyperparameters."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """Initialize with data.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        self.X = X
        self.y = y
        self.results: list[dict] = []
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        (INSIGHTS_DIR / 'tuning').mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def prepare_data(self, sampling_strategy: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data with optional sampling strategy.
        
        Args:
            sampling_strategy: Strategy to use for handling class imbalance
                             ('oversampling', 'undersampling', 'SMOTE', or None)
        
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            random_state=RANDOM_STATE,
            stratify=self.y
        )
        
        # Print original class distribution
        if sampling_strategy is None:
            logging.info("\nOriginal Class Distribution:")
            logging.info(f"Fatal: {sum(y_train == 1)}")
            logging.info(f"Non-Fatal: {sum(y_train == 0)}")
            logging.info(f"Total samples: {len(y_train)}")
        
        # Apply sampling strategy if specified
        if sampling_strategy == 'oversampling':
            sampler = RandomUnderSampler(
                sampling_strategy='majority',
                random_state=RANDOM_STATE
            )
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            logging.info("\nOversampling Class Distribution:")
            
        elif sampling_strategy == 'undersampling':
            sampler = RandomUnderSampler(
                sampling_strategy='majority',
                random_state=RANDOM_STATE
            )
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            logging.info("\nUndersampling Class Distribution:")
            
        elif sampling_strategy == 'SMOTE':
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info("\nSMOTE Class Distribution:")
            
        if sampling_strategy is not None:
            logging.info(f"Fatal: {sum(y_train == 1)}")
            logging.info(f"Non-Fatal: {sum(y_train == 0)}")
            logging.info(f"Total samples: {len(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model: DecisionTreeClassifier, 
                      X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray,
                      model_name: str) -> dict:
        """Evaluate a model and store results.
        
        Args:
            model: The trained model to evaluate
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            model_name: Name of the model for saving results
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5,
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
        
        # Create visualizations
        self._create_visualizations(model, model_name, y_test, y_test_pred, y_test_prob)
        
        return results
    
    def _create_visualizations(self, model: DecisionTreeClassifier, model_name: str, y_test: np.ndarray,
                             y_test_pred: np.ndarray, y_test_prob: np.ndarray) -> None:
        """Create and save model evaluation visualizations.
        
        Args:
            model: The trained model
            model_name: Name of the model
            y_test: Test labels
            y_test_pred: Predicted labels
            y_test_prob: Predicted probabilities
        """
        # Create visualizations directory
        viz_dir = INSIGHTS_DIR / 'tuning' / model_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png')
        plt.close()
        
        # 2. ROC Curve
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
        
        # 3. Classification Report Visualization
        plt.figure(figsize=(8, 6))
        cr_dict = classification_report(y_test, y_test_pred, output_dict=True)
        
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
        
        sns.heatmap(cr_df.round(2), annot=True, cmap='RdPu', fmt='.2f', cbar=True)
        plt.title(f'Classification Report - {model_name}')
        plt.tight_layout()
        plt.savefig(viz_dir / 'classification_report.png')
        plt.close()
        
        # 4. Save classification report as text
        report = classification_report(y_test, y_test_pred)
        with open(viz_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            f.write("\n\nModel Parameters:\n")
            f.write("-"*20 + "\n")
            f.write(str(model.get_params()))
    
    def save_results(self) -> None:
        """Save tuning results to CSV and markdown files."""
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save to CSV
        results_df.to_csv(INSIGHTS_DIR / 'tuning' / 'tuning_results.csv', index=False)
        
        # Save to markdown
        with open(INSIGHTS_DIR / 'tuning' / 'tuning_results.md', 'w') as f:
            f.write("# Decision Tree Hyperparameter Tuning Results\n\n")
            f.write(results_df.to_markdown()) 