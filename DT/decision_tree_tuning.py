import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
from preprocessing_pipeline import create_preprocessing_pipeline
import pandas as pd
    
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HyperparameterTuning:
    def __init__(self, X, y):
        """Initialize with data"""
        self.X = X
        self.y = y
        self.results = []
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['insights/dt_tuning']
        for dir_path in dirs:
            Path(f'{dir_path}').mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, sampling_strategy=None):
        """Prepare data with optional sampling strategy"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            random_state=48,
            stratify=self.y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply sampling strategy if specified
        if sampling_strategy == 'oversampling':
            # Random oversampling
            sampler = RandomUnderSampler(
                sampling_strategy='majority',
                random_state=48
            )
            X_train_scaled, y_train = sampler.fit_resample(X_train_scaled, y_train)
        
        elif sampling_strategy == 'undersampling':
            # Random undersampling
            sampler = RandomUnderSampler(
                sampling_strategy='majority',
                random_state=48
            )
            X_train_scaled, y_train = sampler.fit_resample(X_train_scaled, y_train)
        
        elif sampling_strategy == 'SMOTE':
            # SMOTE
            smote = SMOTE(random_state=48)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate a model and store results"""
        # Train the model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            'Model': model_name,
            'Train Acc.': accuracy_score(y_train, y_train_pred) * 100,
            'Test Acc.': accuracy_score(y_test, y_test_pred) * 100,
            'Precision': precision_score(y_test, y_test_pred),
            'Recall': recall_score(y_test, y_test_pred),
            'F1-Score': f1_score(y_test, y_test_pred),
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
        
        return model
    
    def run_comparison(self):
        """Run comparison with different configurations"""
        sampling_strategies = [None, 'oversampling', 'undersampling', 'SMOTE']
        
        for sampling in sampling_strategies:
            logging.info(f"\nTesting with {sampling if sampling else 'no'} sampling")
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(sampling)
            
            # Basic Decision Tree
            dt_basic = DecisionTreeClassifier(random_state=48)
            self.evaluate_model(
                dt_basic, X_train, X_test, y_train, y_test,
                f"DT_basic_{sampling if sampling else 'original'}"
            )
            
            # Decision Tree with Gini
            dt_gini = DecisionTreeClassifier(
                criterion='gini',
                min_samples_split=5,
                class_weight='balanced',
                random_state=48
            )
            self.evaluate_model(
                dt_gini, X_train, X_test, y_train, y_test,
                f"DT_gini_{sampling if sampling else 'original'}"
            )
            
            # Decision Tree with Entropy
            dt_entropy = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=10,
                min_samples_split=5,
                random_state=48
            )
            self.evaluate_model(
                dt_entropy, X_train, X_test, y_train, y_test,
                f"DT_entropy_{sampling if sampling else 'original'}"
            )
            
            # Decision Tree with Class Weights
            dt_weighted = DecisionTreeClassifier(
                class_weight='balanced',
                random_state=48
            )
            self.evaluate_model(
                dt_weighted, X_train, X_test, y_train, y_test,
                f"DT_weighted_{sampling if sampling else 'original'}"
            )
    
    def save_results(self, dataset_name):
        """Save results to CSV and markdown"""
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save to CSV
        results_df.to_csv(f'insights/dt_tuning/{dataset_name}_results.csv', index=False)
        logging.info(f"\nResults saved to insights/dt_tuning/{dataset_name}_results.csv")
        
        # Create a formatted markdown table
        markdown_table = f"# Decision Tree Tuning Results ({dataset_name})\n\n"
        markdown_table += "| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |\n"
        markdown_table += "|-------|------------|-----------|-----------|---------|-----------|----------|\n"
        
        for result in self.results:
            model_parts = result['Model'].split('_')
            sampling = model_parts[-1] if len(model_parts) > 1 else 'original'
            markdown_table += f"| {result['Model']} | "
            markdown_table += f"{result['Train Acc.']:.2f} | "
            markdown_table += f"{result['Test Acc.']:.2f} | "
            markdown_table += f"{result['Precision']:.4f} | "
            markdown_table += f"{result['Recall']:.4f} | "
            markdown_table += f"{result['F1-Score']:.4f} | "
            markdown_table += f"{sampling} |\n"
        
        # Save markdown table
        with open(f'insights/dt_tuning/{dataset_name}_results.md', 'w') as f:
            f.write(markdown_table)
        
        logging.info(f"Markdown results saved to insights/dt_tuning/{dataset_name}_results.md")

def main():
    """Main execution function"""

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    df = pd.read_csv('data/TOTAL_KSI_6386614326836635957.csv')
    
    # Extract last 10 rows as unseen data
    unseen_data = df.iloc[:-10]  # Removes the last 10 rows from df
    # Save unseen data for reference
    unseen_data.to_csv('data/unseen_data.csv', index=False)
    pipeline = create_preprocessing_pipeline()
    
    # Create target variable for main dataset
    mask_main = df['ACCLASS'] != 'Property Damage O'
    df_processed_main = df[mask_main]
    y_main = (df_processed_main['ACCLASS'] == 'Fatal').astype(int)
    
    # Process main dataset
    X_main = pipeline.fit_transform(df)
    
    # Convert X_main to DataFrame with proper column names
    feature_names = pipeline.named_steps['engineer'].feature_names_
    X_main = pd.DataFrame(X_main, columns=feature_names)

    # Run comparison on main dataset
    logging.info("Running comparison on main dataset...")
    comparison_main = HyperparameterTuning(X_main, y_main)
    comparison_main.run_comparison()
    comparison_main.save_results("Main dataset")
    
    # Process unseen dataset
    X_unseen = pipeline.fit_transform(unseen_data)
    
    # Create target variable for unseen dataset
    mask_unseen = unseen_data['ACCLASS'] != 'Property Damage O'
    df_processed_unseen = unseen_data[mask_unseen]
    y_unseen = (df_processed_unseen['ACCLASS'] == 'Fatal').astype(int)
    
    # Convert X_unseen to DataFrame with proper column names
    X_unseen = pd.DataFrame(X_unseen, columns=feature_names)

    # Run comparison on unseen dataset
    logging.info("Running comparison on unseen dataset...")
    comparison_unseen = HyperparameterTuning(X_unseen, y_unseen)
    comparison_unseen.run_comparison()
    comparison_unseen.save_results("Unseen dataset")

if __name__ == "__main__":
    main() 