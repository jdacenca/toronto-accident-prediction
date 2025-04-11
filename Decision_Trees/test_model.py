"""
Test script for evaluating the Decision Tree model with new data entries.

This script loads the trained Decision Tree model and preprocessing pipeline,
and allows testing with new data entries provided either via manual input,
a CSV file, or sample test cases.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, Any
from utils.config import SERIALIZED_DIR, DATA_DIR
from sklearn.tree import DecisionTreeClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTester:
    """Class for testing the trained Decision Tree model with new data."""
    
    def __init__(self, model_path: Union[str, Path] = None, pipeline_path: Union[str, Path] = None):
        """Initialize the model tester.
        
        Args:
            model_path: Path to the saved model file. Defaults to SERIALIZED_DIR / 'decision_tree_model.pkl'.
            pipeline_path: Path to the saved preprocessing pipeline file. 
                          Defaults to SERIALIZED_DIR / 'preprocessing_pipeline.pkl'.
        """
        # Default paths
        if model_path is None:
            model_path = SERIALIZED_DIR / 'decision_tree_model.pkl'
        if pipeline_path is None:
            pipeline_path = SERIALIZED_DIR / 'preprocessing_pipeline.pkl'
            
        # Convert to Path objects if they are strings
        if isinstance(model_path, str):
            model_path = Path(model_path)
        if isinstance(pipeline_path, str):
            pipeline_path = Path(pipeline_path)
            
        # Check if model and pipeline files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
            
        logging.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        logging.info(f"Loading preprocessing pipeline from {pipeline_path}")
        self.pipeline = joblib.load(pipeline_path)
        
        # Load a sample from the dataset to get column structure
        sample_data_path = DATA_DIR / 'TOTAL_KSI_6386614326836635957.csv'
        if not sample_data_path.exists():
            logging.warning(f"Sample data file not found: {sample_data_path}. Using any CSV file in the data directory.")
            # Try to find any CSV file in the data directory
            csv_files = list(DATA_DIR.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the data directory.")
            sample_data_path = csv_files[0]
            
        self.sample_df = pd.read_csv(sample_data_path)
        self.columns = self.sample_df.columns.tolist()
        
        logging.info(f"Model and pipeline loaded successfully.")
        
    def _format_prediction(self, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Union[str, float]]:
        """Format prediction results.
        
        Args:
            y_pred: Binary prediction (0 or 1)
            y_prob: Probability of the positive class (fatal accident)
            
        Returns:
            Dict with formatted prediction results
        """
        return {
            'prediction': 'FATAL' if y_pred[0] == 1 else 'NON-FATAL',
            'confidence': float(max(y_prob[0])),
            'fatal_probability': float(y_prob[0][1]) if len(y_prob[0]) > 1 else float(y_prob[0]),
            'non_fatal_probability': float(y_prob[0][0]) if len(y_prob[0]) > 1 else 1 - float(y_prob[0])
        }

    
    def predict_single(self, data: dict[str, Any]) -> dict[str, Union[str, float]]:
        """Make a prediction for a single data entry.
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Dict with prediction results
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        for col in self.columns:
            if col not in df.columns:
                df[col] = np.nan
                
        # Apply preprocessing pipeline
        X = self.pipeline.transform(df)
        
        # Make prediction
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        
        return self._format_prediction(y_pred, y_prob)
    
    def predict_batch(self, data: pd.DataFrame) -> list[dict[str, Union[str, float]]]:
        """Make predictions for a batch of data.
        
        Args:
            data: DataFrame with feature values
            
        Returns:
            List of dicts with prediction results
        """
        # Ensure all required columns are present
        for col in self.columns:
            if col not in data.columns:
                data[col] = np.nan
                
        # Apply preprocessing pipeline
        X = self.pipeline.transform(data)
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        
        results = []
        for i in range(len(y_pred)):
            results.append(self._format_prediction(
                y_pred[i].reshape(1, -1),
                y_prob[i].reshape(1, -1)
            ))
        return results
    
    def predict_from_csv(self, file_path: Union[str, Path]) -> list[dict[str, Union[str, float]]]:
        """Make predictions for data from a CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of dicts with prediction results
        """
        # Convert to Path object if it's a string
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        # Load data
        data = pd.read_csv(file_path)
        
        return self.predict_batch(data)
    
    def create_sample_entry(self) -> dict[str, Any]:
        """Create a sample data entry based on the columns in the dataset.
        
        Returns:
            Dict with sample values for each column
        """
        # Create a sample entry with median values for numeric columns and mode for categorical
        sample_entry = {}
        
        for col in self.sample_df.columns:
            if col == 'ACCLASS':
                continue  # Skip the target column
                
            if pd.api.types.is_numeric_dtype(self.sample_df[col]):
                # For numeric columns, use median
                sample_entry[col] = self.sample_df[col].median()
            else:
                # For categorical columns, use mode
                mode_value = self.sample_df[col].mode()[0]
                sample_entry[col] = mode_value if not pd.isna(mode_value) else "UNKNOWN"
                
        return sample_entry

def main():
    """Main function for testing the model."""
    print("\n====== Toronto Accident Severity Prediction Model Tester ======\n")
    
    try:
        # Initialize model tester
        tester = ModelTester()
        
        # Print model info
        print(f"Model type: {type(tester.model).__name__}")
        print(f"Pipeline steps: {[step[0] for step in tester.pipeline.steps]}\n")
        
        print("\nTest Options:")
        print("1. Test with sample data")
        print("2. Test with CSV file")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            # Test with sample data
            sample_entry = tester.create_sample_entry()
            print("\nSample Entry:")
            for key, value in sample_entry.items():
                print(f"  - {key}: {value}")
                
            # Make prediction
            result = tester.predict_single(sample_entry)
            
            print("\nPrediction Result:")
            print(f"  - Predicted Class: {result['prediction']}")
            print(f"  - Confidence: {result['confidence']:.4f}")
            print(f"  - Fatal Accident Probability: {result['fatal_probability']:.4f}")
            print(f"  - Non-Fatal Accident Probability: {result['non_fatal_probability']:.4f}")
            
        elif choice == '2':
            # Test with CSV file
            file_path = input("\nEnter path to CSV file: ")
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return
                
            results = tester.predict_from_csv(file_path)
            
            print(f"\nPredicted {len(results)} entries:")
            for i, result in enumerate(results[:10]):  # Show only first 10 results
                print(f"\nEntry {i+1}:")
                print(f"  - Predicted Class: {result['prediction']}")
                print(f"  - Confidence: {result['confidence']:.4f}")
                print(f"  - Fatal Accident Probability: {result['fatal_probability']:.4f}")
            
            if len(results) > 10:
                print(f"\n... and {len(results) - 10} more entries.")
                
            # Calculate summary statistics
            fatal_count = sum(1 for r in results if r['prediction'] == 'FATAL')
            non_fatal_count = sum(1 for r in results if r['prediction'] == 'NON-FATAL')
            
            print("\nSummary Statistics:")
            print(f"  - Total Entries: {len(results)}")
            print(f"  - Predicted Fatal: {fatal_count} ({fatal_count/len(results)*100:.2f}%)")
            print(f"  - Predicted Non-Fatal: {non_fatal_count} ({non_fatal_count/len(results)*100:.2f}%)")
            
        elif choice == '3':
            print("\nExiting...")
            return
        else:
            print("\nInvalid choice. Please run the script again and select a valid option.")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()