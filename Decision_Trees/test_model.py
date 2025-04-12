"""
Test script for evaluating the Decision Tree model with new data entries.

This script loads the trained Decision Tree model and preprocessing pipeline,
and allows testing with new data entries provided either via manual input,
a CSV file, or sample test cases.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, Any
from utils.config import SERIALIZED_DIR, TARGET

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
                
        # Apply preprocessing pipeline with safe transform
        df = self.pipeline.transform(data)
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        # Make predictions
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
                
        # Apply preprocessing pipeline with safe transform
        df = self.pipeline.transform(data)
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
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
        # Drop 'ACCLASS' column if it exists
        data.drop(columns=['ACCLASS'], errors='ignore', inplace=True)
        return self.predict_batch(data)

def main():
    """Main function for testing the model."""
    parser = argparse.ArgumentParser(description="Batch test the decision tree model")
    
    parser.add_argument('--input-csv', type=str, default=None, help="Path to a CSV file containing test cases")
    
    args = parser.parse_args()
    
    try:
        # Initialize model tester
        tester = ModelTester()
        
        # Get test cases
        if args.input_csv:
            # Use provided CSV file
            test_cases = pd.read_csv(args.input_csv)
            logging.info(f"Loaded {len(test_cases)} test cases from {args.input_csv}")
        else:
            logging.error("No test cases available. Exiting.")
            return
            
        # Make predictions
        logging.info("Making predictions...")
        results = tester.predict_batch(test_cases)
            
        # Calculate summary statistics
        fatal_count = sum(1 for r in results if r['prediction'] == 'FATAL')
        non_fatal_count = sum(1 for r in results if r['prediction'] == 'NON-FATAL')
        
        print("\nSummary Statistics:")
        print(f"  - Total Entries: {len(results)}")
        print(f"  - Predicted Fatal: {fatal_count} ({fatal_count/len(results)*100:.2f}%)")
        print(f"  - Predicted Non-Fatal: {non_fatal_count} ({non_fatal_count/len(results)*100:.2f}%)")
        print("\nDetailed Predictions:")
        for i, result in enumerate(results):
            print(f"Entry {i+1}: {result}")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()