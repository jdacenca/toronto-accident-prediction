"""
Batch testing script for the Decision Tree model.

This script allows you to test the trained decision tree model with multiple
predefined test cases and generates a report of the results.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any, Union
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import from the ModelTester class from test_model.py
from test_model import ModelTester
from utils.config import SERIALIZED_DIR, DATA_DIR, INSIGHTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_cases(num_cases: int = 5) -> pd.DataFrame:
    """Create test cases by randomly selecting from original dataset.
    
    Args:
        num_cases: Number of test cases to generate
        
    Returns:
        DataFrame with test cases
    """
    # Find dataset file
    try:
        data_file = DATA_DIR / 'TOTAL_KSI_6386614326836635957.csv'
        if not data_file.exists():
            data_files = list(DATA_DIR.glob('*.csv'))
            if not data_files:
                raise FileNotFoundError("No dataset files found in the data directory")
            data_file = data_files[0]
            
        # Load dataset
        df = pd.read_csv(data_file)
        
        # Randomly select entries
        if len(df) < num_cases:
            logging.warning(f"Dataset has fewer than {num_cases} entries, using all available entries")
            selected_df = df
        else:
            selected_df = df.sample(num_cases, random_state=42)
            
        return selected_df
    
    except Exception as e:
        logging.error(f"Error creating test cases: {e}")
        return pd.DataFrame()

def create_synthetic_test_cases(num_cases: int = 5) -> pd.DataFrame:
    """Create synthetic test cases by modifying existing data.
    
    Args:
        num_cases: Number of test cases to generate
        
    Returns:
        DataFrame with synthetic test cases
    """
    try:
        # First get some real cases
        real_cases = create_test_cases(num_cases)
        if real_cases.empty:
            return pd.DataFrame()
            
        # Make a copy to avoid modifying the original
        synthetic_cases = real_cases.copy()
        
        # Modify some features to create variations
        if 'TIME' in synthetic_cases.columns:
            # Randomly adjust time
            synthetic_cases['TIME'] = (synthetic_cases['TIME'].astype(float) + 
                                      np.random.randint(-3, 3, size=len(synthetic_cases)))
            # Ensure time is within valid range
            synthetic_cases['TIME'] = synthetic_cases['TIME'].apply(
                lambda x: x % 24 if x >= 0 else (24 + x) % 24)
                
        # Modify location slightly
        for col in ['LATITUDE', 'LONGITUDE']:
            if col in synthetic_cases.columns:
                # Add small random offset to coordinates
                synthetic_cases[col] = synthetic_cases[col] + np.random.normal(0, 0.01, size=len(synthetic_cases))
                
        # Modify some categorical variables
        categorical_cols = [col for col in synthetic_cases.columns 
                           if not pd.api.types.is_numeric_dtype(synthetic_cases[col])]
        
        # For a few categorical columns, swap values
        if categorical_cols and len(categorical_cols) > 1:
            for _ in range(min(3, len(categorical_cols))):
                col = np.random.choice(categorical_cols)
                values = synthetic_cases[col].dropna().unique()
                if len(values) > 1:
                    # For some rows, change the value to another possible value
                    rows_to_change = np.random.choice(
                        synthetic_cases.index, 
                        size=max(1, int(len(synthetic_cases) * 0.3)), 
                        replace=False
                    )
                    for idx in rows_to_change:
                        current_val = synthetic_cases.loc[idx, col]
                        other_vals = [v for v in values if v != current_val]
                        if other_vals:
                            synthetic_cases.loc[idx, col] = np.random.choice(other_vals)
        
        return synthetic_cases
    
    except Exception as e:
        logging.error(f"Error creating synthetic test cases: {e}")
        return pd.DataFrame()

def generate_report(results: List[Dict[str, Any]], test_cases: pd.DataFrame, output_dir: Path) -> None:
    """Generate a report of the batch test results.
    
    Args:
        results: List of prediction results
        test_cases: DataFrame with the test cases
        output_dir: Directory to save the report
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Add original data for reference
    combined_df = pd.concat([
        test_cases.reset_index(drop=True),
        results_df.reset_index(drop=True)
    ], axis=1)
    
    # Save combined results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"batch_test_results_{timestamp}.csv"
    combined_df.to_csv(csv_path, index=False)
    
    # Calculate summary statistics
    total_cases = len(results)
    fatal_count = sum(1 for r in results if r['prediction'] == 'FATAL')
    non_fatal_count = sum(1 for r in results if r['prediction'] == 'NON-FATAL')
    fatal_pct = (fatal_count / total_cases) * 100 if total_cases > 0 else 0
    non_fatal_pct = (non_fatal_count / total_cases) * 100 if total_cases > 0 else 0
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    avg_fatal_prob = np.mean([r['fatal_probability'] for r in results])
    
    # Generate summary report
    report_path = output_dir / f"batch_test_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("=========================================\n")
        f.write("  Toronto Accident Prediction Model     \n")
        f.write("  Batch Testing Report                  \n")
        f.write(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}          \n")
        f.write("=========================================\n\n")
        
        f.write(f"Total test cases: {total_cases}\n")
        f.write(f"Fatal accidents predicted: {fatal_count} ({fatal_pct:.2f}%)\n")
        f.write(f"Non-fatal accidents predicted: {non_fatal_count} ({non_fatal_pct:.2f}%)\n\n")
        
        f.write(f"Average prediction confidence: {avg_confidence:.4f}\n")
        f.write(f"Average fatal probability: {avg_fatal_prob:.4f}\n\n")
        
        f.write("Individual predictions:\n")
        f.write("-----------------------\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Test Case #{i+1}:\n")
            f.write(f"  Prediction: {result['prediction']}\n")
            f.write(f"  Confidence: {result['confidence']:.4f}\n")
            f.write(f"  Fatal Probability: {result['fatal_probability']:.4f}\n")
            f.write(f"  Non-Fatal Probability: {result['non_fatal_probability']:.4f}\n\n")
    
    # Create visualizations
    try:
        # Create a histogram of prediction probabilities
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='fatal_probability', bins=10, kde=True)
        plt.title('Distribution of Fatal Accident Probabilities')
        plt.xlabel('Probability of Fatal Accident')
        plt.ylabel('Count')
        plt.savefig(output_dir / f"probability_distribution_{timestamp}.png")
        plt.close()
        
        # Create a pie chart of predictions
        plt.figure(figsize=(8, 8))
        plt.pie([fatal_count, non_fatal_count], 
                labels=['Fatal', 'Non-Fatal'],
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'])
        plt.title('Prediction Distribution')
        plt.savefig(output_dir / f"prediction_distribution_{timestamp}.png")
        plt.close()
        
        # If actual outcomes are available (ACCLASS column), create confusion matrix
        if 'ACCLASS' in test_cases.columns:
            # Convert ACCLASS to binary (1 for FATAL, 0 otherwise)
            y_true = (test_cases['ACCLASS'] == 'FATAL').astype(int)
            y_pred = np.array([1 if r['prediction'] == 'FATAL' else 0 for r in results])
            
            if len(y_true) == len(y_pred):
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Non-Fatal', 'Fatal'],
                           yticklabels=['Non-Fatal', 'Fatal'])
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.savefig(output_dir / f"confusion_matrix_{timestamp}.png")
                plt.close()
    except Exception as e:
        logging.warning(f"Error creating visualizations: {e}")
    
    logging.info(f"Results saved to {output_dir}")
    logging.info(f"Report saved to {report_path}")
    logging.info(f"Data saved to {csv_path}")

def main():
    """Main function for batch testing the model."""
    parser = argparse.ArgumentParser(description="Batch test the decision tree model")
    
    parser.add_argument('--num-cases', type=int, default=10,
                      help="Number of test cases to generate (default: 10)")
    parser.add_argument('--synthetic', action='store_true',
                      help="Generate synthetic test cases instead of using real data")
    parser.add_argument('--input-csv', type=str, default=None,
                      help="Path to a CSV file containing test cases")
    parser.add_argument('--output-dir', type=str, 
                      default=str(INSIGHTS_DIR / 'batch_test_results'),
                      help="Directory to save the results")
    
    args = parser.parse_args()
    
    try:
        # Initialize model tester
        tester = ModelTester()
        
        # Get test cases
        if args.input_csv:
            # Use provided CSV file
            test_cases = pd.read_csv(args.input_csv)
            logging.info(f"Loaded {len(test_cases)} test cases from {args.input_csv}")
        elif args.synthetic:
            # Generate synthetic test cases
            test_cases = create_synthetic_test_cases(args.num_cases)
            logging.info(f"Generated {len(test_cases)} synthetic test cases")
        else:
            # Use real test cases
            test_cases = create_test_cases(args.num_cases)
            logging.info(f"Generated {len(test_cases)} test cases from real data")
        
        if test_cases.empty:
            logging.error("No test cases available. Exiting.")
            return
            
        # Make predictions
        logging.info("Making predictions...")
        results = tester.predict_batch(test_cases)
        
        # Generate report
        output_dir = Path(args.output_dir)
        generate_report(results, test_cases, output_dir)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()