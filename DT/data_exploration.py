"""Script for exploring accident data using the DataExplorer utility."""

from utils.data_explorer import DataExplorer
from utils.config import DATA_DIR

def main():
    """Main execution function."""
    # Initialize data explorer
    data_path = DATA_DIR / 'TOTAL_KSI_6386614326836635957.csv'
    explorer = DataExplorer(str(data_path))
    
    # Run full analysis
    explorer.run_full_analysis()

if __name__ == "__main__":
    main() 