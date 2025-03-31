"""Data exploration utilities for accident data analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import logging
from typing import Tuple
from .config import INSIGHTS_DIR

class DataExplorer:
    """Class for exploring and analyzing accident data."""
    
    def __init__(self, data_path: str):
        """Initialize the DataExplorer.
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.df: pd.DataFrame = None
        self.missing_values: pd.Series = None
        self.category_mappings: dict = {}
        self._setup_directories()
        self._setup_logging()
    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        dirs = [
            'correlation',
            'time_analysis',
            'severity_analysis',
            'seasonal_analysis',
            'geographic_analysis',
            'performance',
            'tuning'
        ]
        for dir_path in dirs:
            (INSIGHTS_DIR / dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self) -> pd.DataFrame:
        """Load the accident data.
        
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        logging.info("Loading data...")
        self.df = pd.read_csv(self.data_path)
        logging.info(f"Dataset shape: {self.df.shape}")
        
        # Preprocess the data
        self.df = self._preprocess_data(self.df)
        return self.df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for analysis.
        
        Args:
            df: Raw dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df = df.copy()
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Add Hour column for time analysis
        df['Hour'] = df['TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
        
        # Add Season column
        df['Season'] = df['DATE'].dt.month.map(
            lambda m: 'Winter' if m in [12,1,2]
            else 'Spring' if m in [3,4,5]
            else 'Summer' if m in [6,7,8]
            else 'Fall'
        )
        
        # Add TimePeriod column
        df['TimePeriod'] = pd.cut(
            df['Hour'], 
            bins=[-1, 5, 11, 16, 20, 23],
            labels=['Night (0-5)', 'Morning (6-11)', 
                   'Afternoon (12-16)', 'Evening (17-20)', 
                   'Night (21-23)']
        )
        
        return df
    
    def analyze_basic_stats(self) -> pd.Series:
        """Perform initial data analysis.
        
        Returns:
            pd.Series: Missing value counts
        """
        logging.info("\n=== Data Analysis ===")
        logging.info("\nData Info:")
        logging.info(self.df.info())
        
        self.missing_values = self.df.isnull().sum()
        logging.info("\nMissing Values:")
        logging.info(self.missing_values[self.missing_values > 0])
        
        logging.info("\nTarget Class Distribution:")
        logging.info(self.df['ACCLASS'].value_counts())
        
        return self.missing_values
    
    def analyze_correlations(self) -> Tuple[pd.Series, dict]:
        """Create and save correlation matrices.
        
        Returns:
            Tuple[pd.Series, dict]: Target correlations and category mappings
        """
        logging.info("\n=== Creating Correlation Matrices ===")
        
        # Create a copy of the dataframe
        df_encoded = self.df.copy()
        
        # Remove 'Property Damage O' records
        df_encoded = df_encoded[df_encoded['ACCLASS'] != 'Property Damage O']
        
        # Drop derived columns
        df_encoded = df_encoded.drop(columns=['TimePeriod', 'Season', 'Hour'])
        
        # Add target column
        df_encoded['ACCLASS'] = (df_encoded['ACCLASS'] == 'Fatal').astype(float)

        # Encode categorical columns
        for column in df_encoded.select_dtypes(include=['object']).columns:
            if set(df_encoded[column].dropna().unique()).issubset({'Yes', 'No'}):
                df_encoded[column] = (df_encoded[column] == 'Yes').astype(float)
            else:
                unique_values = df_encoded[column].dropna().unique()
                self.category_mappings[column] = {
                    idx: cat for idx, cat in enumerate(unique_values)
                }
                df_encoded[column] = pd.Categorical(df_encoded[column]).codes.astype(float)
        
        # Calculate correlations
        correlations = df_encoded.corr()
        
        # Save category mappings
        self._save_category_mappings()
        
        # Save correlations
        correlations.to_csv(INSIGHTS_DIR / 'correlation/full_feature_correlations.csv')
        
        # Create correlation plots
        self._plot_correlation_matrices(correlations)
        
        # Get target correlations
        target_corr = correlations['ACCLASS'].sort_values(ascending=False)
        target_corr = target_corr.drop('ACCLASS')
        
        # Plot target correlations
        self._plot_target_correlations(target_corr)
        
        return target_corr, self.category_mappings
    
    def _save_category_mappings(self) -> None:
        """Save category mappings to file."""
        with open(INSIGHTS_DIR / 'correlation/category_mappings.txt', 'w') as f:
            f.write("Category Mappings for Encoded Features:\n")
            f.write("=====================================\n\n")
            for column, mapping in self.category_mappings.items():
                f.write(f"\n{column}:\n")
                for code, category in mapping.items():
                    f.write(f"  {code}: {category}\n")
    
    def _plot_correlation_matrices(self, correlations: pd.DataFrame) -> None:
        """Plot correlation matrices.
        
        Args:
            correlations: Correlation matrix
        """
        n_cols = len(correlations.columns)
        max_cols_per_plot = 20
        n_splits = (n_cols + max_cols_per_plot - 1) // max_cols_per_plot
        
        for i in range(n_splits):
            start_idx = i * max_cols_per_plot
            end_idx = min((i + 1) * max_cols_per_plot, n_cols)
            
            plt.figure(figsize=(24, 20))
            sns.heatmap(
                correlations.iloc[start_idx:end_idx, start_idx:end_idx],
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True
            )
            plt.title(f'Full Correlation Matrix Part {i+1}')
            plt.tight_layout()
            plt.savefig(INSIGHTS_DIR / f'correlation/full_correlation_matrix_{i+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_target_correlations(self, target_corr: pd.Series) -> None:
        """Plot target correlations.
        
        Args:
            target_corr: Series of correlations with target
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x=target_corr.values, y=target_corr.index)
        plt.title('Feature Correlations with Accidents')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(INSIGHTS_DIR / 'correlation/full_target_correlations.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_time_patterns(self) -> None:
        """Analyze time-based patterns in accidents."""
        logging.info("\n=== Analyzing Time Patterns ===")
        
        # Hourly Distribution of Severity
        plt.figure(figsize=(12, 6))
        hourly_severity = pd.crosstab(self.df['Hour'], self.df['INJURY'])
        hourly_severity.plot(kind='bar', stacked=True)
        plt.title('Injury Severity by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Accidents')
        plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(INSIGHTS_DIR / 'time_analysis/hourly_severity.png', bbox_inches='tight')
        plt.close()
        
        # Time Period Analysis by Season
        plt.figure(figsize=(12, 6))
        season_time = pd.crosstab([self.df['Season'], self.df['TimePeriod']], self.df['INJURY'])
        season_time.plot(kind='bar', stacked=True)
        plt.title('Injury Severity by Season and Time Period')
        plt.xlabel('Season - Time Period')
        plt.ylabel('Number of Accidents')
        plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(INSIGHTS_DIR / 'time_analysis/season_time_severity.png', bbox_inches='tight')
        plt.close()
        
        # Fatal Accidents Heatmap
        plt.figure(figsize=(12, 6))
        fatal_matrix = pd.crosstab(self.df['Hour'], self.df['Season'], 
                                 values=self.df['FATAL_NO'].notna(), 
                                 aggfunc='sum')
        sns.heatmap(fatal_matrix, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title('Fatal Accidents Heatmap (Hour vs Season)')
        plt.xlabel('Season')
        plt.ylabel('Hour of Day')
        plt.tight_layout()
        plt.savefig(INSIGHTS_DIR / 'time_analysis/fatal_heatmap.png', bbox_inches='tight')
        plt.close()
    
    def analyze_severity_patterns(self) -> None:
        """Analyze accident severity patterns."""
        logging.info("\n=== Analyzing Severity Patterns ===")
        
        # Injury Distribution by Season
        plt.figure(figsize=(12, 6))
        seasonal_injury = pd.crosstab(self.df['Season'], self.df['INJURY'])
        seasonal_injury.plot(kind='bar', stacked=True)
        plt.title('Injury Severity by Season')
        plt.xlabel('Season')
        plt.ylabel('Number of Accidents')
        plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(INSIGHTS_DIR / 'severity_analysis/seasonal_injury.png', bbox_inches='tight')
        plt.close()
        
        # Severity Distribution by Road Condition
        plt.figure(figsize=(12, 6))
        condition_severity = pd.crosstab(self.df['RDSFCOND'], self.df['INJURY'])
        condition_severity.plot(kind='bar', stacked=True)
        plt.title('Injury Severity by Road Condition')
        plt.xlabel('Road Condition')
        plt.ylabel('Number of Accidents')
        plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(INSIGHTS_DIR / 'severity_analysis/condition_severity.png', bbox_inches='tight')
        plt.close()
    
    def analyze_seasonal_patterns(self) -> None:
        """Analyze seasonal patterns in accidents."""
        logging.info("\n=== Analyzing Seasonal Patterns ===")
        
        # Monthly accidents
        plt.figure(figsize=(12, 6))
        monthly_accidents = self.df.groupby(self.df['DATE'].dt.month).size()
        monthly_accidents.plot(kind='bar')
        plt.title('Accidents by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Accidents')
        plt.xticks(range(12), calendar.month_abbr[1:], rotation=45)
        plt.tight_layout()
        plt.savefig(INSIGHTS_DIR / 'seasonal_analysis/monthly_accidents.png', bbox_inches='tight')
        plt.close()
    
    def save_analysis_report(self) -> None:
        """Save a comprehensive analysis report."""
        report_path = INSIGHTS_DIR / 'analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Accident Data Analysis Report\n\n")
            
            f.write("## Dataset Overview\n")
            f.write(f"- Total records: {len(self.df):,}\n")
            f.write(f"- Time period: {self.df['DATE'].min()} to {self.df['DATE'].max()}\n")
            f.write(f"- Features: {len(self.df.columns)}\n\n")
            
            f.write("## Missing Values\n")
            if self.missing_values is not None:
                missing = self.missing_values[self.missing_values > 0]
                f.write("Features with missing values:\n")
                for col, count in missing.items():
                    percentage = (count / len(self.df)) * 100
                    f.write(f"- {col}: {count:,} ({percentage:.2f}%)\n")
            
            f.write("\n## Class Distribution\n")
            class_dist = self.df['ACCLASS'].value_counts()
            for cls, count in class_dist.items():
                percentage = (count / len(self.df)) * 100
                f.write(f"- {cls}: {count:,} ({percentage:.2f}%)\n")
            
            f.write("\n## Generated Visualizations\n")
            f.write("### Correlation Analysis\n")
            f.write("- Full correlation matrices\n")
            f.write("- Target correlation analysis\n")
            
            f.write("\n### Time Analysis\n")
            f.write("- Hourly severity distribution\n")
            f.write("- Seasonal time patterns\n")
            f.write("- Fatal accidents heatmap\n")
            
            f.write("\n### Severity Analysis\n")
            f.write("- Seasonal injury patterns\n")
            f.write("- Road condition impact\n")
            
            f.write("\n### Seasonal Analysis\n")
            f.write("- Monthly accident distribution\n")
            
            logging.info(f"Analysis report saved to {report_path}")
    
    def run_full_analysis(self) -> None:
        """Run all analysis steps."""
        self.load_data()
        self.analyze_basic_stats()
        self.analyze_correlations()
        self.analyze_time_patterns()
        self.analyze_severity_patterns()
        self.analyze_seasonal_patterns()
        self.save_analysis_report() 