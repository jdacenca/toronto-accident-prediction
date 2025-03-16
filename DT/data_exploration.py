import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import calendar
import folium
from folium.plugins import HeatMap
from datetime import datetime

def create_output_dirs():
    """Create necessary output directories if they don't exist"""
    dirs = [
        'insights/correlation',
        'insights/feature_importance',
        'insights/time_analysis',
        'insights/severity_analysis',
        'insights/seasonal_analysis',
        'insights/geographic_analysis',
        'insights/model_performance',
        'insights/dt_tuning'
    ]
    for dir_path in dirs:
        Path(f'{dir_path}').mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the accident data"""
    print("Loading data...")
    df = pd.read_csv('data/TOTAL_KSI_6386614326836635957.csv')
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data for analysis"""
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

def analyze_data(df):
    """Perform initial data analysis"""
    print("\n=== Data Analysis ===")
    print("\nData Info:")
    print(df.info())
    
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print("\nTarget Class Distribution:")
    print(df['ACCLASS'].value_counts())
    
    return missing_values

def create_full_correlation_matrices(df):
    """Create and save correlation matrices for all columns, including categorical ones"""
    print("\n=== Creating Full Correlation Matrices (Including Categorical) ===")
    
    # Create a copy of the dataframe
    df_encoded = df.copy()
    
    # Remove 'Property Damage O' records as they're not relevant for severity analysis
    df_encoded = df_encoded[df_encoded['ACCLASS'] != 'Property Damage O']
    
    # Dictionary to store original categories for each column
    category_mappings = {}
    
    # Handle datetime and special columns first
    df_encoded['Month'] = df_encoded['DATE'].dt.month
    df_encoded['DayOfWeek'] = df_encoded['DATE'].dt.dayofweek
    df_encoded = df_encoded.drop('DATE', axis=1)
    
    # Convert TimePeriod to numeric (use the order of periods as numeric values)
    time_periods = ['Night (0-5)', 'Morning (6-11)', 
                   'Afternoon (12-16)', 'Evening (17-20)', 
                   'Night (21-23)']
    df_encoded['TimePeriod'] = pd.Categorical(
        df_encoded['TimePeriod'], 
        categories=time_periods, 
        ordered=True
    ).codes
    
    # Add target column first (before encoding other categoricals)
    df_encoded['ACCLASS'] = (df_encoded['ACCLASS'] == 'Fatal').astype(float)
    
    # Drop string columns that shouldn't be part of correlation
    columns_to_drop = ['STREET1', 'STREET2', 'OFFSET']
    df_encoded = df_encoded.drop(columns=columns_to_drop, errors='ignore')
    
    # Encode all remaining categorical columns while preserving original names
    for column in df_encoded.select_dtypes(include=['object']).columns:
        # For binary columns (Yes/No)
        if set(df_encoded[column].dropna().unique()).issubset({'Yes', 'No'}):
            df_encoded[column] = (df_encoded[column] == 'Yes').astype(float)
        # For other categorical columns
        else:
            # Store original categories
            unique_values = df_encoded[column].dropna().unique()
            category_mappings[column] = {
                idx: cat for idx, cat in enumerate(unique_values)
            }
            # Use label encoding
            df_encoded[column] = pd.Categorical(df_encoded[column]).codes.astype(float)
    
    # Calculate correlations
    correlations = df_encoded.corr()
    
    # Save category mappings for reference
    with open('insights/correlation/category_mappings.txt', 'w') as f:
        f.write("Category Mappings for Encoded Features:\n")
        f.write("=====================================\n\n")
        for column, mapping in category_mappings.items():
            f.write(f"\n{column}:\n")
            for code, category in mapping.items():
                f.write(f"  {code}: {category}\n")
    
    # Save correlations to CSV
    correlations.to_csv('insights/correlation/full_feature_correlations.csv')
    
    # Create correlation plots
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
        plt.savefig(f'insights/correlation/full_correlation_matrix_{i+1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create specific correlation plot with target
    plt.figure(figsize=(15, 10))
    target_corr = correlations['ACCLASS'].sort_values(ascending=False)
    target_corr = target_corr.drop('ACCLASS')
    
    # Plot top 30 correlations
    top_30_corr = target_corr.head(30)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_30_corr.values, y=top_30_corr.index)
    plt.title('Top 30 Feature Correlations with Accidents')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('insights/correlation/full_target_correlations.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return target_corr, category_mappings

def analyze_time_patterns(df):
    """Analyze time-based patterns in accidents"""
    print("\n=== Analyzing Time Patterns ===")
    
    # Hourly Distribution of Severity
    plt.figure(figsize=(12, 6))
    hourly_severity = pd.crosstab(df['Hour'], df['INJURY'])
    hourly_severity.plot(kind='bar', stacked=True)
    plt.title('Injury Severity by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('insights/time_analysis/hourly_severity.png', bbox_inches='tight')
    plt.close()
    
    # Time Period Analysis by Season
    plt.figure(figsize=(12, 6))
    season_time = pd.crosstab([df['Season'], df['TimePeriod']], df['INJURY'])
    season_time.plot(kind='bar', stacked=True)
    plt.title('Injury Severity by Season and Time Period')
    plt.xlabel('Season - Time Period')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('insights/time_analysis/season_time_severity.png', bbox_inches='tight')
    plt.close()
    
    # Fatal Accidents Heatmap
    plt.figure(figsize=(12, 6))
    fatal_matrix = pd.crosstab(df['Hour'], df['Season'], 
                              values=df['FATAL_NO'].notna(), 
                              aggfunc='sum')
    sns.heatmap(fatal_matrix, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Fatal Accidents Heatmap (Hour vs Season)')
    plt.xlabel('Season')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig('insights/time_analysis/fatal_heatmap.png', bbox_inches='tight')
    plt.close()

def analyze_severity_patterns(df):
    """Analyze accident severity patterns"""
    print("\n=== Analyzing Severity Patterns ===")
    
    # Injury Distribution by Season
    plt.figure(figsize=(12, 6))
    seasonal_injury = pd.crosstab(df['Season'], df['INJURY'])
    seasonal_injury.plot(kind='bar', stacked=True)
    plt.title('Injury Severity by Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('insights/severity_analysis/seasonal_injury.png', bbox_inches='tight')
    plt.close()
    
    # Severity Distribution by Road Condition
    plt.figure(figsize=(12, 6))
    condition_severity = pd.crosstab(df['RDSFCOND'], df['INJURY'])
    condition_severity.plot(kind='bar', stacked=True)
    plt.title('Injury Severity by Road Condition')
    plt.xlabel('Road Condition')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Injury Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('insights/severity_analysis/condition_severity.png', bbox_inches='tight')
    plt.close()

def analyze_seasonal_patterns(df):
    """Analyze seasonal patterns in accidents"""
    print("\n=== Analyzing Seasonal Patterns ===")
    
    # Monthly accidents
    plt.figure(figsize=(12, 6))
    monthly_accidents = df.groupby(df['DATE'].dt.month).size()
    monthly_accidents.plot(kind='bar')
    plt.title('Accidents by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Accidents')
    plt.xticks(range(12), calendar.month_abbr[1:], rotation=45)
    plt.tight_layout()
    plt.savefig('insights/seasonal_analysis/monthly_accidents.png', bbox_inches='tight')
    plt.close()
    
    # Seasonal patterns by neighborhood
    plt.figure(figsize=(12, 6))
    seasonal_hood = pd.crosstab(df['Season'], df['NEIGHBOURHOOD_158'])
    top_neighborhoods = seasonal_hood.sum().nlargest(5).index
    
    seasonal_hood[top_neighborhoods].plot(kind='bar', stacked=True)
    plt.title('Seasonal Accident Distribution by Top 5 Neighborhoods')
    plt.xlabel('Season')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Neighborhood', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('insights/seasonal_analysis/seasonal_neighborhood.png', bbox_inches='tight')
    plt.close()

def analyze_geographic_distribution(df):
    """Analyze geographic distribution of accidents"""
    print("\n=== Analyzing Geographic Distribution ===")
    
    # Create base map
    center_lat = df['LATITUDE'].mean()
    center_lon = df['LONGITUDE'].mean()
    accident_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add heatmap layer
    locations = df[['LATITUDE', 'LONGITUDE']].values.tolist()
    HeatMap(locations).add_to(accident_map)
    
    # Save map
    accident_map.save('insights/geographic_analysis/accident_heatmap.html')

def save_analysis_report(df, missing_values):
    """Save analysis results to markdown file"""
    report = f"""# Comprehensive Data Analysis Report

## Dataset Overview
- Total Records: {len(df)}
- Total Features: {df.shape[1]}
- Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB
- Time Range: {df['DATE'].min()} to {df['DATE'].max()}

## Target Class Distribution
```
{df['ACCLASS'].value_counts().to_string()}
```

## Missing Values Summary
```
{missing_values[missing_values > 0].to_string()}
```

## Time-Based Analysis
- Peak accident hours: {df.groupby('Hour').size().nlargest(3).index.tolist()}
- Most dangerous time period: {df.groupby('TimePeriod').size().idxmax()}
- Season with most accidents: {df.groupby('Season').size().idxmax()}

## Severity Analysis
- Total fatal accidents: {df['FATAL_NO'].notna().sum()}
- Most common injury type: {df['INJURY'].mode()[0]}
- Most dangerous road condition: {df.groupby('RDSFCOND')['FATAL_NO'].count().idxmax()}

## Geographic Analysis
- Number of unique neighborhoods: {df['NEIGHBOURHOOD_158'].nunique()}
- Top 5 neighborhoods by accident count:
```
{df['NEIGHBOURHOOD_158'].value_counts().head().to_string()}
```

## Analysis Insights
1. Features with correlation > 0.1 with Accidents should be considered important
2. Features with correlation < 0.05 might be candidates for removal
3. Missing values in Yes/No columns will be filled with 'No'
4. Seasonal patterns show significant variation in accident rates
5. Geographic distribution reveals accident hotspots
6. Time of day has strong correlation with accident severity

## Output Directory Structure
All analysis outputs are organized in the following structure:

```
insights/
├── correlation/           # Correlation analysis and matrices
├── time_analysis/        # Time-based patterns and trends
├── severity_analysis/    # Severity distribution analysis
├── seasonal_analysis/    # Seasonal patterns
├── geographic_analysis/  # Geographic distribution
├── performance/    # Model evaluation metrics
└── dt_tuning/           # Decision tree tuning analysis results
```

## Generated Visualizations
1. Correlation Analysis (insights/correlation/):
   - correlation_matrix_*.png
   - feature_correlations.csv

2. Time Analysis (insights/time_analysis/):
   - hourly_severity.png
   - season_time_severity.png
   - fatal_heatmap.png

3. Severity Analysis (insights/severity_analysis/):
   - seasonal_injury.png
   - condition_severity.png

4. Seasonal Analysis (insights/seasonal_analysis/):
   - monthly_accidents.png
   - seasonal_neighborhood.png

5. Geographic Analysis (insights/geographic_analysis/):
   - accident_heatmap.html

6. Model Performance (insights/performance/):
   - confusion_matrix.png
   - roc_curve.png
   - precision_recall_curve.png

7. Decision Tree Tuning (insights/dt_tuning/):
   - main_dataset_results.csv
   - main_dataset_results.md
   - unseen_dataset_results.csv
   - unseen_dataset_results.md
"""
    
    with open('insights/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    """Main function to run all analyses"""
    create_output_dirs()
    df = load_data()
    df = preprocess_data(df)
    
    missing_values = analyze_data(df)
    full_correlations, category_mappings = create_full_correlation_matrices(df)
    
    analyze_time_patterns(df)
    analyze_severity_patterns(df)
    analyze_seasonal_patterns(df)
    analyze_geographic_distribution(df)
    
    save_analysis_report(df, missing_values)
    print("\nAnalysis completed! Check the 'insights' directory for results.")

if __name__ == "__main__":
    main() 