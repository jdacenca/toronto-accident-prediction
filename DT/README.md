# Toronto Traffic Accident Prediction

## Project Overview

This project analyzes traffic accidents in Toronto to predict accident severity and identify key factors contributing to fatal accidents. The model uses historical accident data to classify accidents as either Fatal or Non-Fatal.

## Features

- Comprehensive data exploration and visualization
- Decision Tree hyperparameter tuning and sampling strategy analysis
- Feature importance analysis
- Temporal and seasonal pattern analysis
- Geographic distribution analysis
- Unseen data validation

## Project Structure

```
├── data/                      # Data directory
│   ├── TOTAL_KSI_*.csv       # Raw accident data
│   └── unseen_data.csv       # Last 10 rows for final testing
├── insights/                  # Analysis outputs
│   ├── analysis_report.md    # Comprehensive analysis report
│   ├── correlation/          # Correlation matrices and analysis
│   ├── tuning/           # Decision tree tuning analysis results
│   ├── geographic_analysis/  # Geographic distribution
│   ├── performance/         # Model performance metrics and Feature importance analysis
│   ├── seasonal_analysis/   # Seasonal patterns
│   ├── serialized_artifacts/ # Saved model artifacts
│   ├── severity_analysis/   # Severity distribution analysis
│   └── time_analysis/      # Temporal pattern analysis
├── main.py                   # Main execution script
├── decision_tree_tuning.py   # Decision Tree optimization and sampling analysis
├── data_exploration.py       # Data exploration functions
├── preprocessing_pipeline.py  # Data preprocessing pipeline
├── data_cleaning.md          # Data cleaning documentation
├── feature_engineering_doc.md # Feature engineering documentation
└── README.md                 # Project documentation
```

## Decision Tree Analysis

The project focuses on optimizing Decision Tree models through:

1. Hyperparameter Tuning

   - Basic Decision Tree baseline
   - Gini impurity optimization
   - Entropy criterion analysis
   - Class weight balancing

2. Sampling Strategy Evaluation

   - Original data distribution
   - Random oversampling
   - Random undersampling
   - SMOTE (Synthetic Minority Over-sampling Technique)

3. Performance Analysis
   - Training vs Testing accuracy comparison
   - Precision, Recall, and F1-Score metrics
   - Impact of different sampling methods
   - Model generalization assessment

## Key Features Analyzed

- Temporal patterns (time of day, day of week, seasonality)
- Road conditions and environmental factors
- Geographic location
- Vehicle types and characteristics
- Driver and participant information
- Injury severity levels

## Model Evaluation

Models are evaluated using:

- Accuracy (Training and Testing)
- Precision
- Recall
- F1 Score
- Impact of sampling strategies
- Performance on imbalanced data

## Unseen Data Testing

- Last 10 rows of the dataset are reserved for final testing
- Models are trained on the remaining data
- Final evaluation is performed on both test set and unseen data
- Results are compared to ensure model generalization

## Generated Insights

1. Correlation Analysis

   - Feature correlations with Accidents
   - Inter-feature correlation matrices
   - Key factor identification

2. Time-Based Analysis

   - Peak accident hours
   - Seasonal patterns
   - Time period risk assessment

3. Severity Analysis

   - Injury distribution patterns
   - Road condition impact
   - Environmental factor effects

4. Geographic Analysis
   - Accident hotspots
   - Neighborhood-wise distribution
   - Seasonal geographic patterns

## How to Run

1. Setup Environment:

```bash
pip install -r requirements.txt
```

2. Run Data Exploration:

```bash
python data_exploration.py
```

3. Train and Evaluate Models:

```bash
python main.py
```

4. Run Decision Tree Analysis:

```bash
python decision_tree_tuning.py
```

## Output Files

1. Analysis Reports

   - `insights/analysis_report.md`: Comprehensive data analysis
   - `insights/tuning/main_dataset_results.csv`: Decision tree tuning metrics
   - `insights/tuning/unseen_dataset_results.csv`: Unseen data evaluation
   - `insights/performance/*.txt`: Detailed classification reports

2. Visualizations

   - Correlation matrices
   - Feature importance plots
   - Time-based analysis charts
   - Geographic heatmaps
   - Model performance curves

3. Model Artifacts
   - Trained model files
   - Preprocessing pipeline
   - Feature scalers
   - Selected features list

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- folium
- imbalanced-learn

## Contributors

- Hans-Randy Masamba
