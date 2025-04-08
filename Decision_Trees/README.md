# Toronto Traffic Accident Prediction

## Project Overview

This project analyzes traffic accidents in Toronto to predict accident severity and identify key factors contributing to fatal accidents. The model uses historical accident data to classify accidents as either Fatal or Non-Fatal.

## Project Structure

```
├── data/                      # Data directory
│   ├── TOTAL_KSI_*.csv       # Raw accident data
│   └── unseen_data.csv       # Last 10 rows for final testing
├── insights/                  # Analysis outputs
│   ├── analysis_report.md    # Comprehensive analysis report
│   ├── correlation/          # Correlation matrices and analysis
│   ├── tuning/               # Decision tree tuning analysis results
│   ├── geographic_analysis/  # Geographic distribution
│   ├── performance/          # Model performance metrics and Feature importance analysis
│   ├── seasonal_analysis/    # Seasonal patterns
│   ├── serialized_artifacts/ # Saved model artifacts
│   ├── severity_analysis/    # Severity distribution analysis
│   ├── time_analysis/        # Temporal pattern analysis
│   └── unseen_testing/       # Results from testing on unseen data (last 10 rows)
├── utils/                     # Utility scripts
│   ├── config.py             # Configuration settings
│   ├── data_cleaner.py       # Data cleaning utilities
│   ├── data_explorer.py      # Data exploration utilities
│   ├── evaluation.py         # Model evaluation utilities
│   ├── feature_engineer.py   # Feature engineering utilities
│   ├── hyperparameter_tuning.py # Hyperparameter tuning utilities
│   ├── pipeline.py           # Preprocessing pipeline
│   └── visualization.py      # Visualization utilities
├── data_exploration.py        # Script for data exploration
├── decision_tree_tuning.py    # Script for decision tree hyperparameter tuning
├── main.py                    # Main script for training and evaluation
└── README.md                  # Project documentation
```

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- folium
- imbalanced-learn

## Prerequisites

- Python 3.8 or higher
- Install dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Submit a pull request.

## Known Issues

- The dataset is imbalanced, which may affect model performance.

## Additional Outputs

### Serialized Artifacts

- `serialized_artifacts/decision_tree_model.pkl`: Trained Decision Tree model.
- `serialized_artifacts/preprocessing_pipeline.pkl`: Preprocessing pipeline for data transformation.

### Insights Directory

The `insights/` directory contains detailed analysis outputs:

- **Correlation Analysis**: Full correlation matrices and feature-target correlations.
- **Geographic Analysis**: Accident heatmaps.
- **Performance Metrics**: Classification reports, confusion matrices, ROC curves, and SHAP visualizations.
- **Seasonal Analysis**: Monthly and seasonal accident patterns.
- **Severity Analysis**: Injury severity distributions and environmental impacts.
- **Time Analysis**: Temporal patterns and severity heatmaps.
- **Tuning Results**: Hyperparameter tuning results for various sampling strategies.
- **Unseen Testing**: Results from testing on unseen data with different sampling methods.

## Contributors

- Hans-Randy Masamba
