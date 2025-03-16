# Comprehensive Data Analysis Report

## Dataset Overview

- Total Records: 18957
- Total Features: 57
- Memory Usage: 8.12 MB
- Time Range: 2006-01-01 10:00:00 to 2023-12-29 10:00:00

## Target Class Distribution

```
ACCLASS
Non-Fatal Injury     16268
Fatal                 2670
Property Damage O       18
```

## Missing Values Summary

```
ACCNUM            4930
STREET2           1706
OFFSET           15137
ROAD_CLASS         486
DISTRICT           229
ACCLOC            5456
TRAFFCTL            75
VISIBILITY          24
LIGHT                4
RDSFCOND            29
ACCLASS              1
IMPACTYPE           27
INVTYPE             16
INJURY            8897
FATAL_NO         18087
INITDIR           5277
VEHTYPE           3487
MANOEUVER         7953
DRIVACT           9289
DRIVCOND          9291
PEDTYPE          15728
PEDACT           15730
PEDCOND          15711
CYCLISTYPE       18152
CYCACT           18155
CYCCOND          18157
PEDESTRIAN       11269
CYCLIST          16971
AUTOMOBILE        1727
MOTORCYCLE       17273
TRUCK            17788
TRSN_CITY_VEH    17809
EMERG_VEH        18908
PASSENGER        11774
SPEEDING         16263
AG_DRIV           9121
REDLIGHT         17380
ALCOHOL          18149
DISABILITY       18464
```

## Time-Based Analysis

- Peak accident hours: [18, 17, 15]
- Most dangerous time period: Afternoon (12-16)
- Season with most accidents: Summer

## Severity Analysis

- Total fatal accidents: 870
- Most common injury type: Major
- Most dangerous road condition: Dry

## Geographic Analysis

- Number of unique neighborhoods: 159
- Top 5 neighborhoods by accident count:

```
NEIGHBOURHOOD_158
West Humber-Clairville    597
Yonge-Bay Corridor        376
Wexford/Maryvale          361
South Riverdale           353
South Parkdale            304
```

## Analysis Insights

1. **Feature Importance Based on Correlations:**

   - Strong positive correlations (> 0.1):
     - Truck involvement (0.111)
     - Speeding (0.090)
     - Pedestrian involvement (0.099)
   - Moderate positive correlations (0.05-0.1):
     - Location features (Latitude: 0.059)
     - Alcohol involvement (0.029)
   - Weak or negative correlations:
     - Time-based features (Hour: -0.031)
     - Environmental conditions (Visibility: -0.016)
     - Most vehicle types except trucks

2. **Feature Selection Recommendations:**

   - Keep high-impact features: Truck involvement, Speeding, Pedestrian factors
   - Consider removing features with correlations < |0.05|
   - Retain location-based features despite moderate correlations

3. **Data Quality Insights:**

   - High missing value rates in behavioral factors (Alcohol: 18149, Speeding: 16263)
   - Complete data for location and time features
   - Missing values in Yes/No columns can be filled with 'No'

4. **Temporal Patterns:**

   - Afternoon period (12-16) shows highest accident frequency
   - Seasonal variations significant with summer peak
   - Time of day correlations suggest different risk patterns

5. **Geographic Insights:**
   - Clear accident hotspots in specific neighborhoods
   - Location features show consistent but moderate correlations
   - Spatial clustering evident in high-risk areas

## Correlation Analysis Details

1. **Vehicle Type Impact:**

   - Trucks: Strongest correlation with fatalities (0.111)
   - Motorcycles: Minimal correlation (-0.006)
   - Automobiles: Negative correlation (-0.075)
   - Transit vehicles: Moderate positive (0.048)

2. **Behavioral Factors:**

   - Speeding: Strong positive correlation (0.090)
   - Aggressive driving: Negative correlation (-0.033)
   - Red light running: Minimal impact (-0.006)
   - Alcohol involvement: Moderate positive (0.029)

3. **Environmental Conditions:**

   - Visibility: Weak negative (-0.016)
   - Road surface: Minimal impact (-0.005)
   - Light conditions: Moderate negative (-0.049)

4. **Temporal Features:**

   - Hour of day: Weak negative (-0.031)
   - Day of Week: Very weak (-0.008)
   - Month: Minimal (0.006)
   - Season: Weak positive (0.009)

5. **Location Features:**
   - District: Negative correlation (-0.030)
   - Geographic coordinates: Weak positive
   - Neighborhood features: Varying weak correlations

## Output Directory Structure

All analysis outputs are organized in the following structure:

```
insights/
├── analysis_report.md        # This comprehensive analysis report
├── correlation/             # Correlation analysis and matrices
├── dt_tuning/              # Decision tree tuning analysis results
├── geographic_analysis/    # Geographic distribution analysis
├── performance/           # Model performance metrics and feature importance
├── seasonal_analysis/     # Seasonal patterns and analysis
├── serialized_artifacts/  # Saved model and pipeline artifacts
├── severity_analysis/     # Severity distribution analysis
└── time_analysis/        # Time-based patterns and trends
```

## Generated Visualizations

1. Correlation Analysis (insights/correlation/):

   - correlation_matrix.png
   - feature_correlations.csv
   - feature_importance_heatmap.png

2. Decision Tree Tuning (insights/dt_tuning/):

   - main_dataset_results.csv
   - main_dataset_results.md
   - unseen_dataset_results.csv
   - unseen_dataset_results.md
   - hyperparameter_tuning_results.png

3. Geographic Analysis (insights/geographic_analysis/):

   - accident_heatmap.html
   - neighborhood_distribution.png
   - risk_zones_map.png

4. Performance Analysis (insights/performance/):

   - confusion_matrix.png
   - roc_curve.png
   - precision_recall_curve.png
   - feature_importance_plot.png
   - model_comparison.png

5. Seasonal Analysis (insights/seasonal_analysis/):

   - monthly_accidents.png
   - seasonal_neighborhood.png
   - yearly_trends.png
   - weather_impact.png

6. Severity Analysis (insights/severity_analysis/):

   - injury_distribution.png
   - severity_by_factor.png
   - condition_severity.png
   - vehicle_type_severity.png

7. Time Analysis (insights/time_analysis/):

   - hourly_severity.png
   - daily_patterns.png
   - peak_hours_heatmap.png
   - temporal_risk_analysis.png

8. Serialized Artifacts (insights/serialized_artifacts/):
   - trained_model.pkl
   - feature_scaler.pkl
   - preprocessing_pipeline.pkl
   - feature_selector.pkl
