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
