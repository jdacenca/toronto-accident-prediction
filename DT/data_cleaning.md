# Data Cleaning Strategy Documentation

## Latest Dataset Analysis (Toronto Accident Data)

### Dataset Overview

```
Dataset Statistics:
- Total rows: 18,957 accidents
- Total columns: 54
- Target distribution:
  - Non-fatal injuries: 16,268
  - Fatal accidents: 2,670
```

### Feature Removal Strategy

1. **Redundant Identifiers**

   - INDEX
   - OBJECTID
   - FATAL_NO

2. **Deprecated Location Data**

   - HOOD_140
   - NEIGHBOURHOOD_140

3. **Strong Correlation with Deprecated Location Data**
   - HOOD_158

### Categorical Variables Analysis

1. **High-Impact Features**

   ```
   Feature      Unique Values    Correlation
   INVTYPE         8              0.156
   IMPACTYPE       12             0.142
   MANOEUVER       10             -0.037
   TRAFFCTL         8             -0.044
   ```

2. **Low-Impact Features**
   ```
   Feature      Unique Values    Correlation
   ROAD_CLASS       7             -0.013
   VISIBILITY       5             -0.016
   DRIVCOND         5             -0.008
   ```

### Accident-Level Feature Engineering

1. **Severity Aggregations**

   ```
   Feature                  Description
   total_casualties        Count of people per accident
   fatality_count         Count of fatalities
   injury_severity_ratio  Severe injuries / total
   ```

2. **Participant Analysis**
   ```
   Feature              Description
   driver_count        Number of drivers
   passenger_count     Number of passengers
   vehicle_count       Total vehicles
   age_range          Age span of participants
   ```

## Data Quality Improvements

### 1. Missing Value Strategy

```
Column          Strategy
LATITUDE       Median
LONGITUDE      Median
INVAGE        Median
Binary cols    Default 'No'
Categorical    Mode
```

### 2. Outlier Handling

- Location-based outliers handled by clustering
- Temporal features normalized to standard ranges
- Categorical outliers merged into 'Other' category

## Implementation Details

### 1. Cleaning Pipeline Steps

```python
Pipeline([
    ('cleaner', DataCleaner()),
    ('engineer', FeatureEngineer())
])
```

### 2. Feature Type Handling

```
Data Types:
- Numerical: 8 columns
- Categorical: 12 columns
- Binary: 6 columns
- DateTime: 2 columns
- Accident-level: 14 new features
```
