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

1. **Correlation-Based Removal**

   - Features with |correlation| < 0.05 with target
   - Total features removed: 15
   - Memory reduction: ~28%
   - No significant impact on model performance

2. **Redundant Identifiers**

   - INDEX
   - OBJECTID
   - FATAL_NO
   - ACCNUM

3. **Deprecated Location Data**
   - HOOD_140
   - NEIGHBOURHOOD_140
   - HOOD_158

### Categorical Variables Analysis

1. **High-Impact Features** (|correlation| ≥ 0.05)

   ```
   Feature      Unique Values    Correlation
   INVTYPE         8              0.156
   IMPACTYPE       12             0.142
   TimePeriod      5              0.076
   ```

2. **Low-Impact Features** (removed)
   ```
   Feature      Unique Values    Correlation
   ROAD_CLASS       7             -0.013
   TRAFFCTL         8             -0.044
   MANOEUVER       10             -0.037
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
```
