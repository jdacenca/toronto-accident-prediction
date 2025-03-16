# Feature Engineering Documentation

## Time-Based Features

1. Hour: Extracted from TIME to capture time-of-day patterns (|corr| = 0.082)
2. TimePeriod: Encoded categorical periods (|corr| = 0.076)
   - Night (0-5)
   - Morning (6-11)
   - Afternoon (12-16)
   - Evening (17-20)
   - Night (21-23)

## Location-Based Features

1. LOCATION_CLUSTER: K-means clustering (k=10) of accident locations
   - Rationale: Identifies high-risk areas and accident hotspots
   - Implementation: Uses LATITUDE and LONGITUDE
2. DISTANCE_TO_CLUSTER_CENTER: Distance to nearest cluster center
   - Rationale: Captures location patterns relative to accident hotspots

## Feature Selection Based on Correlation Analysis

### Retained Features (|correlation| ≥ 0.05)

1. Strong Correlations:

   - INVTYPE (|corr| = 0.156)
   - IMPACTYPE (|corr| = 0.142)
   - INVAGE (|corr| = 0.128)
   - PEDESTRIAN (|corr| = 0.112)
   - SPEEDING (|corr| = 0.095)

2. Moderate Correlations:
   - LATITUDE (|corr| = 0.087)
   - LONGITUDE (|corr| = 0.083)
   - Hour (|corr| = 0.082)
   - TimePeriod (|corr| = 0.076)
   - AGGRESSIVE (|corr| = 0.068)

### Removed Features (|correlation| < 0.05)

1. Traffic-related:

   - MANOEUVER (|corr| = -0.037)
   - ROAD_CLASS (|corr| = -0.013)
   - TRAFFCTL (|corr| = -0.044)

2. Environmental:

   - VISIBILITY (|corr| = -0.016)
   - RDSFCOND (|corr| = -0.005)

3. Driver-related:

   - INITDIR (|corr| = 0.021)
   - DRIVACT (|corr| = -0.013)
   - DRIVCOND (|corr| = -0.011)

4. Vehicle-related:
   - MOTORCYCLE (|corr| = -0.006)
   - TRSN_CITY_VEH (|corr| = 0.048)
   - PASSENGER (|corr| = -0.004)

## Data Preprocessing Steps

1. Removal of irrelevant records:

   - 'Property Damage O' accidents excluded
   - Zero-importance cyclist features removed

2. Handling missing values:

   - Numerical columns: Filled with median
   - Categorical columns: Filled with mode
   - Binary columns: Defaulted to 'No'

3. Feature encoding:
   - Binary columns: Converted to 0/1
   - Categorical columns: Label encoded
   - Time features: Custom encoding for periods

## Model Performance Impact

- Overall accuracy: 92%
- Macro-average metrics:
  - Precision: 84%
  - Recall: 85%
  - F1-score: 85%
