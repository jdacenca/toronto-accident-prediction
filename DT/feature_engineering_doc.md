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

### Removed Features

1. Traffic-related:

   - REDLIGHT (|corr| = -0.006)

2. Environmental:

   - EMERG_VEH (|corr| = -0.015)

3. Driver-related:

   - DISABILITY (|corr| = -0.007)

4. Deprecated neighborhood columns:

   - HOOD_140
   - NEIGHBOURHOOD_140

5. Strong correlation with deprecated neighborhood columns:

   - HOOD_148

## Data Preprocessing Steps

1. Removal of irrelevant records:

   - 'Property Damage O' accidents excluded

2. Handling missing values:

   - Numerical columns: Filled with median
   - Categorical columns: Filled with mode
   - Binary columns: Defaulted to 'No'

3. Feature encoding:
   - Binary columns: Converted to 0/1
   - Categorical columns: Label encoded
   - Time features: Custom encoding for periods
