# Feature Engineering Documentation

## Time-Based Features
1. Hour: Extracted from DATE to capture time-of-day patterns
2. DayOfWeek: Captures weekly patterns (0=Monday, 6=Sunday)
3. Month: Captures seasonal patterns
4. IsWeekend: Binary indicator for weekend accidents
5. IsNight: Binary indicator for nighttime accidents (10 PM - 5 AM)

## Combined Features
1. POOR_ROAD_CONDITIONS: Aggregated from multiple road condition columns
   - Rationale: Simplifies multiple road condition indicators into a single feature
   - Implementation: Takes maximum value across all road condition columns

2. POOR_VISIBILITY: Aggregated from multiple visibility-related columns
   - Rationale: Combines various visibility factors into a single indicator
   - Implementation: Takes maximum value across all visibility columns

## Feature Selection Criteria
1. Correlation with target (Fatal accidents)
2. Domain knowledge about accident factors
3. Feature importance from initial model training
4. Low multicollinearity between selected features

## Data Cleaning Steps
1. Removal of 'Property Damage O' accidents
2. Filling missing values in binary (Yes/No) columns with 'No'
3. Encoding categorical variables
4. Standardization of numerical features

Note: The actual feature selection will be performed after initial model training
and feature importance analysis.