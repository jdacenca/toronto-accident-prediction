"""Configuration module for the Decision Tree model training pipeline."""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INSIGHTS_DIR = BASE_DIR / "insights"
SERIALIZED_DIR = INSIGHTS_DIR / "serialized_artifacts"
PERFORMANCE_DIR = INSIGHTS_DIR / "performance"

# Data cleaning constants
COLUMNS_TO_DROP = [
    'INDEX', 'OBJECTID', 'FATAL_NO',
    'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION',
    'HOOD_158', 'ACCNUM', 
    'STREET1', 'STREET2',
    'OFFSET',
    'INJURY', 'INITDIR',
    'VEHTYPE', 'MANOEUVER', 'DRIVACT',
    'DRIVCOND', 'PEDTYPE', 'PEDACT',
    'CYCLISTYPE', 'CYCACT', 'x', 'y',
    'DATE', 'TIME'
]

BINARY_COLUMNS = [
    'PEDESTRIAN',
    'CYCLIST',
    'AUTOMOBILE',
    'MOTORCYCLE',
    'TRUCK',
    'TRSN_CITY_VEH',
    'EMERG_VEH',
    'PASSENGER',
    'SPEEDING',
    'AG_DRIV',
    'REDLIGHT',
    'ALCOHOL',
    'DISABILITY'
]

COLUMNS_TO_LABEL_ENCODE = [
    'ROAD_CLASS',
    'DISTRICT',
    'ACCLOC',
    'TRAFFCTL',
    'VISIBILITY',
    'LIGHT',
    'RDSFCOND',
    'IMPACTYPE',
    'INVTYPE',
    'INVAGE',
    'PEDCOND',
    'CYCCOND',
    'NEIGHBOURHOOD_158',
]

NA_FILL_COLUMNS = ['PEDCOND', 'CYCCOND']

TARGET = 'ACCLASS'

# MONTH_MAPPING = {
#     'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
#     'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
#     'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
# }

# DAY_MAPPING = {
#     'MONDAY': 1, 'TUESDAY': 2, 'WEDNESDAY': 3, 'THURSDAY': 4,
#     'FRIDAY': 5, 'SATURDAY': 6, 'SUNDAY': 7 
# }

# WEEK_MAPPING = {
#     'MONDAY': 1, 'TUESDAY': 2, 'WEDNESDAY': 3, 'THURSDAY': 4,
#     'FRIDAY': 5, 'SATURDAY': 6, 'SUNDAY': 7 
# }

# DAYOFWEEK_MAPPING = {
#     'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3,
#     'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6 
# }

BINARY_MAPPING = {'YES': 1, 'NO': 0}

TARGET_MAPPING = {'FATAL': 1, 'NON-FATAL INJURY': 0}

# Model parameters
MODEL_PARAMS = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

# Scoring metrics
SCORING_METRICS = {
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy'
}

# Random state for reproducibility
RANDOM_STATE = 48