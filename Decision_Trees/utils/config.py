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
]
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

# Time periods for feature engineering
TIME_PERIODS = [
    ['Night (0-5)'], ['Morning (6-11)'],
    ['Afternoon (12-16)'], ['Evening (17-20)'],
    ['Night (21-23)']
]

# Time period bins
TIME_BINS = [-1, 5, 11, 16, 20, 23]
TIME_LABELS = ['Night (0-5)', 'Morning (6-11)', 
               'Afternoon (12-16)', 'Evening (17-20)', 
               'Night (21-23)']

# Random state for reproducibility
RANDOM_STATE = 48

# Number of location clusters
N_LOCATION_CLUSTERS = 10 