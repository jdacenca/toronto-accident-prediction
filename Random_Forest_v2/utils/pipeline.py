"""Main preprocessing pipeline for accident data."""

from sklearn.pipeline import Pipeline
from utils.data_cleaner import DataCleaner
from utils.feature_engineer import FeatureEngineer


def create_preprocessing_pipeline() -> Pipeline:
    """Create the main preprocessing pipeline.

    Returns:
        Pipeline: A scikit-learn pipeline that combines data cleaning and feature engineering.
    """
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
    ])