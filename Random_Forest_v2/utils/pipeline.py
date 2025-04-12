
from sklearn.pipeline import Pipeline
from utils.data_cleaner import DataCleaner
from utils.feature_engineer import FeatureEngineer


def create_preprocessing_pipeline() -> Pipeline:

    return Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
    ])