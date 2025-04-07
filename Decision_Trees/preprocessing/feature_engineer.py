"""Feature engineering transformer for accident data."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering focused on accident severity prediction."""
    
    def __init__(self):
        pass
    
    def fit(self, X: pd.DataFrame) -> 'FeatureEngineer':
        """Fit the feature engineer."""
            
        return self
    
    def _create_time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'TIME' in X.columns:
            # Extract hour as integer (0-23)
            X['HOUR'] = X['TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
            # Drop original TIME column after extraction
            X = X.drop(['TIME'], axis=1)
        
        if 'DATE' in X.columns:
            # Convert DATE to datetime if not already
            X['DATE'] = pd.to_datetime(X['DATE'])
            
            # Extract month (1-12)
            X['MONTH'] = X['DATE'].dt.month
            
            # Extract day of month (1-31)
            X['DAY'] = X['DATE'].dt.day
            
            # Extract week number (1-53)
            X['WEEK'] = X['DATE'].dt.isocalendar().week
            
            # Extract day of week (0-6, where 0 is Monday)
            X['DAYOFWEEK'] = X['DATE'].dt.dayofweek
            
            # Drop original DATE column after extraction
            X = X.drop(['DATE'], axis=1)
        
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding engineered features."""
        X = X.copy()
        
        # Create time-based features
        X = self._create_time_features(X)
        
        return X