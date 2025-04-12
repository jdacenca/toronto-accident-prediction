"""Feature engineering transformer for accident data."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering focused on accident severity prediction."""
    
    def __init__(self):
        pass
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit the feature engineer."""       
        return self
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'TIME' in df.columns:
            # Extract hour as integer (0-23)
            df['HOUR'] = df['TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
        
        if 'DATE' in df.columns:
            # Convert DATE to datetime if not already
            df['DATE'] = pd.to_datetime(df['DATE'])            
            # Extract month (1-12)
            df['MONTH'] = df['DATE'].dt.month            
            # Extract day of month (1-31)
            df['DAY'] = df['DATE'].dt.day            
            # Extract week number (1-53)
            df['WEEK'] = df['DATE'].dt.isocalendar().week            
            # Extract day of week (0-6, where 0 is Monday)
            df['DAYOFWEEK'] = df['DATE'].dt.dayofweek
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding engineered features."""        
        # Create time-based features 
        return self._create_time_features(df)