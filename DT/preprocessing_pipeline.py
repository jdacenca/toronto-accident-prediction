import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning the accident data"""
    
    # Columns that should be dropped during initial cleaning
    COLUMNS_TO_DROP = [
        # Redundant identifiers and IDs
        'INDEX', 'OBJECTID', 'FATAL_NO',
        
        # Deprecated neighborhood columns
        'HOOD_140', 'NEIGHBOURHOOD_140',

        # Strong correlation with deprecated neighborhood columns
        'HOOD_158',
        
        # Features with very weak correlations
        'REDLIGHT',          # corr: -0.006
        'EMERG_VEH',         # corr: -0.015
        'DISABILITY',        # corr: -0.007
        'ALCOHOL',           # corr: -0.006
        
        # Text fields with low predictive value
        'STREET1',           # Text data
        'STREET2',           # Text data
        'OFFSET',            # Text data
    ]
    
    def __init__(self):
        self.binary_cols = None
        self.label_encoders = {}
        self.numerical_medians = {}
        self.categorical_modes = {}
        self.feature_names_ = None
    
    def _clean_initial_data(self, df):
        """
        Perform initial data cleaning by dropping unnecessary columns.
        """
        df = df.copy()
        columns_to_drop = [col for col in self.COLUMNS_TO_DROP]
        return df.drop(columns=columns_to_drop, errors='ignore', inplace=False)
    
    def _identify_binary_columns(self, X):
        """Identify columns with binary (Yes/No) values"""
        binary_cols = X.select_dtypes(include=['object']).apply(
            lambda x: x.nunique() <= 2 and set(x.unique()).issubset({'Yes', 'No', np.nan})
        )
        return binary_cols[binary_cols].index
    
    def _store_numerical_medians(self, X):
        """Store median values for numerical columns"""
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            self.numerical_medians[col] = X[col].median()
    
    def _initialize_categorical_encoders(self, X):
        """Initialize label encoders and store modes for categorical columns"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'ACCLASS']
        
        for col in categorical_cols:
            if col not in self.binary_cols:
                self.categorical_modes[col] = X[col].mode()[0]
                self.label_encoders[col] = LabelEncoder()
                non_null_values = X[col].fillna(self.categorical_modes[col])
                self.label_encoders[col].fit(non_null_values)
    
    def _transform_binary_columns(self, X):
        """Transform binary columns to 0/1 values"""
        for col in self.binary_cols:
            if col in X.columns:
                X[col] = X[col].fillna('No')
                X[col] = (X[col] == 'Yes').astype(int)
        return X
    
    def _transform_numerical_columns(self, X):
        """Fill missing values in numerical columns"""
        for col, median in self.numerical_medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(median)
        return X
    
    def _transform_categorical_columns(self, X):
        """Transform categorical columns using label encoding"""
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna(self.categorical_modes[col])
                X[col] = encoder.transform(X[col])
        return X
    
    def fit(self, X, y=None):
        # Apply initial cleaning
        X = self._clean_initial_data(X)
        
        # Identify column types and store necessary values
        self.binary_cols = self._identify_binary_columns(X)
        self._store_numerical_medians(X)
        self._initialize_categorical_encoders(X)
        
        return self
    
    def transform(self, X):
        # Apply initial cleaning
        X = self._clean_initial_data(X)
        X = X.copy()
        
        # Remove property damage only accidents
        if 'ACCLASS' in X.columns:
            X = X[X['ACCLASS'] != 'Property Damage O']
            X = X.drop('ACCLASS', axis=1)
        
        # Apply transformations
        X = self._transform_binary_columns(X)
        X = self._transform_numerical_columns(X)
        X = self._transform_categorical_columns(X)
        
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering focused on accident severity prediction"""
    
    def __init__(self):
        self.feature_names_ = None
        self.location_clusters = None
        self.time_period_encoder = LabelEncoder()
    
    def fit(self, X, y=None):
        # Fit location clusters if latitude and longitude are available
        if all(col in X.columns for col in ['LATITUDE', 'LONGITUDE']):
            from sklearn.cluster import KMeans
            coords = X[['LATITUDE', 'LONGITUDE']].copy()
            self.location_clusters = KMeans(n_clusters=10, random_state=48)
            self.location_clusters.fit(coords)
            
        # Fit encoders on the full set of possible values
        self.time_period_encoder.fit([
            'Night (0-5)', 'Morning (6-11)', 
            'Afternoon (12-16)', 'Evening (17-20)', 
            'Night (21-23)'
        ])
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Time-based features
        if 'TIME' in X.columns:
            # Extract hour as integer (0-23)
            X['Hour'] = X['TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
        
        if 'DATE' in X.columns:
            X['DATE'] = pd.to_datetime(X['DATE'])
            
            # Add TimePeriod
            X['TimePeriod'] = pd.cut(
                X['Hour'], 
                bins=[-1, 5, 11, 16, 20, 23],
                labels=['Night (0-5)', 'Morning (6-11)', 
                       'Afternoon (12-16)', 'Evening (17-20)', 
                       'Night (21-23)']
            )
            # Encode TimePeriod
            X['TimePeriod'] = self.time_period_encoder.transform(X['TimePeriod'])
            
            # Drop intermediate columns
            X = X.drop(['DATE'], axis=1)
        
        # Location-based features
        if self.location_clusters is not None and all(col in X.columns for col in ['LATITUDE', 'LONGITUDE']):
            X['LOCATION_CLUSTER'] = self.location_clusters.predict(X[['LATITUDE', 'LONGITUDE']])
            
            # Calculate distance to cluster center
            cluster_centers = self.location_clusters.cluster_centers_
            distances = np.sqrt(((X[['LATITUDE', 'LONGITUDE']].values[:, np.newaxis] - 
                               cluster_centers) ** 2).sum(axis=2))
            X['DISTANCE_TO_CLUSTER_CENTER'] = distances.min(axis=1)
        
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        return X

def create_preprocessing_pipeline():
    """Create the main preprocessing pipeline"""
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
    ])