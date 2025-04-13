import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning the accident data"""

    def __init__(self):
        self.binary_cols = None
        self.ordinal_encoders = {}
        self.numerical_medians = {}
        self.categorical_modes = {}
        self.feature_names_ = None
        self.columns_to_drop = [
        'INDEX', 'OBJECTID', 'FATAL_NO', 'HOOD_140',
        'NEIGHBOURHOOD_140', 'HOOD_158',
    ]
    def _clean_initial_data(self, df):
        index_range_to_keep = df['ACCLASS'] != 'Property Damage O'
        df = df[index_range_to_keep]
        columns_to_drop = [col for col in self.COLUMNS_TO_DROP]
        return df.drop(columns=columns_to_drop, errors='ignore', inplace=False)
    def _identify_binary_columns(self, df_cleaned):
        binary_cols = df_cleaned.select_dtypes(include=['object']).apply(
            lambda x: x.nunique() <= 2 and set(x.unique()).issubset({'Yes', 'No', np.nan})
        )
        return binary_cols[binary_cols].index
    def _store_numerical_medians(self, df_cleaned):
        numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            self.numerical_medians[col] = df_cleaned[col].median()
    def _initialize_categorical_encoders(self, df_cleaned):
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'ACCLASS']
        for col in categorical_cols:
            if col not in self.binary_cols:
                self.categorical_modes[col] = df_cleaned[col].mode()[0]
                self.ordinal_encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                non_null_values = df_cleaned[col].fillna(self.categorical_modes[col])
                self.ordinal_encoders[col].fit(non_null_values.values.reshape(-1, 1))
    def _transform_binary_columns(self, X):
        for col in self.binary_cols:
            if col in X.columns:
                X[col] = X[col].fillna('No')
                X[col] = (X[col] == 'Yes').astype(int)
        return X
    def _transform_numerical_columns(self, X):
        for col, median in self.numerical_medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(median)
        return X
    def _transform_categorical_columns(self, X):
        for col, encoder in self.ordinal_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna(self.categorical_modes[col])
                encoded_values = encoder.transform(X[col].values.reshape(-1, 1))
                X[col] = encoded_values.flatten()
        return X
    def fit(self, df):
        # Apply initial cleaning first to identify columns correctly on relevant data
        df_cleaned = self._clean_initial_data(df.copy()) # Work on a copy
        self.binary_cols = self._identify_binary_columns(df_cleaned)
        self._store_numerical_medians(df_cleaned)
        self._initialize_categorical_encoders(df_cleaned)
        # Store cleaned feature names (excluding target) before transform adds/removes cols
        self.feature_names_ = df_cleaned.drop('ACCLASS', axis=1, errors='ignore').columns.tolist()
        return self
    def transform(self, df):
        # Apply initial cleaning
        df_cleaned = self._clean_initial_data(df.copy()) # Work on a copy
        X = df_cleaned.drop('ACCLASS', axis=1, errors='ignore')
        # Apply transformations
        X = self._transform_binary_columns(X)
        X = self._transform_numerical_columns(X)
        X = self._transform_categorical_columns(X)
        # Ensure only columns present during fit are returned, in the correct order
        X = X[self.feature_names_]
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering focused on accident severity prediction"""
    def __init__(self):
        self.feature_names_ = None
        self.location_clusters = None
        self.time_period_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    def fit(self, X, y=None): # Modified to accept y=None for pipeline compatibility
        if all(col in X.columns for col in ['LATITUDE', 'LONGITUDE']):
            coords = X[['LATITUDE', 'LONGITUDE']].copy()
            self.location_clusters = KMeans(n_clusters=10, random_state=48, n_init=10) # Added n_init
            self.location_clusters.fit(coords)
        time_periods = [
            ['Night (0-5)'], ['Morning (6-11)'],
            ['Afternoon (12-16)'], ['Evening (17-20)'],
            ['Night (21-23)']
        ]
        self.time_period_encoder.fit(time_periods)
        # Store feature names seen during fit
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        X_transformed = X.copy() # Work on a copy
        original_cols = self.feature_names_ # Columns seen during fit

        # Time-based features
        if 'TIME' in original_cols:
            # Ensure TIME is treated consistently, handle potential errors
            try:
                X_transformed['Hour'] = X_transformed['TIME'].astype(int).astype(str).str.zfill(4).str[:2].astype(int)
            except ValueError:
                 # Handle cases where TIME might not be purely numeric, e.g., fill with median or mode hour
                 # For simplicity, let's fill NaNs resulting from conversion errors with a common hour like 12
                 X_transformed['Hour'] = pd.to_numeric(X_transformed['TIME'], errors='coerce')
                 median_hour = X_transformed['Hour'].astype(str).str.zfill(4).str[:2].astype(float).median()
                 if pd.isna(median_hour): median_hour = 12 # Default if median can't be calculated
                 X_transformed['Hour'] = X_transformed['TIME'].astype(str).str.zfill(4).str[:2]
                 X_transformed['Hour'] = pd.to_numeric(X_transformed['Hour'], errors='coerce').fillna(median_hour).astype(int)

        if 'DATE' in original_cols:
            X_transformed['DATE'] = pd.to_datetime(X_transformed['DATE'], errors='coerce') # Handle potential errors

            if 'Hour' in X_transformed.columns: # Check if Hour was created
                 X_transformed['TimePeriod'] = pd.cut(
                    X_transformed['Hour'],
                    bins=[-1, 5, 11, 16, 20, 23],
                    labels=['Night (0-5)', 'Morning (6-11)',
                           'Afternoon (12-16)', 'Evening (17-20)',
                           'Night (21-23)']
                 )
                 # Handle potential NaNs introduced by cut if Hour had issues
                 if X_transformed['TimePeriod'].isnull().any():
                     mode_period = X_transformed['TimePeriod'].mode()[0] if not X_transformed['TimePeriod'].mode().empty else 'Afternoon (12-16)'
                     X_transformed['TimePeriod'] = X_transformed['TimePeriod'].cat.add_categories('Unknown').fillna('Unknown')


                 encoded_values = self.time_period_encoder.transform(
                    X_transformed['TimePeriod'].values.reshape(-1, 1)
                 )
                 X_transformed['TimePeriod'] = encoded_values.flatten()
            else:
                 # Handle case where Hour couldn't be created (e.g., TIME column missing/problematic)
                 # Create a default TimePeriod or handle as needed
                 X_transformed['TimePeriod'] = -1 # Assign default unknown value


            # Drop intermediate columns safely
            X_transformed = X_transformed.drop(['DATE'], axis=1, errors='ignore')
            X_transformed = X_transformed.drop(['TIME'], axis=1, errors='ignore') # Also drop TIME if Hour is derived
            X_transformed = X_transformed.drop(['Hour'], axis=1, errors='ignore') # Drop Hour after use

        # Location-based features
        if self.location_clusters is not None and all(col in original_cols for col in ['LATITUDE', 'LONGITUDE']):
            # Ensure lat/lon columns exist before proceeding
            if all(col in X_transformed.columns for col in ['LATITUDE', 'LONGITUDE']):
                 # Handle potential NaNs in coordinates before prediction
                 coords_to_predict = X_transformed[['LATITUDE', 'LONGITUDE']].fillna(X_transformed[['LATITUDE', 'LONGITUDE']].median())

                 X_transformed['LOCATION_CLUSTER'] = self.location_clusters.predict(coords_to_predict)

                 cluster_centers = self.location_clusters.cluster_centers_
                 # Use coords_to_predict for distance calculation as well
                 distances = np.sqrt(((coords_to_predict.values[:, np.newaxis] -
                                   cluster_centers) ** 2).sum(axis=2))
                 X_transformed['DISTANCE_TO_CLUSTER_CENTER'] = distances.min(axis=1)
            else:
                # Handle missing LAT/LON columns if they were expected but not present
                X_transformed['LOCATION_CLUSTER'] = -1 # Default unknown value
                X_transformed['DISTANCE_TO_CLUSTER_CENTER'] = -1 # Default unknown value


        # Store final feature names
        self.feature_names_ = X_transformed.columns.tolist()

        return X_transformed

def create_preprocessing_pipeline():
    """Create the main preprocessing pipeline"""
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()), # Uncomment if you want feature engineering
    ])
