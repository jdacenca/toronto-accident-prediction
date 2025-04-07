"""Data cleaning transformer for accident data."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from utils.config import COLUMNS_TO_DROP

class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning the accident data."""
    
    def __init__(self):
        self.binary_cols: list[str] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.numerical_medians: dict[str, float] = {}
    
    def _clean_initial_data(self, df: pd.DataFrame) -> None:
        """Perform initial data cleaning by dropping unnecessary columns."""
        # Fill missing ACCLASS values with 'Fatal'
        df['ACCLASS'] = df['ACCLASS'].fillna('Fatal')
        # Drop Property Damage Only records
        df.drop(df[df['ACCLASS'] == 'Property Damage Only'].index, inplace=True)
        # Drop unnecessary columns
        df.drop(columns=COLUMNS_TO_DROP, errors='ignore', inplace=True)
    
    def _identify_binary_columns(self, df: pd.DataFrame) -> list[str]:
        """Identify columns with binary (Yes/No) values."""
        binary_cols = df.select_dtypes(include=['object']).apply(
            lambda x: x.nunique() <= 2 and set(x.unique()).issubset({'Yes', 'No', np.nan})
        )
        return binary_cols[binary_cols].index.tolist()
    
    def _store_numerical_medians(self, df: pd.DataFrame) -> None:
        """Store median values for numerical columns."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            self.numerical_medians[col] = df[col].median()
    
    def _initialize_categorical_encoders(self, df: pd.DataFrame) -> None:
        """Initialize label encoders and store modes for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'ACCLASS']
        
        for col in categorical_cols:
            if col not in self.binary_cols:
                # Special handling for PEDCOND and CYCCOND
                if col in ['PEDCOND', 'CYCCOND']:
                    # Fill missing values with 'NA' before fitting
                    values = df[col].fillna('NA')
                else:
                    values = df[col].fillna('Other')
                
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(values)
    
    def _transform_binary_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform binary columns to 0/1 values."""
        for col in self.binary_cols:
            if col in X.columns:
                X[col] = X[col].fillna('No')
                X[col] = (X[col] == 'Yes').astype(int)
        return X
    
    def _transform_numerical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in numerical columns."""
        for col, median in self.numerical_medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(median)
        return X
    
    def _transform_categorical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using label encoding."""
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                # Special handling for PEDCOND and CYCCOND
                if col in ['PEDCOND', 'CYCCOND']:
                    X[col] = X[col].fillna('NA')
                else:
                    X[col] = X[col].fillna('Other')
                X[col] = encoder.transform(X[col])
        return X
    
    def fit(self, df: pd.DataFrame) -> 'DataCleaner':
        """Fit the data cleaner."""
        # Apply initial cleaning
        self._clean_initial_data(df)     
        # Identify column types and store necessary values
        self.binary_cols = self._identify_binary_columns(df)
        self._store_numerical_medians(df)
        self._initialize_categorical_encoders(df)        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        X = df.drop('ACCLASS', axis=1)
        # Apply transformations
        X = self._transform_binary_columns(X)
        X = self._transform_numerical_columns(X)
        X = self._transform_categorical_columns(X)
        return X