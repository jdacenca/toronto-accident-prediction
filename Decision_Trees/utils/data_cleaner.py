"""Data cleaning transformer for accident data."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from utils.config import BINARY_COLUMNS, BINARY_MAPPING, COLUMNS_TO_DROP, TARGET, NA_FILL_COLUMNS, TARGET_MAPPING

class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning the accident data."""
    
    def __init__(self):
        self.categorical_cols = []
        self.encoded_categorical_cols = {} 
        self.numerical_cols = []
        self.binary_cols = BINARY_COLUMNS
        self.columns_to_drop = COLUMNS_TO_DROP
        self.target_mapping = TARGET_MAPPING
        self.binary_mapping = BINARY_MAPPING
        self.na_fill_cols = NA_FILL_COLUMNS
        
    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> None:
        """Drop columns that are not needed."""
        df.drop(columns=self.columns_to_drop, errors='ignore', inplace=True)
        
    def _convert_strings_to_uppercase(self, df: pd.DataFrame) -> None:
        """Convert all string columns to uppercase."""
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            # Convert to uppercase
            df[col] = df[col].str.upper()
            
    def _process_target_variable(self, df: pd.DataFrame) -> None:
        """Process the target variable (ACCLASS) by handling missing values and encoding."""
        if TARGET in df.columns:
            # Fill missing ACCLASS values with 'Fatal'
            df.fillna({ TARGET : 'FATAL' }, inplace=True)
            # Drop Property Damage Only records
            df.drop(df[df[TARGET] == 'PROPERTY DAMAGE O'].index, inplace=True)
            # Convert ACCLASS to binary (1 for FATAL, 0 for NON-FATAL)
            df[TARGET] = df[TARGET].map(self.target_mapping).astype(int)
    
    def _initialize_categorical_cols(self, df: pd.DataFrame) -> None:
        """Initializ categorical columns."""
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        self.categorical_cols = [col for col in self.categorical_cols if col != TARGET]

    def _initialize_numerical_cols(self, df: pd.DataFrame) -> None:
        """Initialize numerical columns."""
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                
    def _fill_missing_values_in_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in binary columns."""
        for col in self.binary_cols:
            if col in df.columns:
                # Fill missing values with 'NO'
                df.fillna({ col: 'NO' }, inplace=True)
    
    def _fill_missing_values_in_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in numerical columns."""
        for col in self.numerical_cols:
            if col in df.columns:
                median = df[col].median()
                # Fill missing values with the median
                df.fillna({ col: median }, inplace=True)
    
    def _fill_missing_values_in_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in categorical columns."""
        for col in self.categorical_cols:
            if col in df.columns:
                # Special handling for columns that need NA filling
                if col in self.na_fill_cols:
                    df.fillna({ col: 'NA' }, inplace=True)
                # Fill missing values with 'OTHER' for other categorical columns    
                else:
                    df.fillna({ col: 'OTHER' }, inplace=True)
    
    def _transform_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform binary columns to 0/1 values."""
        for col in self.binary_cols:
            if col in df.columns:
                df[col] = df[col].map(self.binary_mapping).astype(int)

    def _transform_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using Label Encoding."""
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Apply label encoding to the column
                df[col] = le.fit_transform(df[col])
                # Store the label encoder for future use (if needed)
                self.encoded_categorical_cols[col] = df[col]
    
    def fit(self, df: pd.DataFrame) -> 'DataCleaner':
        """Fit the data cleaner."""
        self._initialize_categorical_cols(df)      
        self._initialize_numerical_cols(df)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._drop_unnecessary_columns(df)
        self._convert_strings_to_uppercase(df)
        self._process_target_variable(df)
        self._fill_missing_values_in_binary_columns(df)
        self._fill_missing_values_in_numerical_columns(df)
        self._fill_missing_values_in_categorical_columns(df)
        self._transform_binary_columns(df)
        self._transform_categorical_columns(df)
        return df