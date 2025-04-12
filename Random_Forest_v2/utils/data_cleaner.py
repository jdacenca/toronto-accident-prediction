
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from utils.config import COLUMNS_TO_DROP


class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.binary_cols: list[str] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.numerical_medians: dict[str, float] = {}

    def _clean_initial_data(self, df: pd.DataFrame) -> None:
        """Perform initial data cleaning by dropping unnecessary columns."""
        # Drop unnecessary columns
        df.drop(columns=COLUMNS_TO_DROP, errors='ignore', inplace=True)
        # Convert all string columns to uppercase, excluding date fields
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            # Skip columns that are likely date fields based on name
            if any(date_term in col.upper() for date_term in ['DATE', 'TIME']):
                continue
            # Convert to uppercase
            df[col] = df[col].str.upper()
        # Check if ACCLASS exists before performing operations on it
        if 'ACCLASS' in df.columns:
            # Fill missing ACCLASS values with 'Fatal'
            df['ACCLASS'] = df['ACCLASS'].fillna('FATAL')
            # Drop Property Damage Only records
            df.drop(df[df['ACCLASS'] == 'PROPERTY DAMAGE O'].index, inplace=True)

    def _identify_binary_columns(self, df: pd.DataFrame) -> list[str]:
        binary_cols = df.select_dtypes(include=['object']).apply(
            lambda x: x.nunique() <= 2 and set(x.unique()).issubset({'YES', 'NO', np.nan})
        )
        return binary_cols[binary_cols].index.tolist()

    def _store_numerical_medians(self, df: pd.DataFrame) -> None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            self.numerical_medians[col] = df[col].median()

    def _initialize_categorical_encoders(self, df: pd.DataFrame) -> None:
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'ACCLASS']

        for col in categorical_cols:
            if col not in self.binary_cols:
                # Special handling for PEDCOND and CYCCOND
                if col in ['PEDCOND', 'CYCCOND']:
                    # Fill missing values with 'NA' before fitting
                    values = df[col].fillna('NA')
                else:
                    values = df[col].fillna('OTHER')

                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(values)

    def _transform_binary_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.binary_cols:
            if col in X.columns:
                X[col] = X[col].fillna('NO')
                X[col] = (X[col] == 'YES').astype(int)
        return X

    def _transform_numerical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, median in self.numerical_medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(median)
        return X

    def _transform_categorical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                # Special handling for PEDCOND and CYCCOND
                if col in ['PEDCOND', 'CYCCOND']:
                    X[col] = X[col].fillna('NA')
                else:
                    X[col] = X[col].fillna('OTHER')
                X[col] = encoder.transform(X[col])
        return X

    def fit(self, df: pd.DataFrame) -> 'DataCleaner':
        # Apply initial cleaning
        self._clean_initial_data(df)
        # Identify column types and store necessary values
        self.binary_cols = self._identify_binary_columns(df)
        self._store_numerical_medians(df)
        self._initialize_categorical_encoders(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop('ACCLASS', axis=1)
        # Apply transformations
        X = self._transform_binary_columns(X)
        X = self._transform_numerical_columns(X)
        X = self._transform_categorical_columns(X)
        return X