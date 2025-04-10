"""Data cleaning transformer for accident data."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
# from category_encoders import BinaryEncoder

class LRPreprocessor(BaseEstimator, TransformerMixin):
    """Custom Transformer to clean and transform KSI datasets"""
    cols_drop = [
    'INDEX', 'OBJECTID', 'STREET1', 'STREET2',
    'INJURY', 'INITDIR',
    'VEHTYPE', 'MANOEUVER', 'DRIVACT','DRIVCOND',
    'PEDTYPE', 'PEDACT',
    'CYCLISTYPE', 'CYCACT',
    'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION',
    'HOOD_158',
    'FATAL_NO',
    'OFFSET','x','y'
    ]
    # Columns to be used during transformation
    cols_drop2 = ['ACCNUM','DATE','TIME']
    cols_drop3 = ['ROAD_CLASS','DISTRICT','ACCLOC','INVAGE','INVTYPE',
                  'PEDCOND','CYCCOND','PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE',
                  'TRSN_CITY_VEH','EMERG_VEH','PASSENGER','SPEEDING','AG_DRIV','NEIGHBOURHOOD_158']
    cols_drop.extend(cols_drop2)
    cols_drop.extend(cols_drop3)

    def __init__(self, level = 2):
        self._level = level # Level 1, performs all the clean-up, imputation and encoding
                            # Level 2, only performs the minimal required encoding

    def _init_clean(self, X):
        ###############
        # Clean up DATE and TIME columns to correct type

        # Only extract date
        X['DATE'] = pd.to_datetime(X['DATE'], format='%m/%d/%Y %I:%M:%S %p').dt.normalize()

        # Original 'TIME' column is 'int64' so 0006 is 6 which is an issue for to_datetime()
        # Fix the format with leading zero with zfill()
        X['TIME'] = X['TIME'].apply(lambda x: str(x).zfill(4))
        X['TIME'] = pd.to_datetime(X['TIME'],format='%H%M').dt.time

        ###############
        # Clean up the ROAD_CLASS for trailing space
        X['ROAD_CLASS'] = X['ROAD_CLASS'].str.strip()

    def _impute(self, X):
        ###############
        # Impute the empty ACCNUM from date, time, longtitue and latitude

        datetime = pd.to_datetime(X['DATE'].astype(str) + ' ' + X['TIME'].astype(str))
        long_lat = X[['LONGITUDE', 'LATITUDE']].astype(str).agg('_'.join, axis=1)
        datetime_long_lat = datetime.astype(str) + "_" + long_lat
        X['ACCNUM'] = X['ACCNUM'].fillna(datetime_long_lat).astype(str)

        ###############
        # Impute the ACCLASS
        X['ACCLASS'] = X['ACCLASS'].fillna(X['INJURY'])
        X['ACCLASS'] = X['ACCLASS'].replace('Property Damage O', 'Non-Fatal Injury')   # Or dropped

        ###############
        # Impute the missing DISTRICT based on HOOD_158, NEIGHBOURHOOD_158
        empty_district_index = X['DISTRICT'].isna()
        X['DISTRICT'] = X.groupby('HOOD_158')['DISTRICT'].transform(lambda x: x.fillna(x.mode()[0]))

        ###############
        # Impute blank with 'Other'
        other_columns = ['ROAD_CLASS', 'ACCLOC', 'TRAFFCTL','VISIBILITY',
                         'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE']

        X[other_columns] = X[other_columns].fillna("Other")

        ###############
        # Impute blank with 'Unknown'
        unknown_columns = ['PEDCOND', 'CYCCOND']
        X[unknown_columns] = X[unknown_columns].fillna("Unknown")

        ###############
        # Perform binary column transformation inside impute
        # self._binary_column_transform(X)

    def _boolean_column_transform(self, X):
        ###############
        # Binary column imputation with True / False
        col_boolean = ['ACCLASS', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH', 'PASSENGER',
                      'SPEEDING', 'AG_DRIV', 'ALCOHOL', 'DISABILITY', 'REDLIGHT', 'TRSN_CITY_VEH']

        for col in col_boolean:
            X[col] = X[col].map(
                {'Yes': True, 'No': False, np.nan: False, 'Fatal': True, 'Non-Fatal Injury': False, True: True, False: False})

        print(X['ACCLASS'].value_counts())
    def _add_datetime_new(self, X):
        X['DATETIME'] = pd.to_datetime(X['DATE'].astype(str) + ' ' + X['TIME'].astype(str))
        df = X.assign(
            # YEAR=X['DATETIME'].dt.year,
            MONTH=X['DATETIME'].dt.month,
            WEEK=X['DATETIME'].dt.isocalendar().week,
            DAY = X['DATETIME'].dt.day,
            DAYOFWEEK=X['DATETIME'].dt.dayofweek,
            HOUR=X['DATETIME'].dt.hour
        )
        # df.drop(columns=['DATE', 'TIME','DATETIME'], inplace = True)
        df.drop(columns=['DATETIME','DATE','TIME'], inplace = True)
        return df

    # Cyclic Encoding for cyclic columns extracted from DATETIME
    def _cyclic_enc(self, df, column, max_value):
        df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
        df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
        # return df

    def _cyclic_enc_transform(self, X):
        col_cyclic_encoding = ['MONTH', 'WEEK', 'DAY', 'DAYOFWEEK', 'HOUR']
        max_cyclic_encoding = [12, 53, 31, 7, 24]
        for col, max in zip(col_cyclic_encoding, max_cyclic_encoding):
            # print(f"{col} -> {max}")
            self._cyclic_enc(X, col, max)
        X.drop(col_cyclic_encoding, axis=1, inplace=True)
        # return df

    def _ohe_transform(self, X):
        ohe_cols = X.select_dtypes(include=['object']).columns.tolist()
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # to handle the unseen category as other
        df_ohe = ohe.fit_transform(X[ohe_cols])
        df_ohe = pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out(ohe_cols))

        df_new = pd.concat([X, df_ohe], axis=1)
        df_new.drop(ohe_cols, axis=1, inplace=True)
        return df_new

    def _labelenc_transform(self, X):
        labelenc_cols = X.select_dtypes(include=['object']).columns.tolist()
        lblenc = LabelEncoder() # to handle the unseen category as other
        for col in labelenc_cols:
            X[col] = lblenc.fit_transform(X[col])


    def _aggregate(self, X):
        pass

    def _preprocess(self, X):
        pass

    def _drop_columns(self, X):
        cols_drop = []
        for col in self.cols_drop:   # Only drop those columns still exist in the X
            if col in X.columns:
                cols_drop.append(col)

        X.drop(cols_drop, axis=1, inplace=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Transforming data")
        if self._level == 1:
            self._init_clean(X)
            self._impute(X)
            self._boolean_column_transform(X)
            self._aggregate(X)


        self._boolean_column_transform(X)
        # X = self._add_datetime_new(X)
        # self._cyclic_enc_transform(X)
        self._drop_columns(X)
        X = self._ohe_transform(X)

        return X
