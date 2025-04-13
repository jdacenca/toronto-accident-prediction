"""Data cleaning transformer for accident data."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTENC

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
    # cols_drop3 = ['ROAD_CLASS','DISTRICT','ACCLOC','INVAGE','INVTYPE',
    #               'PEDCOND','CYCCOND','PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE',
    #               'TRSN_CITY_VEH','EMERG_VEH','PASSENGER','SPEEDING','AG_DRIV','NEIGHBOURHOOD_158']
    # cols_drop4 = ['ACCLOC', 'INVTYPE']
    # cols_drop4 = ['INVTYPE']
    cols_drop4 = ['INVTYPE', 'INVAGE', 'PEDCOND', 'CYCCOND','NEIGHBOURHOOD_158', 'IMPACTYPE']
    cols_drop5 = [ 'NEIGHBOURHOOD_158',
                    'ALCOHOL',
                    'WEEK',
                    'AG_DRIV',
                    'AUTOMOBILE',
                    'DAY',
                    'VISIBILITY',
                    'EMERG_VEH',
                    'CYCCOND',
                    'DAYOFWEEK',
                    'MONTH',
                    'DISABILITY',
                    'REDLIGHT',
                    'MOTORCYCLE',
                    'PASSENGER',
                    'ROAD_CLASS',
                    'PEDCOND',
                    'RDSFCOND',]
    cols_drop6 = ['PEDCOND',
                     'ACCLOC',
                     'HOUR',
                     'TRAFFCTL',
                     'WEEK',
                     'VISIBILITY',
                     'TRSN_CITY_VEH',
                     'LONGITUDE',
                     'AUTOMOBILE',
                     'MONTH',
                     'CYCLIST',
                     'DAY',
                     'RDSFCOND',
                     'CYCCOND',
                     'DAYOFWEEK',
                     'MOTORCYCLE',
                     'PASSENGER',
                     'ROAD_CLASS',
                     'REDLIGHT',
                     'INVTYPE',
                     'ALCOHOL',
                     'EMERG_VEH',
                     'DISABILITY']# Based on SHAP importance
    cols_drop.extend(cols_drop2)
    # cols_drop.extend(cols_drop3)
    # cols_drop.extend(cols_drop4)
    # cols_drop.extend(cols_drop5)
    cols_drop.extend(cols_drop6)


    def __init__(self, level = 2):
        self._level = level # Level 1, performs all the clean-up, imputation and encoding, sampling
                            # Level 2, only performs the minimal required encoding
    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value


    def _init_clean(self, df):
        print("Preprocessing data : _init_clean")
        ###############
        # Clean up DATE and TIME columns to correct type

        # Only extract date
        df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y %I:%M:%S %p').dt.normalize()

        # Original 'TIME' column is 'int64' so 0006 is 6 which is an issue for to_datetime()
        # Fix the format with leading zero with zfill()
        df['TIME'] = df['TIME'].apply(lambda x: str(x).zfill(4))
        df['TIME'] = pd.to_datetime(df['TIME'], format='%H%M').dt.time

        ###############
        # Clean up the ROAD_CLASS for trailing space
        df['ROAD_CLASS'] = df['ROAD_CLASS'].str.strip()

    def _district_fit(self, df):
        print("Preprocessing data : _district_fit")
        # Fit: Get the most frequent DISTRICT per HOOD_158
        self.hood_to_district_mode = df.groupby('HOOD_158')['DISTRICT'].agg(lambda x: x.mode()[0])

    def _impute(self, df):
        print("Preprocessing data : _impute")
        ###############
        # Impute the empty ACCNUM from date, time, longtitue and latitude

        datetime = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
        long_lat = df[['LONGITUDE', 'LATITUDE']].astype(str).agg('_'.join, axis=1)
        datetime_long_lat = datetime.astype(str) + "_" + long_lat
        df['ACCNUM'] = df['ACCNUM'].fillna(datetime_long_lat).astype(str)

        ###############
        # Impute the ACCLASS
        df['ACCLASS'] = df['ACCLASS'].fillna(df['INJURY'])
        df['ACCLASS'] = df['ACCLASS'].replace('Property Damage O', 'Non-Fatal Injury')   # Or dropped

        ###############
        # Impute the missing DISTRICT based on HOOD_158, NEIGHBOURHOOD_158
        empty_district_index = df['DISTRICT'].isna()
        # X['DISTRICT'] = X.groupby('HOOD_158')['DISTRICT'].transform(lambda x: x.fillna(x.mode()[0]))
        df['DISTRICT'].fillna(df['HOOD_158'].map(self.hood_to_district_mode))


        ###############
        # Impute blank with 'Other'
        other_columns = ['ROAD_CLASS', 'ACCLOC', 'TRAFFCTL','VISIBILITY',
                         'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE']

        df[other_columns] = df[other_columns].fillna("Other")

        ###############
        # Impute blank with 'Unknown'
        unknown_columns = ['PEDCOND', 'CYCCOND']
        df[unknown_columns] = df[unknown_columns].fillna("Unknown")

        ###############
        # Perform binary column transformation inside impute
        # self._binary_column_transform(X)

    def _boolean_column_transform(self, df):
        print("Preprocessing data : _boolean_column_transform")
        ###############
        # Binary column imputation with True / False
        col_boolean = ['ACCLASS', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH', 'PASSENGER',
                      'SPEEDING', 'AG_DRIV', 'ALCOHOL', 'DISABILITY', 'REDLIGHT', 'TRSN_CITY_VEH']

        for col in col_boolean:
            df[col] = df[col].map(
                {'Yes': True, 'No': False, np.nan: False, 'Fatal': True, 'Non-Fatal Injury': False, True: True, False: False})

        print(df['ACCLASS'].value_counts())

    def _add_datetime_new(self, df):
        print("Preprocessing data : _add_datetime_new")
        df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
        df = df.assign(
            # YEAR=X['DATETIME'].dt.year,
            MONTH=df['DATETIME'].dt.month.astype(int),
            WEEK=df['DATETIME'].dt.isocalendar().week.astype(int),
            DAY = df['DATETIME'].dt.day.astype(int),
            DAYOFWEEK=df['DATETIME'].dt.dayofweek.astype(int),
            HOUR=df['DATETIME'].dt.hour.astype(int)
        )
        # df.drop(columns=['DATE', 'TIME','DATETIME'], inplace = True)
        df.drop(columns=['DATETIME','DATE','TIME'], inplace = True)
        return df

    def _smote_debug(self, df):
        print("Preprocessing data : _smote_debug")
        for i, col in enumerate(df.columns):
            try:
                print(f"Testing column: {col} (index {i})")
                _ = np.array(df[col], dtype=np.uint32)
            except Exception as e:
                print(f"❌ Column '{col}' at index {i} caused an issue: {e}")

    def _apply_smote(self, df):
        print("Preprocessing data : _apply_smote")
        """Apply SMOTENC on combined [X, y] array"""
        orig_cat_features = [ 'ROAD_CLASS', 'DISTRICT', 'ACCLOC','TRAFFCTL',
                                'VISIBILITY','LIGHT','RDSFCOND','IMPACTYPE',
                                'INVTYPE','INVAGE','PEDCOND','CYCCOND',
                                'NEIGHBOURHOOD_158']
        features_to_drop = ['PEDCOND',
                     'ACCLOC',
                     'HOUR',
                     'TRAFFCTL',
                     'WEEK',
                     'VISIBILITY',
                     'TRSN_CITY_VEH',
                     'LONGITUDE',
                     'AUTOMOBILE',
                     'MONTH',
                     'CYCLIST',
                     'DAY',
                     'RDSFCOND',
                     'CYCCOND',
                     'DAYOFWEEK',
                     'MOTORCYCLE',
                     'PASSENGER',
                     'ROAD_CLASS',
                     'REDLIGHT',
                     'INVTYPE',
                     'ALCOHOL',
                     'EMERG_VEH',
                     'DISABILITY']

        # categorical_features = [ 'ROAD_CLASS',
        #                         'DISTRICT',
        #                         'ACCLOC',
        #                         'TRAFFCTL',
        #                         'VISIBILITY',
        #                         'LIGHT',
        #                          'RDSFCOND',
        #                         'IMPACTYPE',
        #                          'INVTYPE',
        #                         'INVAGE',
        #                          'PEDCOND',
        #                          'CYCCOND',
        #                         'NEIGHBOURHOOD_158'
        #                         ]
        categorical_features =[feature for feature in orig_cat_features if feature not in features_to_drop]
        X_df, y = df.drop(['ACCLASS'], axis=1), df['ACCLASS']
        cat_indices = [X_df.columns.get_loc(col) for col in categorical_features]

        smote = SMOTENC(categorical_features=cat_indices, random_state=54)
        X_res, y_res = smote.fit_resample(X_df, y)
        y_res = pd.Series(y_res, name=y.name)  # Preserve the original name
        X_res = pd.DataFrame(X_res, columns=X_df.columns)  # Preserve column names

        # return np.hstack((X_res, y_res.values.reshape(-1, 1)))
        return pd.concat([X_res, y_res], axis=1)

    # Cyclic Encoding for cyclic columns extracted from DATETIME
    def _cyclic_enc(self, df, column, max_value):
        print("Preprocessing data : _cyclic_enc")
        df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
        df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
        # return df

    def _cyclic_enc_transform(self, df):
        print("Preprocessing data : _cyclic_enc_transform")
        col_cyclic_encoding = ['MONTH', 'WEEK', 'DAY', 'DAYOFWEEK', 'HOUR']
        max_cyclic_encoding = [12, 53, 31, 7, 24]
        for col, max in zip(col_cyclic_encoding, max_cyclic_encoding):
            # print(f"{col} -> {max}")
            self._cyclic_enc(df, col, max)
        df.drop(col_cyclic_encoding, axis=1, inplace=True)
        # return df

    def _ohe_transform(self, df):
        print("Preprocessing data : _ohe_transform")
        ohe_cols = df.select_dtypes(include=['object']).columns.tolist()
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # to handle the unseen category as other
        df_ohe = ohe.fit_transform(df[ohe_cols])
        df_ohe = pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out(ohe_cols))

        df_new = pd.concat([df, df_ohe], axis=1)
        df_new.drop(ohe_cols, axis=1, inplace=True)
        return df_new

    def _labelenc_transform(self, df):
        print("Preprocessing data : _labelenc_transform")
        labelenc_cols = df.select_dtypes(include=['object']).columns.tolist()
        lblenc = LabelEncoder()
        for col in labelenc_cols:
            df[col] = lblenc.fit_transform(df[col])

    def _standard_scaler(self, df, numeric_features):
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

    def _aggregate(self, df):
        pass

    def _preprocess(self, df):
        pass

    def _drop_columns(self, df):
        print("Preprocessing data : _drop_columns")
        cols_drop = []
        for col in self.cols_drop:   # Only drop those columns still exist in the X
            if col in df.columns:
                cols_drop.append(col)

        df.drop(cols_drop, axis=1, inplace=True)

    def fit(self, df, y=None):
        print("Preprocessing data : fit")
        self._district_fit(df)
        return self

    def transform(self, df, y=None):
        print("Preprocessing data : Transform")
        data = df.copy()
        # if self._level == 1:
        self._init_clean(data)
        self._impute(data)
        self._boolean_column_transform(data)
        # self._aggregate(data)
        # self._boolean_column_transform(data)

        data = self._add_datetime_new(data)
        # self._cyclic_enc_transform(X)

        self._drop_columns(data)

        # Perform SMOTENC to the training data (level = 1)
        if self._level == 1:
            self._smote_debug(data)
            data = self._apply_smote(data)

        # X = self._ohe_transform(X)
        self._labelenc_transform(data)

        numeric_features = data.select_dtypes(include=[np.number]).columns
        self._standard_scaler(data, numeric_features)
        return data
