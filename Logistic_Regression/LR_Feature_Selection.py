import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

# %%
# Load the data
def load_data():
    """Load csv file into dataframe"""
    if '__file__' in globals():
        filepath = Path(__file__).parent / "data/"
    else:
        filepath = Path(r'./data/')

    filepath = filepath / 'TOTAL_KSI_6386614326836635957.csv'
    df_ksi = pd.read_csv(filepath)

    return df_ksi

# %%
def init_clean(data):
    df = data.copy()
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
    return df

# %%
def impute(data):
    print("Preprocessing data : _impute")
    df = data.copy()
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
    hood_to_district_mode = df.groupby('HOOD_158')['DISTRICT'].agg(lambda x: x.mode()[0])
    # X['DISTRICT'] = X.groupby('HOOD_158')['DISTRICT'].transform(lambda x: x.fillna(x.mode()[0]))
    df['DISTRICT'].fillna(df['HOOD_158'].map(hood_to_district_mode))


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

    return df

#%%
def boolean_column_transform(data):
    df = data.copy()
    print("Preprocessing data : _boolean_column_transform")
    ###############
    # Binary column imputation with True / False
    col_boolean = ['ACCLASS', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH', 'PASSENGER',
                  'SPEEDING', 'AG_DRIV', 'ALCOHOL', 'DISABILITY', 'REDLIGHT', 'TRSN_CITY_VEH']

    for col in col_boolean:
        df[col] = df[col].map(
            {'Yes': True, 'No': False, np.nan: False, 'Fatal': True, 'Non-Fatal Injury': False, True: True, False: False})

    print(df['ACCLASS'].value_counts())
    return df

# %%
def add_datetime_new(df):
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

# %%
def labelenc_transform(data):
    df = data.copy()
    print("Preprocessing data : _labelenc_transform")
    labelenc_cols = df.select_dtypes(include=['object']).columns.tolist()
    lblenc = LabelEncoder()
    for col in labelenc_cols:
        df[col] = lblenc.fit_transform(df[col])

    return df, labelenc_cols

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2, SelectKBest

def chi2_feature_importance(df, categorical_features, target_col, top_n=None, plot=True):
    """
    Perform Chi-squared test on categorical features and visualize importance.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        categorical_features (list): List of categorical feature names.
        target_col (str): Name of the target variable (must be binary or non-negative integers).
        top_n (int or None): Number of top features to return/plot. If None, all features are shown.
        plot (bool): Whether to show a bar plot of feature importances.

    Returns:
        pd.DataFrame: Sorted dataframe of features and their Chi-squared scores.
    """
    # Step 1: One-hot encode categorical variables
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(df[categorical_features])
    feature_names = encoder.get_feature_names_out(categorical_features)

    # Step 2: Chi-squared test
    selector = SelectKBest(score_func=chi2, k='all')
    selector.fit(X_encoded, df[target_col])
    scores = selector.scores_

    # Step 3: Combine scores into a dataframe
    result_df = pd.DataFrame({
        'Feature': feature_names,
        'Chi2 Score': scores
    }).sort_values(by='Chi2 Score', ascending=False)

    # Step 4: Optional plotting
    if plot:
        plot_df = result_df if top_n is None else result_df.head(top_n)
        plt.figure(figsize=(10, 0.4 * len(plot_df)))
        sns.barplot(x='Chi2 Score', y='Feature', data=plot_df, palette='viridis')
        plt.title('Chi-squared Scores for Categorical Features')
        plt.xlabel('Chi-squared Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    return result_df.reset_index(drop=True)


# %%
def chi_square_test(data, labelenc_cols):
    df = data.copy()
    from sklearn.feature_selection import chi2, SelectKBest

    X = df[labelenc_cols]
    y = df['ACCLASS']

    selector = SelectKBest(score_func=chi2, k='all')
    X_new = selector.fit_transform(X, y)

    feature_scores = selector.scores_
    selected_features = X.columns[selector.get_support()]

    feature_scores_df = pd.DataFrame({
        'Feature': X.columns,
        'Chi2 Score': selector.scores_,
        'Selected': selector.get_support()
    })

    # Optional: sort by score
    feature_scores_df = feature_scores_df.sort_values(by='Chi2 Score', ascending=False)

    print(feature_scores_df)

    # print("Feature Scores:", feature_scores)
    # print("Selected Features:", selected_features)

# %%
data_ksi = load_data()
df_ksi1 = init_clean(data_ksi)
df_ksi2 = impute(df_ksi1)
df_ksi3 = boolean_column_transform(df_ksi2)
df_ksi4 = add_datetime_new(df_ksi3)
df_ksi5, cat_labels = labelenc_transform(df_ksi4)
df_ksi6 = chi_square_test(df_ksi4, cat_labels)

cols_drop = [
    'INDEX', 'OBJECTID', 'STREET1', 'STREET2',
    'INJURY', 'INITDIR',
    'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND',
    'PEDTYPE', 'PEDACT',
    'CYCLISTYPE', 'CYCACT',
    'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION',
    'HOOD_158',
    'FATAL_NO',
    'OFFSET', 'x', 'y'
]
# cols_drop2 = ['ACCNUM','DATE','TIME']
cols_drop2 = ['ACCNUM']
cols_drop.extend(cols_drop2)

df_ksi4_drop = df_ksi4.drop(columns=cols_drop)

numeric_cols = ['LATITUDE', 'LONGITUDE']

df_ksi5_drop, cat_labels = labelenc_transform(df_ksi4_drop)
df_ksi5_drop_col = df_ksi5_drop.columns.tolist()

for col in numeric_cols:
    df_ksi5_drop_col.remove(col)

chi2_feature_importance(df_ksi3, df_ksi5_drop_col, 'ACCLASS', top_n=31, plot=True)

df_ksi6 = chi_square_test(df_ksi5_drop, df_ksi5_drop_col)