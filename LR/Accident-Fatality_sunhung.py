# Student Name: Sun Hung Tsang
# Student ID: 301329154
# 25W --Supervised Learning (SEC. 003)

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV




# %%
# Load the data

if '__file__' in globals():
    filepath = Path(__file__).parent
else:
    filepath = Path(r'./')

filepath = filepath / 'TOTAL_KSI_6386614326836635957.csv'
df_ksi = pd.read_csv(filepath)

# %%

pd.set_option('display.max_columns', None)

df_ksi.info()
df_ksi.describe()

df_ksi.isnull().sum()
df_ksi['ACCLASS'].value_counts()
df_ksi['ACCLASS'].value_counts(normalize=True, dropna=False)

# # sns.swarmplot(x="ALCOHOL", y="ACCLASS", data=df_ksi)
# # plt.show()
# ct_ksi1 = pd.crosstab(df_ksi['ACCLASS'], df_ksi['ALCOHOL'])
# sns.heatmap(ct_ksi1, cmap='YlGnBu')
# plt.show()
#
# # sns.countplot(x="ACCLASS", hue="ALCOHOL", data=df_ksi)
# sns.countplot(hue="ACCLASS", x="ALCOHOL", data=df_ksi)
# plt.show()
#
# df_ksi[['ACCLASS', 'ALCOHOL']].value_counts(dropna=False)

# %%
# Data Clean-Up

# DATE and TIME columns
df_ksi['DATE'] = pd.to_datetime(df_ksi['DATE'], format='%m/%d/%Y %I:%M:%S %p').dt.normalize()
# Only extract date

# Original 'TIME' column is 'int64' so 0006 is 6 which is an issue for to_datetime()
# Fix the format with leading zero with zfill()
df_ksi['TIME'] = df_ksi['TIME'].apply(lambda x: str(x).zfill(4))
df_ksi['TIME'] = pd.to_datetime(df_ksi['TIME'],format='%H%M').dt.time

df_ksi['DATETIME'] = pd.to_datetime(df_ksi['DATE'].astype(str) + ' ' + df_ksi['TIME'].astype(str))
df_ksi.drop(columns=['DATE', 'TIME'], inplace = True)


# Impute the empty ACCNUM

datetime_long_lat = df_ksi[['DATETIME','LONGITUDE','LATITUDE']].astype(str).agg('_'.join, axis = 1)
df_ksi['ACCNUM'] = df_ksi['ACCNUM'].fillna(datetime_long_lat)

# Impute the ACCLASS
df_ksi['ACCLASS'] = df_ksi['ACCLASS'].fillna(df_ksi['INJURY'])
df_ksi['ACCLASS'] = df_ksi['ACCLASS'].replace('Property Damage O','Non-Fatal Injury')

# Add new components of time to the dataframe
df_ksi = df_ksi.assign(
    YEAR = df_ksi['DATETIME'].dt.year,
    MONTH = df_ksi['DATETIME'].dt.month,
    WEEK = df_ksi['DATETIME'].dt.isocalendar().week,
    DAYOFWEEK = df_ksi['DATETIME'].dt.dayofweek,
    HOUR = df_ksi['DATETIME'].dt.hour
)

# %%
# Drop the columns that are not required

cols_to_drop = ['OBJECTID','INDEX','STREET1','STREET2','OFFSET','INVTYPE','INVAGE','INJURY','FATAL_NO','INITDIR','MANOEUVER','DRIVACT','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND','HOOD_158','HOOD_140','NEIGHBOURHOOD_140','DIVISION','x','y']

cols_to_drop.extend(['VEHTYPE','DRIVCOND','IMPACTYPE'])

df_ksi.drop(cols_to_drop, axis = 1, inplace = True)

# %%
# Feature Engineering


# %%
# Perform the transformation by aggregation

cols_to_chk = df_ksi.columns.values.tolist()
cols_to_chk.remove('ACCNUM')

df_ksi_agg = df_ksi.groupby('ACCNUM')

# ['ACCNUM', 'ROAD_CLASS', 'DISTRICT', 'LATITUDE', 'LONGITUDE', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'NEIGHBOURHOOD_158', 'DATETIME', 'YEAR', 'MONTH', 'WEEK', 'DAYOFWEEK', 'HOUR']

def unique_or_value(series):
    unique_vals = series.unique()
    return unique_vals[0] if len(unique_vals) == 1 else unique_vals

df_ksi_aggnew = df_ksi_agg.agg(
    **{col: (col, unique_or_value) for col in cols_to_chk}
).reset_index()

df_ksi_aggnew.drop('ACCNUM', axis = 1, inplace = True)
# df_ksi_aggnew_nuique = df_ksi_aggnew[df_ksi_aggnew.gt(1).any(axis=1)]

# Using applymap to check for lists (or arrays)
# check_is_list = df_ksi_aggnew[cols_to_chk].map(lambda x: isinstance(x, (list, np.ndarray)))
check_is_list = df_ksi_aggnew[cols_to_chk].apply(lambda col: col.apply(lambda x: isinstance(x, (list, np.ndarray))))

# Output the result: display which cells contain lists or arrays
print("Cells that contain lists or arrays:")
# print(check_is_list)

# Stack the DataFrame to filter rows where lists/arrays are present
stacked_check = check_is_list.stack()  # Flatten the DataFrame into a Series

# Filter for rows that contain lists/arrays
filtered_stacked_check = stacked_check[stacked_check]  # Only keep True values

# To get row and column information where lists/arrays exist:
print("\nRows and columns with lists or arrays:")
print(filtered_stacked_check)

# %%
# Perform other clean up and encoding

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

col_ohe = ['ROAD_CLASS','DISTRICT','ACCLOC','TRAFFCTL','VISIBILITY','LIGHT','RDSFCOND','NEIGHBOURHOOD_158']
col_cyclic_encoding = ['MONTH','WEEK','DAYOFWEEK','HOUR']
max_cyclic_encoding = [12,53,7,24]
col_binary = ['ACCLASS','PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','EMERG_VEH','PASSENGER','SPEEDING','AG_DRIV','ALCOHOL','DISABILITY','REDLIGHT','TRSN_CITY_VEH']
col_numeric = ['LATITUDE','LONGITUDE']
col_target = ['ACCLASS']
col_to_drop2 = ['DATETIME','YEAR']

# Custom Mapping for binary columns
for col in col_binary:
    df_ksi_aggnew[col] = df_ksi_aggnew[col].map({'Yes':True, 'No':False, np.nan:False, 'Fatal': True, 'Non-Fatal Injury': False})

# Simple Imputation
# simputer = SimpleImputer(strategy='constant', fill_value=False)
# df_ksi_aggnew[col_binary] =simputer.fit_transform(df_ksi_aggnew[col_binary])

# One Hot encoding

ohe = OneHotEncoder(sparse_output = False)
df_ksi_aggnew_ohe = ohe.fit_transform(df_ksi_aggnew[col_ohe])
df_ksi_aggnew_ohe = pd.DataFrame(df_ksi_aggnew_ohe, columns = ohe.get_feature_names_out(col_ohe))

df_ksi_aggnew = pd.concat([df_ksi_aggnew, df_ksi_aggnew_ohe], axis = 1)
df_ksi_aggnew.drop(col_ohe, axis = 1, inplace = True)

# Cyclic Encoding for cyclic columns extracted from DATETIME
def cyclic_enc(df, column, max_value):
    df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df

for col, max in zip(col_cyclic_encoding, max_cyclic_encoding):
    # print(f"{col} -> {max}")
    cyclic_enc(df_ksi_aggnew, col, max)
df_ksi_aggnew.drop(col_cyclic_encoding, axis = 1, inplace = True)

df_ksi_aggnew.drop(col_to_drop2, axis = 1, inplace = True)

# %%
# Now split the data

X_train, X_test, y_train, y_test = train_test_split(df_ksi_aggnew.drop('ACCLASS', axis=1 ), df_ksi_aggnew['ACCLASS'], test_size = 0.2, random_state = 54)

# %%
# Build the model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 54)
log_reg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=10000, random_state=54)

# Train the model
# log_reg.fit(X_train, y_train)

cv_score_default = cross_val_score(log_reg, X_train, y_train, cv = skf, scoring = 'accuracy')
print(cv_score_default)
print(cv_score_default.mean())
# cv_scores = cross_val_score(log_reg, X_train, y_train, cv=10, scoring='accuracy')

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1: {f1_score(y_test, y_pred)*100:.2f}%")

# log_reg.score(X_train, y_train)
# log_reg.score(X_test, y_test)

# %%
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from collections import Counter

print("Random Over Sampling")
ros = RandomOverSampler(random_state=54)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
cv_score_oversample = cross_val_score(log_reg, X_resampled, y_resampled, cv = skf, scoring = 'accuracy')
print(cv_score_oversample)
print(cv_score_oversample.mean())

log_reg.fit(X_resampled, y_resampled)
y_pred = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1: {f1_score(y_test, y_pred)*100:.2f}%")

# %%
from imblearn.under_sampling import RandomUnderSampler

print("Random Under Sampling")
# Perform undersampling
rus = RandomUnderSampler(random_state=54)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

cv_score_undersample = cross_val_score(log_reg, X_resampled, y_resampled, cv = skf, scoring = 'accuracy')
print(cv_score_undersample)
print(cv_score_undersample.mean())

log_reg.fit(X_resampled, y_resampled)
y_pred = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1: {f1_score(y_test, y_pred)*100:.2f}%")


# %%
from imblearn.over_sampling import SMOTE

print("SMOTE Over Sampling")
# Perform SMOTE
smote = SMOTE(random_state=54)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

cv_score_smote = cross_val_score(log_reg, X_resampled, y_resampled, cv = skf, scoring = 'accuracy')
print(cv_score_smote)
print(cv_score_smote.mean())

log_reg.fit(X_resampled, y_resampled)
y_pred = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1: {f1_score(y_test, y_pred)*100:.2f}%")

Counter(X_resampled)


