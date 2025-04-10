import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helper import clean_dataset, data_description, unique_values, select_best_features, recursive_feature_elimination, variance_threshold
from sampling_helper import random_sampling, tomek_links, near_miss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

np.random.seed(1)

# Default values
filename = 'TOTAL_KSI_6386614326836635957.csv'
input_directory = "../data/"
output_directory = "./output/"

ksi_df = pd.read_csv(f'{input_directory}{filename}')

print("Raw data:")
data_description(ksi_df)

# Clean the data
# Drop fields
dropped_fields = [
    'INDEX',
    'ACCNUM',
    'OBJECTID',
    'STREET1',
    'STREET2',
    'DISTRICT',
    'LATITUDE',
    'LONGITUDE',
    'ACCLOC',
    'INVTYPE',
    'INVAGE',
    'INJURY',
    'OFFSET',
    'FATAL_NO',
    #'VEHTYPE',
    'INITDIR',
    'MANOEUVER',
    'DRIVACT',
    'DRIVCOND',
    'PEDTYPE',
    'PEDACT',
    'PEDCOND',
    'CYCLISTYPE',
    'CYCACT',
    'CYCCOND',
    'HOOD_158',
    'HOOD_140',
    'NEIGHBOURHOOD_140',
    'DIVISION',
    'x',
    'y',
]

df, cleaned_df = clean_dataset(ksi_df, dropped_fields)# Checking the Feature scores

print(df.isnull().sum())
cat = df.select_dtypes(include=[object, 'category']).columns.tolist()
df[cat] = df[cat].astype('category')
encoder = OrdinalEncoder()
df[cat] = encoder.fit_transform(df[cat])
x = df.drop(columns=["ACCLASS"], axis=1)
y = df["ACCLASS"]

print("\nFeature Scores:")
select_best_features(x, y, df.columns)

recursive_feature_elimination(x, y)

variance_threshold(x, y)

categorical_columns = cleaned_df.select_dtypes(include=[object, 'category']).columns.tolist()
print("Number of Unique Counts", cleaned_df.nunique())
unique_values(cleaned_df)

# Remove duplicates
cleaned_df = cleaned_df.drop_duplicates()

print("Cleaned data:")
data_description(cleaned_df)
########################################################################################

# Binary mapping
binary_mapping = {'YES': 1, 'NO': 0}
target_mapping = {'FATAL': 1, 'NON-FATAL INJURY': 0}
column_binary = [
    'PEDESTRIAN',
    'CYCLIST',
    'AUTOMOBILE',
    'MOTORCYCLE',
    'TRUCK',
    'TRSN_CITY_VEH',
    'EMERG_VEH',
    'PASSENGER',
    'SPEEDING',
    'AG_DRIV',
    'REDLIGHT',
    'ALCOHOL',
    'DISABILITY'
]
column_le = [
    'DATE',
    'ROAD_CLASS',
    'TRAFFCTL',
    'VISIBILITY',
    'LIGHT',
    'RDSFCOND',
    'IMPACTYPE',
    'VEHTYPE',
    'NEIGHBOURHOOD_158'
]

# Apply binary mapping using .loc
for column in column_binary:
    cleaned_df[column] = cleaned_df[column].map(lambda x: binary_mapping[x.upper()])

cleaned_df["ACCLASS"] = cleaned_df["ACCLASS"].replace(target_mapping)

le = LabelEncoder()

for column in column_le:
    cleaned_df[column] = le.fit_transform(cleaned_df[column])

# Remove 5 entries for each classification
unseen_fatal = cleaned_df[cleaned_df["ACCLASS"] == 1].head(5)
unseen_notfatal = cleaned_df[cleaned_df["ACCLASS"] == 0].head(5)
print(unseen_fatal)
print(unseen_notfatal)

# Remove unseen data from set
cleaned_df = cleaned_df.drop(unseen_fatal.index)
cleaned_df = cleaned_df.drop(unseen_notfatal.index)

features = cleaned_df.drop(columns=["ACCLASS"], axis=1)
target = cleaned_df["ACCLASS"]

# Random Sampling
x_random_sampling, y_random_sampling = random_sampling(features, target)
X_train, X_test, y_train, y_test = train_test_split(x_random_sampling, y_random_sampling, test_size=0.2, random_state=17)
#mlp_classifier("Random Sampling", X_train, X_test, y_train, y_test, unseen_fatal, unseen_notfatal)

# Tomek Links
x_tomek_links, y_tomek_links = tomek_links(features, target)
X_train, X_test, y_train, y_test = train_test_split(x_tomek_links, y_tomek_links, test_size=0.2, random_state=17)
#mlp_classifier("Random Sampling", X_train, X_test, y_train, y_test, unseen_fatal, unseen_notfatal)

# Near Miss
x_near_miss, y_near_miss = near_miss(features, target)
X_train, X_test, y_train, y_test = train_test_split(x_near_miss, y_near_miss, test_size=0.2, random_state=17)
#mlp_classifier("Random Sampling", X_train, X_test, y_train, y_test, unseen_fatal, unseen_notfatal)