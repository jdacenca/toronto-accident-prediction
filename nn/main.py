import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helper import clean_dataset, data_description, unique_values
from undersampling_helper import random_sampling
from sklearn.model_selection import train_test_split
from neural_network import mlp_classifier

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

cleaned_df = clean_dataset(ksi_df, dropped_fields)
categorical_columns = cleaned_df.select_dtypes(include=[object, 'category']).columns.tolist()

unique_values(cleaned_df)

# Remove duplicates
cleaned_df = cleaned_df.drop_duplicates()

print("Cleaned data:")
data_description(cleaned_df)

features = cleaned_df.drop(columns=["ACCLASS"], axis=1)
target = cleaned_df["ACCLASS"]

x_random_sampling, y_random_sampling = random_sampling(features, target)
X_train, X_test, y_train, y_test = train_test_split(x_random_sampling, y_random_sampling, test_size=0.2, random_state=17)

mlp_classifier(X_train, X_test, y_train, y_test)