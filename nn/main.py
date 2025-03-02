import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dynamic_visualizer import grouped_barplots
from helper import convert_to_time
from sklearn.model_selection import train_test_split

np.random.seed(1)

# Default values
filename = 'TOTAL_KSI_6386614326836635957.csv'
input_directory = "../data/"
output_directory = "./output/"

ksi_df = pd.read_csv(f'{input_directory}{filename}')

# Area Data
area_columns = [
    'STREET1',
    'STREET2',
    'OFFSET',
    'HOOD_158',
    'NEIGHBOURHOOD_158',
    'HOOD_140',
    'NEIGHBOURHOOD_140',
    'DIVISION',
    'x',
    'y'
]

# Save data for later
str_update = {
    ' STRE': ' ST',
    ' AV': ' AVE',
    ' ROAD': ' RD',
    ' AVEN': ' AVE',
    ' AVEE': ' AVE',
    ' AVEEN': ' AVE',
    ' AVEENUE': ' AVE',
    ' WAY': ' WY',
    ' EAST': ' E',
    ' WEST': ' W',
    ' BV': ' BLVD',
    ' RD.': ' RD',
}

area_data = ksi_df[area_columns]
area_data['STREET1'] = area_data['STREET1'].str.replace(r'\d+', '', regex=True)
area_data['STREET1'] = area_data['STREET1'].str.strip()
area_data['STREET1'] = area_data['STREET1'].str.replace(r'\s{2,}', ' ', regex=True)

for k, v in str_update.items():
    area_data['STREET1'] = area_data['STREET1'].str.replace(k, v, regex=True)

#-----------------------------------------------------------------------------------------
# Clean the data
# Understand the data 
print(ksi_df.head(5))
print(ksi_df.info())
print(ksi_df.describe())
print(ksi_df.isnull().sum())

# Drop fields
dropped_fields = [
    'INDEX',
    'OBJECTID',
    'STREET1',
    'STREET2',
    'DISTRICT',
    'OFFSET',
    'FATAL_NO',
    'HOOD_140',
    'NEIGHBOURHOOD_140',
    'DIVISION',
    'x',
    'y',
]

ksi_df.drop(dropped_fields, axis=1, inplace=True)
ksi_df.dropna(subset=['ACCLASS'], inplace=True)

# Fill in empty fields
boolean_columns = [
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

unknown_column = [
    'INITDIR',
    'VEHTYPE',
    'MANOEUVER',
    'DRIVCOND',
    'PEDCOND',
    'CYCLISTYPE',
    'CYCCOND',
    'TRAFFCTL',
    'INJURY'
]

other_column =  [
    'ROAD_CLASS',
    'ACCLOC',
    'VISIBILITY',
    'LIGHT',
    'RDSFCOND',
    'IMPACTYPE',
    'INVTYPE',
    'DRIVACT',
    'PEDACT',
    'CYCACT'
]

ksi_df[boolean_columns] = ksi_df[boolean_columns].fillna("No")
ksi_df[unknown_column] = ksi_df[unknown_column].fillna("Unknown")
ksi_df[other_column] = ksi_df[other_column].fillna("Other")
ksi_df['PEDTYPE'] = ksi_df['PEDTYPE'].fillna("OTHER / UNDEFINED")
categorical_columns = ksi_df.select_dtypes(include=[object, 'category']).columns.tolist()
#-----------------------------------------------------------------------------------------
# Updating the data
ksi_df[categorical_columns] = ksi_df[categorical_columns].apply(lambda col: col.str.upper())

ksi_df['DATE'] = pd.to_datetime(ksi_df['DATE']).dt.date
ksi_df['TIME'] = ksi_df['TIME'].apply(convert_to_time)
ksi_df['ROAD_CLASS'] = ksi_df['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False)

# Check all the unique data
for x in categorical_columns:
    print(f"\nUnique values in column {x}:")
    print(ksi_df[x].unique())

# Get back to this in checking the Street
#pd.DataFrame(ksi_df['STREET1'].unique()).to_csv("unique.csv")

# Check if there are anymore missing data
print(ksi_df.isnull().sum())


grouped_barplots(ksi_df, 'ACCLASS', 'VISIBILITY')

#-----------------------------------------------------------------------------------------
# Plot the data distribution
n_plots = len(ksi_df.columns)
n_cols = 3
n_rows = n_plots // n_cols + (n_plots % n_cols > 0)

numerical_fig, numerical_axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
numerical_axes = numerical_axes.flatten()       

categorical_fig, categorical_axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
categorical_axes = categorical_axes.flatten()
for i, col in enumerate(ksi_df.columns):
    if pd.api.types.is_numeric_dtype(ksi_df[col]):
        # Histogram
        sns.histplot(ksi_df[col], ax=numerical_axes[i], kde=True)
        numerical_axes[i].set_title(f'Distribution of {col}')

for i, col in enumerate(ksi_df.columns):
    if pd.api.types.is_object_dtype(ksi_df[col]):
        # Bar chart for categorical data
        sns.countplot(x=col, data=ksi_df, ax=categorical_axes[i])
        categorical_axes[i].set_title(f'Distribution of {col}')
        #categorical_axes[i].set_xticklabels(x=col, rotation=45)  # Rotate x-axis labels if needed

plt.tight_layout()

for p in range(len(ksi_df), len(numerical_axes)):
    numerical_fig.delaxes(numerical_axes[p])
plt.show()

for p in range(len(ksi_df), len(categorical_axes)):
    categorical_fig.delaxes(categorical_axes[p])
plt.show()


print("Done.")