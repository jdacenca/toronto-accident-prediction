import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Map month numbers to seasons
def month_to_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

# Clean and preprocess the data
def data_cleaning(data_ksi):
    data_ksi = data_ksi.copy()

    # Drop unnecessary columns
    #data_ksi.drop(columns=columns_to_drop, inplace=True)
    
    # Handle missing target values and specific rows
    #data_ksi['ACCLASS'] = data_ksi['ACCLASS'].fillna('Fatal')
    #data_ksi.drop(data_ksi[data_ksi['ACCLASS'] == 'Property Damage O'].index, inplace=True)
    #data_ksi.drop_duplicates(inplace=True)

    #data_ksi.drop(columns=['ACCNUM'], inplace=True)
    
    # Format date and time
    #data_ksi["DATE"] = pd.to_datetime(data_ksi["DATE"]).dt.to_period("D").astype(str)
    
    # Extract date components from the 'DATE' column
    #data_ksi['MONTH'] = pd.to_datetime(data_ksi['DATE']).dt.month
    
    # Extract season
    data_ksi['SEASON'] = data_ksi['MONTH'].apply(month_to_season).astype(float)
    
    # Replace specific values
    #data_ksi['ROAD_CLASS'] = data_ksi['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False)
    
    # Fill missing values
    unknown_columns = ['PEDCOND', 'DRIVCOND', 'MANOEUVER', 'CYCACT',
                        'VEHTYPE', 'INVTYPE', 'IMPACTYPE', 'DISTRICT', 'INITDIR','TRAFFCTL']
    other_columns = ['ROAD_CLASS', 'ACCLOC', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'DRIVACT']
    boolean_columns = ['PEDESTRIAN', 'CYCLIST', 'MOTORCYCLE',
                       'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'TRSN_CITY_VEH', 'DISABILITY','AUTOMOBILE','TRUCK']

    categorical_columns = [
        'ROAD_CLASS', 'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT',
        'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INITDIR', 'VEHTYPE', 'MANOEUVER',
        'DRIVACT', 'DRIVCOND', 'PEDCOND', 'CYCACT','ACCLASS'
    ]

    data_ksi[categorical_columns] = data_ksi[categorical_columns].apply(lambda col: col.str.upper())
    data_ksi[other_columns] = data_ksi[other_columns].fillna("OTHER")
    data_ksi[unknown_columns] = data_ksi[unknown_columns].fillna("UNKNOWN")
    
    data_ksi[boolean_columns] = data_ksi[boolean_columns].fillna("No")
    
    data_ksi['INVAGE'] = data_ksi['INVAGE'].fillna("unknown")

    # Handle age column
    data_ksi['INVAGE'] = data_ksi['INVAGE'].replace('unknown', np.nan)
    data_ksi['INVAGE'] = data_ksi['INVAGE'].str.replace('OVER 95', '95 to 100')
    data_ksi[['min_age', 'max_age']] = data_ksi['INVAGE'].str.split(' to ', expand=True)
    data_ksi['min_age'] = pd.to_numeric(data_ksi['min_age'], errors='coerce')
    data_ksi['max_age'] = pd.to_numeric(data_ksi['max_age'], errors='coerce')
    data_ksi['AVG_AGE'] = data_ksi[['min_age', 'max_age']].mean(axis=1).astype(float)
    
    data_ksi.drop(columns=['INVAGE','min_age', 'max_age'], inplace=True)

    data_ksi['INVAGE'] = data_ksi['AVG_AGE'].fillna(data_ksi['AVG_AGE'].mean()).astype(float)

    # Convert boolean columns to numeric
    pd.set_option('future.no_silent_downcasting', True)
    data_ksi[boolean_columns] = data_ksi[boolean_columns].replace({'Yes': 1, 'No': 0}).astype(float)
    
    data_ksi["DIVISION"] = data_ksi["DIVISION"].replace('NSA', '00').str[1:].astype(float)

    # Replace 'NSA' with '00' and convert HOOD columns to float
    for col in ['HOOD_158', 'HOOD_140']:
        data_ksi[col] = data_ksi[col].replace('NSA', '00').astype(float)

    # Convert LATITUDE and LONGITUDE to float
    data_ksi[['LATITUDE', 'LONGITUDE']] = data_ksi[['LATITUDE', 'LONGITUDE']].astype(float)

    data_ksi['HOUR'] = data_ksi['TIME'].apply(lambda x: f"{int(x) // 100:02d}" if x >= 100 else '00')  # Extract hours for 3 or 4 digits
    data_ksi['MINUTE'] = data_ksi['TIME'].apply(lambda x: f"{int(x) % 100:02d}" if x >= 100 else f"{int(x):02d}")  # Extract minutes
    
    data_ksi['HOUR'] = data_ksi['HOUR'].astype(int)
    data_ksi['MINUTE'] = data_ksi['MINUTE'].astype(int)
    data_ksi.drop(columns=['TIME','DATE','AVG_AGE'], inplace=True)

    return data_ksi