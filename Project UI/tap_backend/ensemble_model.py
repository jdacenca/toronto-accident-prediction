import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Random seed for reproducibility
np.random.seed(17)

# ===================== LOAD AND CLEAN DATA =====================
data_ksi = pd.read_csv("./data/Total_KSI.csv")

# Drop unnecessary columns            
columns_to_drop = [ 'OBJECTID', 'INDEX',  # index_id 
    'FATAL_NO', # sequence No. - high missing values
    'OFFSET', #high missing values
    'x', 'y','CYCLISTYPE', 'PEDTYPE', 'PEDACT', # high correlation
    'EMERG_VEH',       # 0 permutation importance 
    'CYCCOND',         # 0 permutation importance 
    "NEIGHBOURHOOD_158","NEIGHBOURHOOD_140","STREET1","STREET2","INJURY" # based on feature importance
]

# ===================== MONTH TO SEASON =====================
# Map month numbers to seasons using the MONTH column
def month_to_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall
    
# ===================== DATA CLEANING =====================
def data_cleaning(df, columns_to_drop, class_imb='original'):
    """Cleans the dataset by handling missing values, dropping unnecessary columns, and balancing classes."""
    df2 = df.copy()

    # Drop unnecessary columns
    df2.drop(columns=columns_to_drop, inplace=True)

    # Handle missing target values and specific rows
    df2['ACCLASS'] = df2['ACCLASS'].fillna('Fatal')

    df2.drop(df2[df2['ACCLASS'] == 'Property Damage O'].index, inplace=True)
    df2.drop_duplicates(inplace=True)

    # Separate fatal rows
    fatal_rows = df2[df2['ACCLASS'] == 'Fatal']

    # Separate non-fatal rows
    non_fatal_rows = df2[df2['ACCLASS'] != 'Fatal']

    # Apply aggregation logic on non-fatal rows based on ACCNUM
    aggregated_data = non_fatal_rows.groupby(['ACCNUM'], as_index=False).apply(aggregate_rows, include_groups=False).reset_index(drop=True)

    # Combine the aggregated data with the fatal rows
    df2 = pd.concat([aggregated_data, fatal_rows], ignore_index=True)

    # Shuffle the combined data
    df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)

    # Format date and time
    df2["DATE"] = pd.to_datetime(df2["DATE"]).dt.to_period("D").astype(str)

    # Extract date components
    df2['MONTH'] = pd.to_datetime(df2['DATE']).dt.month
    df2['DAY'] = pd.to_datetime(df2['DATE']).dt.day

    # Extract season
    df2['SEASON'] = df2['MONTH'].apply(month_to_season).astype(float)

    # Replace specific values
    df2['ROAD_CLASS'] = df2['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False)

    # Fill missing values
    unknown_columns = ['PEDCOND', 'DRIVCOND', 'MANOEUVER', 'CYCACT',
                     'VEHTYPE', 'INVTYPE', 'IMPACTYPE', 'DISTRICT', 'INITDIR','INVAGE',"TRAFFCTL"]
    other_columns = ['ROAD_CLASS', 'ACCLOC', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'DRIVACT']
    boolean_columns = ['PEDESTRIAN', 'CYCLIST', 'MOTORCYCLE',
                    'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'TRSN_CITY_VEH', 'DISABILITY','AUTOMOBILE','TRUCK']

    df2[other_columns] = df2[other_columns].fillna("OTHER")
    df2[unknown_columns] = df2[unknown_columns].fillna("UNKNOWN")
    df2[boolean_columns] = df2[boolean_columns].fillna("No")

    # Convert boolean columns to numeric
    df2[boolean_columns] = df2[boolean_columns].replace({'Yes': 1, 'No': 0}).astype(float)

    # Handle age column
    df2['INVAGE'] = df2['INVAGE'].replace('Unknown', np.nan)
    df2['INVAGE'] = df2['INVAGE'].str.replace('OVER 95', '95 to 100')
    df2[['min_age', 'max_age']] = df2['INVAGE'].str.split(' to ', expand=True)
    df2['min_age'] = pd.to_numeric(df2['min_age'], errors='coerce')
    df2['max_age'] = pd.to_numeric(df2['max_age'], errors='coerce')
    df2['AVG_AGE'] = df2[['min_age', 'max_age']].mean(axis=1).astype(float)
    df2.drop(columns=['INVAGE','min_age', 'max_age'], inplace=True)
    df2['INVAGE'] = df2['AVG_AGE'].fillna(df2['AVG_AGE'].mean()).astype(float)

    # Extract hour from TIME
    df2['HOUR'] = df2['TIME'].apply(lambda x: f"{int(x) // 100:02d}" if x >= 100 else '00')  # Extract hours for 3 or 4 digits
    df2['MINUTE'] = df2['TIME'].apply(lambda x: f"{int(x) % 100:02d}" if x >= 100 else f"{int(x):02d}")  # Extract minutes

    df2['HOUR'] = df2['HOUR'].astype(int)
    df2['MINUTE'] = df2['MINUTE'].astype(int)

    df2["DIVISION"] = df2["DIVISION"].replace('NSA', '00').str[1:].astype(float)
    print("\n===================== DIVISION =====================")
    print(df2["DIVISION"])

    # Replace 'NSA' with '00' and convert HOOD columns to float
    for col in ['HOOD_158', 'HOOD_140']:
        df2[col] = df2[col].replace('NSA', '00').astype(float)

    # Convert LATITUDE and LONGITUDE to float
    df2[['LATITUDE', 'LONGITUDE']] = df2[['LATITUDE', 'LONGITUDE']].astype(float)

    df2.drop(columns=['TIME','DATE','DAY','ACCNUM','AVG_AGE'], inplace=True)

    # Handle class imbalance
    if class_imb == 'oversampling':
        ros = RandomOverSampler(random_state=17)
        X_res, y_res = ros.fit_resample(df2.drop(columns=['ACCLASS']), df2['ACCLASS'])
        df2 = pd.concat([X_res, y_res], axis=1).sample(frac=1, random_state=17).reset_index(drop=True)
    elif class_imb == 'undersampling':
        rus = RandomUnderSampler(random_state=17)
        X_res, y_res = rus.fit_resample(df2.drop(columns=['ACCLASS']), df2['ACCLASS'])
        df2 = pd.concat([X_res, y_res], axis=1).sample(frac=1, random_state=17).reset_index(drop=True)

    print("\n===================== DATA CLEANING DONE =====================")
    print("\nShape of the DataFrame after cleaning:", df2.shape)
    print("Class Distribution:\n", df2['ACCLASS'].value_counts())

    return df2

# ===================== AGGREGATE ROWS =====================
def aggregate_rows(group):
    # Find the row with the maximum number of non-null values in the non-'Fatal' group
    max_non_null_row_idx = group.notnull().sum(axis=1).idxmax()
    max_non_null_row = group.loc[max_non_null_row_idx].copy()  
    
    # Apply aggregation based on the column type (mean for numerical, mode for categorical)
    for col in max_non_null_row.index:
        if pd.api.types.is_numeric_dtype(group[col]):
            # For numerical columns, apply the mean
            mean_value = group[col].mean()
            max_non_null_row[col] = mean_value
        else:
            # For categorical columns, apply the mode
            mode_value = group[col].mode().iloc[0] if not group[col].mode().empty else None
            max_non_null_row[col] = mode_value
    
    # Return the processed non-Fatal row
    return max_non_null_row.to_frame().T

# ===================== DATA SAMPLING =====================
def sample_and_update_data(cleaned_df):
    """Splits the dataset into training and unseen data."""
    features = cleaned_df.drop(columns=["ACCLASS"])
    target = cleaned_df["ACCLASS"]

    unseen_features = features[-10:]
    unseen_labels = target[-10:]

    features = features[:-10]
    target = target[:-10]

    cleaned_df = cleaned_df.drop(cleaned_df.index[-10:])

    # Encode the target variable
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)

    # Encode unseen labels
    unseen_labels = label_encoder.transform(unseen_labels)

    return unseen_features, unseen_labels, cleaned_df, features, target

def data_preprocessing(features):
    """Prepares the data for SVM by applying preprocessing and optional SMOTE."""
    num_features = features.select_dtypes(include=['int', 'float']).columns.tolist()
    cat_features = features.select_dtypes(include=['object']).columns.tolist()

    print("\n===================== FEATURES INFO =====================")
    print("\nNumerical Features:", num_features)
    print("\nCategorical Features:", cat_features)

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='mode')),
        ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    return preprocessor

# ===================== TRAIN AND EVALUATE MODELS =====================
def process_and_train(data, columns_to_drop, class_imb, results):
    print(f"\n===================== {class_imb.upper()} =====================")

    # Clean the data
    cleaned_df = data_cleaning(data, columns_to_drop, class_imb=class_imb if class_imb != "original" else None)

    # Split the data into features and target
    unseen_features, unseen_labels, cleaned_df, features, target = sample_and_update_data(cleaned_df)

    print(features.columns)
    print(features.info())
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)

    # Split the data into train & test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, stratify=target, test_size=0.2, random_state=17
    )

    # Preprocess the data
    preprocessor = data_preprocessing(features)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    unseen_features = preprocessor.transform(unseen_features)

    return preprocessor


def start():
    # ===================== MAIN EXECUTION =====================
    # Initialize results list
    results = []

    # Process and train for each class imbalance method
    p = process_and_train(data_ksi, columns_to_drop, class_imb='undersampling', results=results)

    return p


start