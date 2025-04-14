import json
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask_cors import CORS
import utils.data_util as data_util
from typing import List, TypedDict
from load_models import load_model
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from config import hood_id_158_upp, hood_id_140_upp, hood_158_vs_140_upp

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Input data path
input_file_path = './data/TOTAL_KSI_6386614326836635957.csv'

# Load necessary data files
data = data_util.load_data(input_file_path)
clean_data = data_util.clean_data(data)

# Load each model
dt_model = load_model("./pickle_files/decision_tree_model.pkl")
dt_preprocessing_model = load_model("./pickle_files/preprocessing_pipeline.pkl")
rf_model = load_model("./pickle_files/random_forest_model.pkl")
rf_preprocessing_model = load_model("./pickle_files/preprocessing_pipeline_rf.pkl")
nn_model = load_model("./pickle_files/mlp_model_Smote+Tomek.pkl")    #./pickle_files/mlp_model_Random Sampling.pkl
svm_model = load_model("./pickle_files/best_svm_rbf_oversampling.pkl")

ensemble_preprocessor = load_model("./pickle_files/ensemble/preprocessor.pkl")
hardvoting_model = load_model("./pickle_files/ensemble/Hard_Voting_Classifier.pkl")
softvoting_model = load_model("./pickle_files/ensemble/Soft_Voting_Classifier.pkl")

@app.route('/predict/<model>', methods=['POST'])
def get_prediction(model):
    """
    Endpoint to get the number of accidents per month.
    """
    if request.method == 'OPTIONS':
        # Flask-CORS should handle this automatically, but sometimes explicit handling helps
        return '', 204, {'Access-Control-Allow-Origin': 'http://localhost:5173',
                         'Access-Control-Allow-Methods': 'POST, OPTIONS',
                         'Access-Control-Allow-Headers': 'Content-Type'}
    elif request.method == 'POST':
        if model == 'nn':
            #print(request.data)
            string_data = request.data.decode('utf-8')

            # Parse the JSON string into a Python dictionary
            data_dict = json.loads(string_data) 
            
            new_data = pd.DataFrame([data_dict])
            print(new_data)
            new_data = new_data.drop(columns=['ACCLASS', 'ACCLOC', 'INVTYPE', 'HOUR', 'DAYOFWEEK', 'WEEK'], axis=0)
            p = nn_model.predict(new_data)
            print(f"PREDICITON: {p}")
        elif model == 'dt':
            
            string_data = request.data.decode('utf-8')

            # Parse the JSON string into a Python dictionary
            data_dict = json.loads(string_data) 

            new_data = pd.DataFrame([data_dict])
            print(new_data)

            data = dt_preprocessing_model.transform(new_data)
            X = data.drop(columns=['ACCLASS'])
            p = dt_model.predict(X)
            print(p)
        elif model == 'svm':
            string_data = request.data.decode('utf-8')
            data_dict = json.loads(string_data) 

            boolean_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH',
                        'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY','EMERG_VEH']
            
            new_data = pd.DataFrame([data_dict])
            print(new_data)
            new_data[boolean_columns] = new_data[boolean_columns].replace({'YES': 1, 'NO': 0}).astype(float)
            new_data['INVAGE'] = new_data['INVAGE'].str.replace('OVER 95', '95 TO 100')
            new_data['INVAGE'] = new_data['INVAGE'].replace('UNKNOWN', np.nan)
            new_data[['min_age', 'max_age']] = new_data['INVAGE'].str.split(' TO ', expand=True)
            new_data['min_age'] = pd.to_numeric(new_data['min_age'], errors='coerce')
            new_data['max_age'] = pd.to_numeric(new_data['max_age'], errors='coerce')
            new_data['AVG_AGE'] = new_data[['min_age', 'max_age']].mean(axis=1).astype(float)
            new_data.drop(columns=['INVAGE','min_age', 'max_age'], inplace=True)
            new_data['INVAGE'] = new_data['AVG_AGE'].fillna(new_data['AVG_AGE'].mean()).astype(float)

            if new_data['MONTH'].isin([12, 1, 2]).any():
                new_data['SEASON'] = 0  # Winter
            elif new_data['MONTH'].isin([3, 4, 5]).any():
                new_data['SEASON'] =  1  # Spring
            elif new_data['MONTH'].isin([6, 7, 8]).any():
                new_data['SEASON'] =  2  # Summer
            else:
                new_data['SEASON'] =  3  # Fall
            
            print(new_data)

            p = svm_model.predict(new_data)
            print(p)
        elif model == 'rf':
            print("RF")
            string_data = request.data.decode('utf-8')

            # Parse the JSON string into a Python dictionary
            data_dict = json.loads(string_data) 

            new_data = pd.DataFrame([data_dict])
            print(new_data)

            data = rf_preprocessing_model.transform(new_data)
            #X = data.drop(columns=['ACCLASS'])
            p = rf_model.predict(data)
            print(p)
        elif model == 'sv':
            string_data = request.data.decode('utf-8')
            data_dict = json.loads(string_data) 

            boolean_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH',
                        'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
            
            new_data = pd.DataFrame([data_dict])

            new_data[boolean_columns] = new_data[boolean_columns].replace({'YES': 1, 'NO': 0}).astype(float)
            new_data['INVAGE'] = new_data['INVAGE'].str.replace('OVER 95', '95 TO 100')
            new_data['INVAGE'] = new_data['INVAGE'].replace('UNKNOWN', np.nan)
            new_data[['min_age', 'max_age']] = new_data['INVAGE'].str.split(' TO ', expand=True)
            new_data['min_age'] = pd.to_numeric(new_data['min_age'], errors='coerce')
            new_data['max_age'] = pd.to_numeric(new_data['max_age'], errors='coerce')
            new_data['AVG_AGE'] = new_data[['min_age', 'max_age']].mean(axis=1).astype(float)
            new_data.drop(columns=['INVAGE','min_age', 'max_age'], inplace=True)
            new_data['INVAGE'] = new_data['AVG_AGE'].fillna(new_data['AVG_AGE'].mean()).astype(float)

            if new_data['MONTH'].isin([12, 1, 2]).any():
                new_data['SEASON'] = 0  # Winter
            elif new_data['MONTH'].isin([3, 4, 5]).any():
                new_data['SEASON'] =  1  # Spring
            elif new_data['MONTH'].isin([6, 7, 8]).any():
                new_data['SEASON'] =  2  # Summer
            else:
                new_data['SEASON'] =  3  # Fall

            val = new_data['NEIGHBOURHOOD_158'].to_string(index=False)

            new_data['HOOD_158'] = hood_id_158_upp[val]
            hood_140 = hood_158_vs_140_upp[val]
            print(f"HOOD_140: {hood_140}")
            new_data['HOOD_140'] = hood_id_140_upp[hood_140]

            new_data['DIVISION'] = new_data['DIVISION'].replace('NSA', '00').str[1:].astype(float)
            new_data.drop(columns=['NEIGHBOURHOOD_158'], inplace=True)

            processed_data = ensemble_preprocessor.transform(new_data)
            print(processed_data)

            p = softvoting_model.predict(processed_data)
            print(p)

        elif model == 'hv':
            string_data = request.data.decode('utf-8')
            data_dict = json.loads(string_data) 

            boolean_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH',
                        'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
            
            new_data = pd.DataFrame([data_dict])

            new_data[boolean_columns] = new_data[boolean_columns].replace({'YES': 1, 'NO': 0}).astype(float)
            new_data['INVAGE'] = new_data['INVAGE'].str.replace('OVER 95', '95 TO 100')
            new_data['INVAGE'] = new_data['INVAGE'].replace('UNKNOWN', np.nan)
            new_data[['min_age', 'max_age']] = new_data['INVAGE'].str.split(' TO ', expand=True)
            new_data['min_age'] = pd.to_numeric(new_data['min_age'], errors='coerce')
            new_data['max_age'] = pd.to_numeric(new_data['max_age'], errors='coerce')
            new_data['AVG_AGE'] = new_data[['min_age', 'max_age']].mean(axis=1).astype(float)
            new_data.drop(columns=['INVAGE','min_age', 'max_age'], inplace=True)
            new_data['INVAGE'] = new_data['AVG_AGE'].fillna(new_data['AVG_AGE'].mean()).astype(float)

            if new_data['MONTH'].isin([12, 1, 2]).any():
                new_data['SEASON'] = 0  # Winter
            elif new_data['MONTH'].isin([3, 4, 5]).any():
                new_data['SEASON'] =  1  # Spring
            elif new_data['MONTH'].isin([6, 7, 8]).any():
                new_data['SEASON'] =  2  # Summer
            else:
                new_data['SEASON'] =  3  # Fall

            val = new_data['NEIGHBOURHOOD_158'].to_string(index=False)

            new_data['HOOD_158'] = hood_id_158_upp[val]
            hood_140 = hood_158_vs_140_upp[val]
            print(f"HOOD_140: {hood_140}")
            new_data['HOOD_140'] = hood_id_140_upp[hood_140]

            new_data['DIVISION'] = new_data['DIVISION'].replace('NSA', '00').str[1:].astype(float)
            new_data.drop(columns=['NEIGHBOURHOOD_158'], inplace=True)

            processed_data = ensemble_preprocessor.transform(new_data)
            print(processed_data)

            p = hardvoting_model.predict(processed_data)
            print(p)
        
        
        if p[0] == 1:
            json_data = json.dumps({"prediction": "FATAL"})
        else:
            json_data = json.dumps({"prediction": "NON-FATAL"})

        print(json_data)
        return json_data, 200

# ===================== DATA PREPROCESSING =====================
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
#=============================================================================
@app.route('/data/year_list', methods=['GET'])
def get_unique_year():
    """
    Endpoint to get the number of accidents per month.
    """
    # Group by month and count the number of accidents
    year = clean_data['YEAR'].unique().tolist()

    # Convert to JSON format
    year_json = json.dumps(year)

    return year_json, 200

@app.route('/data/accident/total/<year>', methods=['GET'])
def get_total_yearly_accidents(year):
    """
    Endpoint to get the total number of accidents per year.
    """
    idx = pd.IndexSlice

    # Group by year and count the number of accidents
    yearly_accidents = clean_data.groupby(['MONTH', 'YEAR'])['ACCNUM'].agg(accidents='nunique', count='size')
    y = yearly_accidents.loc[idx[:, int(year)], 'accidents']
    print(y)
    specific_year_accidents = y.to_list()
    
    year_accidents_value = json.dumps({"per_month": specific_year_accidents, "total": int(y.sum())})

    return year_accidents_value, 200

@app.route('/data/accident/total/<year>/<injury>', methods=['GET'])
def get_total_yearly_accidents_by_injury(year, injury):
    """
    Endpoint to get the number of accidents per yaer and by injury.
    """
    idx = pd.IndexSlice

    print(injury)
    injury = injury.upper()

    # Group by year and count the number of accidents
    yearly_accidents_w_injury = clean_data.groupby(['MONTH', 'YEAR', 'ACCLASS'])['ACCNUM'].agg(accidents='nunique', count='size')
    y = yearly_accidents_w_injury.loc[idx[:, int(year), injury], 'accidents']
    specific_year_accidents = y.to_list()
    
    year_accidents_value = json.dumps({"per_month": specific_year_accidents, "total": int(y.sum())})

    return year_accidents_value, 200

@app.route('/data/latlong/<year>/<injury>', methods=['GET'])
def get_latlong_accidents(year, injury):
    """
    Endpoint to get the number of accidents per month.
    """
    idx = pd.IndexSlice

    print(injury)
    injury = injury.upper()

    # Group by accnum and latitude and longitude
    latlong_grp = clean_data.groupby(['YEAR', 'ACCLASS', 'LATITUDE', 'LONGITUDE'])['ACCNUM'].agg(accidents='nunique')
    y = latlong_grp.loc[idx[int(year), injury], 'accidents']
    latlong_accidents = pd.DataFrame()
    latlong_accidents["LAT"] = y.index.get_level_values('LATITUDE')
    latlong_accidents["LONG"] = y.index.get_level_values('LONGITUDE')
    latlong_accidents.reset_index(drop=True, inplace=True)
    latlong_accidents["id"] = latlong_accidents.index
    
    # Convert to JSON format
    latlong_accidents_json = latlong_accidents.to_json()

    return latlong_accidents_json, 200

@app.route('/data/base/<year>', methods=['GET'])
def get_base_accidents(year):
    """
    Endpoint to get the number of accidents per month.
    """
    idx = pd.IndexSlice

    # Group by accnum and latitude and longitude
    base = data.groupby(['YEAR', 'ACCNUM', 'ACCLASS', 'NEIGHBOURHOOD_158', 'VISIBILITY', 'LIGHT', 'RDSFCOND'])['ACCNUM'].agg(accidents='nunique')
    y = base.loc[idx[int(year)], 'accidents'].to_frame().reset_index()

    base_accidents = {}
    base_accidents["Id"] = y['ACCNUM']
    base_accidents["Injury"] = y['ACCLASS']
    base_accidents["Neighborhood"] = y['NEIGHBOURHOOD_158']
    base_accidents["Visibility"] = y['VISIBILITY']
    base_accidents["Light"] = y['LIGHT']
    base_accidents["RDSFCOND"] = y['RDSFCOND']
    
    formed_base_accidents = []

    for index, row in y.iterrows():
        accident_dict = {
            'id': index+1,
            'Accnum': row['ACCNUM'],
            'Injury': row['ACCLASS'],
            'Neighborhood': row['NEIGHBOURHOOD_158'],
            'Visibility': row['VISIBILITY'],
            'Light': row['LIGHT'],
            'RDSFCOND': row['RDSFCOND']
        }
        formed_base_accidents.append(accident_dict)

    # Convert to JSON format
    base_accidents_json = json.dumps(formed_base_accidents)

    return base_accidents_json, 200

#=======================================================================================================
# Data Exploration APIs
#=======================================================================================================

@app.route('/data/distribution/<col>', methods=['GET'])
def get_field_distribution(col):
    """
    Endpoint to get the distribution for the specific field.
    """

    print(col)
    col = col.upper()

    fields_to_drop = [
        'INDEX', 'ACCNUM', 'DATE', 'TIME',
        'LATITUDE', 'LONGITUDE', 'FATAL_NO',
        "STREET1",
        "STREET2",
        "NEIGHBOURHOOD_158",
        "NEIGHBOURHOOD_140",
        "DIVISION",
        'HOOD_158', 'HOOD_140', 'x', 'y',
    ]

    data_copy = clean_data.copy(deep=True)
    data_copy.drop(columns=fields_to_drop, inplace=True)


    count = data_copy.groupby("ACCLASS")[col].value_counts()

    fatal = count.loc["FATAL"]
    non = count.loc["NON-FATAL INJURY"]
    property = count.loc["PROPERTY DAMAGE O"]

    combined = pd.concat([fatal, non, property], axis=1)
    combined.columns = ["FATAL", "NON-FATAL INJURY", "PROPERTY DAMAGE O"]
    combined = combined.fillna(0)

    #combined = combined.T
    count_json = combined.to_dict()
    #print(count_json)

    transformed_data = {}
    for key, inner_dict in count_json.items():
        transformed_data[key] = list(inner_dict.values())

    print(transformed_data)

    # Convert the transformed data to JSON format
    json_data = {"data": transformed_data, "labels": combined.index.tolist()}
    return json_data, 200

@app.route('/data/missing', methods=['GET'])
def get_field_missing_distribution():
    """
    Endpoint to get the missing for the specific field.
    """

    count = clean_data.isnull().sum()

    null_value_counts = count[count > 0]

    count_json = null_value_counts.to_json()
    print(count_json)

    return count_json, 200

class LocationData(TypedDict):
    key: str
    location: dict[str, float]

@app.route('/data/map/basic', methods=['GET'])
def get_map_basic():
    """
    Endpoint to get the missing for the specific field.
    """
    idx = pd.IndexSlice
    
    markers = clean_data.groupby(["LATITUDE", "LONGITUDE"])["ACCLASS"].value_counts()
    fatal = markers.loc[idx[:,:,"FATAL"]].to_frame()
    non = markers.loc[idx[:,:,"NON-FATAL INJURY"]].to_frame()
    property = markers.loc[idx[:,:,"PROPERTY DAMAGE O"]].to_frame()

    fatal = fatal.reset_index()
    non = non.reset_index()
    property = property.reset_index()

    # FATAL
    keys_array = list(range(len(fatal["LATITUDE"])))
    fatal_locations: List[LocationData] = [
        {'key': key, 'location': {'lat': fatal["LATITUDE"][index], 'lng': fatal["LONGITUDE"][index]}}
        for index, key in enumerate(keys_array)
    ]
    
    keys_array = list(range(len(non["LATITUDE"])))
    non_locations: List[LocationData] = [
        {'key': key, 'location': {'lat': non["LATITUDE"][index], 'lng': non["LONGITUDE"][index]}}
        for index, key in enumerate(keys_array)
    ]

    keys_array = list(range(len(property["LATITUDE"])))
    property_locations: List[LocationData] = [
        {'key': key, 'location': {'lat': property["LATITUDE"][index], 'lng': property["LONGITUDE"][index]}}
        for index, key in enumerate(keys_array)
    ]

    json_data = {"fatal": fatal_locations, "non": non_locations, "property": property_locations}
    markers = json.dumps(json_data)


    return markers, 200

@app.route('/data/distribution_list', methods=['GET'])
def get_field_distribution_list():
    """
    Endpoint to get the distribution for the specific field.
    """

    fields_to_drop = [
        'INDEX', 'ACCNUM', 'DATE', 'TIME',
        'LATITUDE', 'LONGITUDE', 'FATAL_NO',
        "STREET1", "ACCLASS",
        "STREET2",
        "NEIGHBOURHOOD_158",
        "NEIGHBOURHOOD_140",
        "DIVISION", "OFFSET",
        'HOOD_158', 'HOOD_140', 'x', 'y',
    ]

    data_copy = data.copy(deep=True)
    data_copy.drop(columns=fields_to_drop, inplace=True)

    columns = data_copy.columns.tolist()
    columns_json = json.dumps(columns)
    print(columns_json)
    return columns_json, 200



if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)