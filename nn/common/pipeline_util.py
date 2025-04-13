import pandas as pd
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from transform import apply_binary_mapping, apply_target_mapping, apply_label_encoding

# Define the mappings and column lists
binary_mapping = {'YES': 1, 'NO': 0}
target_mapping = {'FATAL': 1, 'NON-FATAL INJURY': 0, 'PROPERTY DAMAGE O': 0}
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
    'ROAD_CLASS',
    'DISTRICT',
    'TRAFFCTL',
    'VISIBILITY',
    'LIGHT',
    'RDSFCOND',
    'IMPACTYPE',
    'INVAGE',
    'PEDCOND',
    'CYCCOND',
    'NEIGHBOURHOOD_158'
]

def create_pickle(model, desc):

    # Create the pipeline
    preprocessor = Pipeline(steps=[
        ('binary_mapping', FunctionTransformer(apply_binary_mapping, kw_args={'columns': column_binary, 'mapping': binary_mapping})),
        ('label_encoding', FunctionTransformer(apply_label_encoding, kw_args={'columns': column_le}))
    ])

    # Create the full pipeline with preprocessing and the model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Use the preprocessor pipeline as a step
        ('classifier', model)  # Add your model as the next step
    ])

    with open(f'./output/mlp_model_preprocessing_{desc}.pkl', 'wb') as file:
        joblib.dump(preprocessor, file)

    with open(f'./output/mlp_single_model_{desc}.pkl', 'wb') as file:
        joblib.dump(model, file)

    print("Model saved as mlp_model.pkl")

    loaded_pipeline = joblib.load(f'./output/mlp_model_{desc}.pkl')
    print("Model loaded successfully in the saving script!")
    print(loaded_pipeline)
