import joblib
import pandas as pd
from pathlib import Path

# %%
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
df_ksi = load_data()

pipeline = joblib.load(r'.\pickles\log_reg_preprocessing_pipeline.pkl')
lr_model = joblib.load(r'.\pickles\logistic_regression_model.pkl')

# %%
testdict = {'ROAD_CLASS': {0: 'LOCAL'}, 'DISTRICT': {0: 'NORTH YORK'}, 'LATITUDE': {0: 43.684166917643296}, 'LONGITUDE': {0: -79.43998504705428}, 'ACCLOC': {0: 'AT INTERSECTION'}, 'TRAFFCTL': {0: 'NO CONTROL'}, 'VISIBILITY': {0: 'CLEAR'}, 'LIGHT': {0: 'DARK'}, 'RDSFCOND': {0: 'DRY'}, 'ACCLASS': {0: 'Fatal'}, 'IMPACTYPE': {0: 'APPROACHING'}, 'INVTYPE': {0: 'DRIVER'}, 'INVAGE': {0: '25 TO 29'}, 'PEDCOND': {0: 'NORMAL'}, 'CYCCOND': {0: 'NORMAL'}, 'PEDESTRIAN': {0: 'NO'}, 'CYCLIST': {0: 'NO'}, 'AUTOMOBILE': {0: 'NO'}, 'MOTORCYCLE': {0: 'NO'}, 'TRUCK': {0: 'NO'}, 'TRSN_CITY_VEH': {0: 'NO'}, 'EMERG_VEH': {0: 'NO'}, 'PASSENGER': {0: 'NO'}, 'SPEEDING': {0: 'NO'}, 'AG_DRIV': {0: 'NO'}, 'REDLIGHT': {0: 'NO'}, 'ALCOHOL': {0: 'NO'}, 'DISABILITY': {0: 'NO'}, 'NEIGHBOURHOOD_158': {0: 'WOODBINE-LUMSDEN'}}

testpkl1 = pd.DataFrame.from_dict(testdict)

# df_test = pd.DataFrame()
pd.set_option('display.max_columns', None)
df_transformed = pipeline.transform(testpkl1)
print(df_transformed)

output = lr_model.predict(df_transformed.drop('ACCLASS', axis=1))

print(output)
# lr_model.predict()