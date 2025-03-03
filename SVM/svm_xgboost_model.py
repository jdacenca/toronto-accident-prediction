import numpy as np
import pandas as pd

# Setting a random seed at beginning for consistency
np.random.seed(17)

try:
    # Loading & Checking the data
    data_ksi = pd.read_csv('./data/Total_KSI.csv')
    
    if data_ksi is not None:
        pd.set_option("display.max_columns", 100)
        
        print("\n===================== DATA OVERVIEW =====================")
        print("\nDisplaying First 3 Records:\n", data_ksi.head(3))
        print("\nShape of the dataframe:", data_ksi.shape)
        print("\nData Type of the dataframe:", type(data_ksi))
        
        print("\n===================== DATA DESCRIPTION =====================")
        print("\nStatistical Summary:\n", data_ksi.describe())
        
        print("\n===================== COLUMN INFORMATION =====================")
        data_ksi.info()
        
        print("\n===================== MISSING VALUES =====================")
        # Calculate and display missing data
        missing_data = data_ksi.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing_data/len(data_ksi)) * 100
        print(pd.concat([missing_data, missing_percent], axis=1, keys=['Total Missing', 'Percent Missing']))

        # Drop unnecessary columns
        columns_to_drop = ['INDEX', 'ACCNUM', 'OBJECTID', 'HOOD_158', 'NEIGHBOURHOOD_158', 
                           'HOOD_140', 'NEIGHBOURHOOD_140', 'STREET1', 'STREET2', 'OFFSET', 
                           'FATAL_NO', 'DISTRICT', 'DIVISION','x','y','INJURY','INVTYPE','INVAGE','INITDIR','VEHTYPE',
                           'MANOEUVER','DRIVACT','PEDTYPE','PEDACT','CYCLISTYPE','CYCACT',]
        
        print("\n===================== UNIQUE VALUES =====================")
        print("\nUnique values in DATE:", data_ksi["DATE"].unique())
        print("\nUnique values in TIME:", data_ksi["TIME"].unique())
        print("\nUnique values in ROAD_CLASS:", data_ksi["ROAD_CLASS"].unique())
        print("\nUnique values in ACCLOC:",data_ksi["ACCLOC"].unique())
        print("\nUnique values in TRAFFCTL:", data_ksi["TRAFFCTL"].unique())
        print("\nUnique values in VISIBILITY:", data_ksi["VISIBILITY"].unique())
        print("\nUnique values in LIGHT:", data_ksi["LIGHT"].unique())
        print("\nUnique values in RDSFCOND:", data_ksi["RDSFCOND"].unique())
        print("\nUnique values in IMPACTYPE:", data_ksi["IMPACTYPE"].unique())
        print("\nUnique values in INVTYPE:", data_ksi["INVTYPE"].unique())
        print("\nUnique values in INVAGE:", data_ksi["INVAGE"].unique())
        print("\nUnique values in INJURY:", data_ksi["INJURY"].unique())
        print("\nUnique values in INITDIR:", data_ksi["INITDIR"].unique())
        print("\nUnique values in VEHTYPE:", data_ksi["VEHTYPE"].unique())
        print("\nUnique values in MANOEUVER:", data_ksi["MANOEUVER"].unique())
        print("\nUnique values in DRIVACT:", data_ksi["DRIVACT"].unique())
        print("\nUnique values in DRIVCOND:", data_ksi["DRIVCOND"].unique())
        print("\nUnique values in PEDTYPE:", data_ksi["PEDTYPE"].unique())
        print("\nUnique values in PEDACT:", data_ksi["PEDACT"].unique())
        print("\nUnique values in PEDCOND:", data_ksi["PEDCOND"].unique())
        print("\nUnique values in CYCLISTYPE:", data_ksi["CYCLISTYPE"].unique())
        print("\nUnique values in CYCACT:", data_ksi["CYCACT"].unique())
        print("\nUnique values in CYCCOND:", data_ksi["CYCCOND"].unique())
        print("\nUnique values in PEDESTRIAN:", data_ksi["PEDESTRIAN"].unique())
        print("\nUnique values in CYCLIST:", data_ksi["CYCLIST"].unique())
        print("\nUnique values in AUTOMOBILE:",data_ksi["AUTOMOBILE"].unique())
        print("\nUnique values in MOTORCYCLE:", data_ksi["MOTORCYCLE"].unique())
        print("\nUnique values in TRUCK:", data_ksi["TRUCK"].unique())
        print("\nUnique values in TRSN_CITY_VEH:", data_ksi["TRSN_CITY_VEH"].unique())
        print("\nUnique values in EMERG_VEH:", data_ksi["EMERG_VEH"].unique())
        print("\nUnique values in PASSENGER:", data_ksi["PASSENGER"].unique())
        print("\nUnique values in SPEEDING:", data_ksi["SPEEDING"].unique())
        print("\nUnique values in AG_DRIV:", data_ksi["AG_DRIV"].unique())
        print("\nUnique values in REDLIGHT:", data_ksi["REDLIGHT"].unique())
        print("\nUnique values in ALCOHOL:", data_ksi["ALCOHOL"].unique())
        print("\nUnique values in DISABILITY:", data_ksi["DISABILITY"].unique())
        print("\nUnique values in ACCLASS:", data_ksi["ACCLASS"].unique())

        # Drop unnecessary columns
        data_ksi.drop(columns=columns_to_drop, inplace=True)

        data_ksi = data_ksi.dropna(subset=["ACCLASS"])

        data_ksi["PEDESTRIAN"] = data_ksi["PEDESTRIAN"].fillna("No")
        data_ksi["CYCLIST"] = data_ksi["CYCLIST"].fillna("No")
        data_ksi["AUTOMOBILE"] = data_ksi["AUTOMOBILE"].fillna("No")
        data_ksi["MOTORCYCLE"] = data_ksi["MOTORCYCLE"].fillna("No")
        data_ksi["TRUCK"] = data_ksi["TRUCK"].fillna("No")
        data_ksi["TRSN_CITY_VEH"] = data_ksi["TRSN_CITY_VEH"].fillna("No")
        data_ksi["EMERG_VEH"] = data_ksi["EMERG_VEH"].fillna("No")
        data_ksi["PASSENGER"] = data_ksi["PASSENGER"].fillna("No")
        data_ksi["SPEEDING"] = data_ksi["SPEEDING"].fillna("No")
        data_ksi["AG_DRIV"] = data_ksi["AG_DRIV"].fillna("No")
        data_ksi["REDLIGHT"] = data_ksi["REDLIGHT"].fillna("No")
        data_ksi["ALCOHOL"] = data_ksi["ALCOHOL"].fillna("No")
        data_ksi["DISABILITY"] = data_ksi["DISABILITY"].fillna("No")
        data_ksi["ACCLASS"] = data_ksi["ACCLASS"].replace("Property Damage O", "Non-Fatal Injury")

        # Separate features & target
        target = data_ksi["ACCLASS"]
        features = data_ksi.drop(columns=["ACCLASS"])

        # Isolate last 10 records for unseen data
        unseen_features = features.iloc[-10:]
        unseen_labels = target.iloc[-10:]
        
        # Remove last 10 records from main dataset
        features = features.iloc[:-10]
        target = target.iloc[:-10]
        
        print("\n===================== FEATURES INFO =====================")
        print("\nFeatures Info:\n")
        features.info()
        
except Exception as e:
    print("\n===================== ERROR =====================")
    print(f"An error occurred in loading & checking the data: {e}")

# Step b: Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Visualizer import Visualizer

try:
    visualizer = Visualizer(data_ksi)

    visualizer.scatter_plot("./images/scatter_plot.png")
    visualizer.pie_chart("./images/pie_chart.png")
    visualizer.bar_chart("./images/bar_chart.png")
    visualizer.heatmap_correlation("./images/heatmap_correlation.png")
    visualizer.heatmap_missing_values("./images/heatmap_missing_values.png")
    visualizer.hist_plot("./images/hist_plot.png")
    visualizer.pair_plot("./images/pair_plot.png")
    visualizer.spline_plot("./images/spline_plot.png")

    # Encode the target variable (ACCLASS) using LabelEncoder
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)

    # Split the data into train & test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=17)

    num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = features.select_dtypes(include=['object']).columns.tolist()

except Exception as e:
    print("\n===================== ERROR =====================")
    print(f"An error occurred in visualization: {e}")

# Step c: Pre-process the model
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

try:
    # Identify numerical and categorical features
    num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = features.select_dtypes(include=['object']).columns.tolist()

    print("\n===================== NUMERICAL & CATEGORIAL FEATURES =====================")
    print("\nNumerical features:", num_features)
    print("\nCategorical features:", cat_features)

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())  
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('encoder', OneHotEncoder(handle_unknown="ignore"))  
    ])

    # Combine transformations using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    model_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(random_state=17))
    ])

    preprocessor1 = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder='passthrough')

    model_xgb = Pipeline([
        ('preprocessor', preprocessor1),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=17))
    ])

    param_grid_combined = {
    'svm__svm__kernel': ['rbf', 'poly'],
    'svm__svm__C': [0.1,1,100],
    'svm__svm__gamma': [0.03,3.0],
    'svm__svm__degree': [3],
    'xg_boost__xgb__learning_rate': [0.2,0.25],
    'xg_boost__xgb__max_depth': [7,8],
    'xg_boost__xgb__n_estimators': [200,250],
    'xg_boost__xgb__subsample': [0.7,0.75,0.8],
    'xg_boost__xgb__colsample_bytree': [0.7,0.75,0.8]
    }

    # Voting Classifier (Hard Voting)
    voting_clf = VotingClassifier(estimators=[('svm', model_svm), ('xg_boost', model_xgb)], voting='hard')

    # Step c: Train the model with GridSearchCV
    grid_search_ksi = GridSearchCV(estimator=voting_clf, param_grid=param_grid_combined, scoring='accuracy', refit=True, verbose=3)


except Exception as e:
    print("\n===================== ERROR =====================")
    print(f"An error occurred in pre-processing the model: {e}")
 

# Step d: Train the  model
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from ModelPerformance import ModelPerformance

try:
    print ("\n===================== GRID SEARCH FIT =====================")
    grid_search_ksi.fit(X_train, y_train)

    best_model = grid_search_ksi.best_estimator_

    y_pred = best_model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    modelPerformance = ModelPerformance(best_model, X_test, y_test)
    modelPerformance.conf_matrix("confusion_matrix.png")

    print ("\n===================== CONFUSION MATRIX =====================")
    print("\nConfusion Matrix:\n", cm)

    # Print classification report for detailed performance metrics
    print("\n===================== CLASSIFICATION REPORT =====================")
    print("\nClassification Report:\n", classification_report(y_test, y_pred,zero_division=1))

    # Save the best model
    print("\n===================== BEST MODEL METRICS =====================")
    print("\nBest Parameters:", grid_search_ksi.best_params_)
    print("Best Estimator:", grid_search_ksi.best_estimator_)
    print("Best Training Accuracy:", grid_search_ksi.best_score_)

    accuracy = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    label_map = {'Fatal': 0, 'Non-Fatal Injury': 1, 'Property Damage O': 2}
    unseen_labels_numeric = unseen_labels.map(label_map)

    # Then fit the model and predict as usual
    best_model.fit(unseen_features, unseen_labels_numeric)
    unseen_predictions = best_model.predict(unseen_features)
    unseen_accuracy = best_model.score(unseen_features, unseen_labels_numeric)

    print("\n===================== UNSEEN DATA METRICS =====================")
    print("\nUnseen Predictions:", unseen_predictions)
    for i in range(len(unseen_features)):
        print(f"Predicted: {unseen_predictions[i]} Actual: {unseen_labels_numeric.iloc[i]}")

    print(f"Unseen Data Accuracy: {unseen_accuracy:.4f}")

    # Save the model using joblib
    joblib.dump(best_model, "best_svm_xg_boost_model.pkl")
    print("\n===================== BEST MODEL SAVED =====================")

except Exception as e:
    print("\n===================== ERROR =====================")
    print(f"An error occurred in training the model: {e}")
 
