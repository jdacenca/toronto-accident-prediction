import numpy as np
import pandas as pd

# Setting a random seed at beginning for consistency
np.random.seed(17)

try:
    # Loading & Checking the data
    data_ksi = pd.read_csv('./Total_KSI.csv')
    
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
                           'FATAL_NO', 'DISTRICT', 'DIVISION']
        data_ksi.drop(columns=columns_to_drop, inplace=True)
        
        # Separate features & target
        target = data_ksi["ACCLASS"]
        features = data_ksi.drop(columns=["ACCLASS"])
        
        print("\n===================== UNIQUE VALUES =====================")
        print("\nUnique values in ROAD_CLASS:", features["ROAD_CLASS"].unique())
        print("\nUnique values in ACCLASS:", target.unique())
        
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

    visualizer.scatter_plot("scatter_plot.png")
    visualizer.pie_chart("pie_chart.png")
    visualizer.bar_chart("bar_chart.png")
    visualizer.heatmap_correlation("heatmap_correlation.png")
    visualizer.heatmap_missing_values("heatmap_missing_values.png")
    visualizer.hist_plot("hist_plot.png")
    visualizer.pair_plot("pair_plot.png")
    visualizer.spline_plot("spline_plot.png")

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

    pipe_svm_ksi = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(random_state=17))
    ])

    param_grid_svm = [
        {'svm__kernel': ['linear'], 'svm__C': [1]}, # linear kernel    
        {'svm__kernel': ['rbf'], 'svm__C': [100], 'svm__gamma': [0.03]},  # rbf kernel
        {'svm__kernel': ['poly'], 'svm__C': [0.1], 'svm__gamma': [3.0], 'svm__degree': [3]}  # poly kernel 
    ]
    # param_grid_svm = [
    #     {'svm__kernel': ['linear'], 'svm__C': [0.01, 0.1, 1, 10, 100]},  # linear kernel    
    #     {'svm__kernel': ['rbf'], 'svm__C': [0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},  # rbf kernel
    #     {'svm__kernel': ['poly'], 'svm__C': [0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0], 'svm__degree': [2, 3]}  # poly kernel 
    # ]

    grid_search_ksi = GridSearchCV(estimator=pipe_svm_ksi, param_grid=param_grid_svm, scoring='accuracy', refit=True, verbose=3)

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
    joblib.dump(best_model, "best_svm_model.pkl")
    joblib.dump(pipe_svm_ksi, "pipe_svm_ksi.pkl")
    print("\n===================== BEST MODEL SAVED =====================")

except Exception as e:
    print("\n===================== ERROR =====================")
    print(f"An error occurred in training the model: {e}")
 
