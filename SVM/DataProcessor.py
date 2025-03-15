  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from ModelPerformance import ModelPerformance
from imblearn.pipeline import Pipeline as imPipeline 
from imblearn.over_sampling import SMOTE

def data_overview(df):
        
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_rows", 100)
        
        print("\n===================== DATA OVERVIEW =====================")
        print("\nDisplaying First 3 Records:\n", df.head(3))
        print("\nShape of the dataframe:", df.shape)
        print("\nData Type of the dataframe:", type(df))
        
        print("\n===================== DATA DESCRIPTION =====================")
        print("\nStatistical Summary:\n", df.describe())
        
        print("\n===================== COLUMN INFORMATION =====================")
        df.info()
        
        print("\n===================== MISSING VALUES =====================")
        # calculate and display missing data
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing_data/len(df)) * 100
        print(pd.concat([missing_data, missing_percent], axis=1, keys=['Total Missing', 'Percent Missing']))

        print("\n===================== UNIQUE VALUES =====================")
        for column in df.columns:
            print(f"\nUnique values in {column}:", df[column].unique())

def data_cleaning(df, columns_to_drop):
    
    df2 = df.copy()

    # drop unnecessary columns
    df2.drop(columns=columns_to_drop, inplace=True)

    # drop rows with missing target values
    df2 = df2.dropna(subset=["ACCLASS"])

    # aggregate rows with same ACCNUM, DATE, TIME, LATITUDE, LONGITUDE
    df2 = df2.groupby(['ACCNUM'], as_index=False).apply(aggregate_rows).reset_index(drop=True)
    df2 = df2.groupby(['DATE', 'TIME', 'LATITUDE', 'LONGITUDE'], as_index=False).apply(aggregate_rows).reset_index(drop=True)

    df2.drop(columns=['ACCNUM'], inplace=True)

    # drop rows with property damage only
    df2.drop(df2[df2['ACCLASS'] == 'Property Damage O'].index, inplace = True)

    # drop duplicate rows
    df2.drop_duplicates(inplace=True)

    # date trunc remove time
    df2["DATE"] = pd.to_datetime(df2["DATE"])
    df2['DATE'] = df2["DATE"].dt.to_period("D").astype(str)
    
    # update time to per hour
    df2['TIME'] = pd.to_datetime(df2['TIME'], format='%H%M', errors='coerce').dt.hour

    df2['ROAD_CLASS'] = df2['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False) 

    # unknown_columns = [ 'PEDCOND', 'CYCCOND', 'DRIVCOND'] - dropping for now
    other_columns = ['ROAD_CLASS', 'ACCLOC', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE']
    boolean_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']

    df2[other_columns] = df2[other_columns].fillna("Other")
    #df2[unknown_columns] = df2[unknown_columns].fillna("Unknown")
    df2[boolean_columns] = df2[boolean_columns].fillna("No")

    df2["TRAFFCTL"] = df2["TRAFFCTL"].fillna("No_Control")


    # undersampling based on equal amount of acclass values
    # rus = RandomUnderSampler(sampling_strategy='auto', random_state=17)
    # X_res, y_res = rus.fit_resample(df2.drop(columns=['ACCLASS']), df2['ACCLASS'])

    # oversampling based on equal amount of acclass values
    rus = RandomOverSampler(sampling_strategy='auto', random_state=17)
    X_res, y_res = rus.fit_resample(df2.drop(columns=['ACCLASS']), df2['ACCLASS'])

    # combine resampled features and labels into a single DataFrame
    df2 = pd.concat([X_res, y_res], axis=1)
    df2 = df2.sample(frac=1, random_state=17).reset_index(drop=True)  # Shuffle the data


    print("\n===================== DATA CLEANING DONE =====================")
    print("\nShape of the dataframe after cleaning:", df2.shape)
    print("Count of each class:", df2['ACCLASS'].value_counts())
    
    return df2

def aggregate_rows(group):
    # If all rows are identical, keep the first row
    if group.nunique().eq(1).all():
        return group.iloc[0]
    else:
        # Otherwise, aggregate numerical columns with mean, categorical with mode
        aggregated = group.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean() if x.dtype != 'O' else None)
        return aggregated

def sample_and_update_data(cleaned_df, class_1_label='Fatal', class_2_label='Non-Fatal Injury', sample_size=5, random_state=17):
    # Separate data by class
    class_1_data = cleaned_df[cleaned_df['ACCLASS'] == class_1_label]
    class_2_data = cleaned_df[cleaned_df['ACCLASS'] == class_2_label]

    # Check if there are enough records in both classes
    if len(class_1_data) >= sample_size and len(class_2_data) >= sample_size:
        # Sample from both classes
        sample_class_1 = class_1_data.sample(n=sample_size, random_state=random_state)
        sample_class_2 = class_2_data.sample(n=sample_size, random_state=random_state)

        # Concatenate samples and shuffle
        final_sample = pd.concat([sample_class_1, sample_class_2]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Set final_sample to be the unseen features and labels
        unseen_features = final_sample.drop(columns=["ACCLASS"])
        unseen_labels = final_sample["ACCLASS"]

        # Remove sampled rows from the original dataset
        cleaned_df = cleaned_df.drop(final_sample.index)

        # Update features and target after removal
        features = cleaned_df.drop(columns=["ACCLASS"])
        target = cleaned_df["ACCLASS"]

        return unseen_features, unseen_labels, cleaned_df, features, target
    
    else:
        raise ValueError("Error: One or both classes have fewer than the required number of records.")

def data_preprocessing_svm(features):
    # Identifying numerical and categorical features
    num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = features.select_dtypes(include=['object']).columns.tolist()

    # Displaying feature info
    print("\n===================== FEATURES INFO =====================")
    print("\nFeatures Info:\n")
    features.info()

    print("\n===================== NUMERICAL & CATEGORICAL FEATURES =====================")
    print("\nNumerical features:", num_features)
    print("\nCategorical features:", cat_features)

    # Numerical transformation pipeline
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())  
    ])

    # Categorical transformation pipeline
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('encoder', OneHotEncoder(handle_unknown="ignore"))  
    ])

    # Combining transformations using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    # Defining the pipeline with SVM classifier
    pipe_svm_ksi = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(random_state=17))
    ])

    # Using SMOTE to handle class imbalance
    # pipe_svm_ksi = imPipeline([  # Use imPipeline to handle SMOTE
    # ('preprocessor', preprocessor),
    # ('smote', SMOTE(random_state=17)),  # Apply SMOTE to balance classes
    # ('svm', SVC(random_state=17))  # SVM classifier
    # ])


    # SVM hyperparameter grid 
    param_grid_svm = [
        {'svm__kernel': ['linear'], 'svm__C': [1, 10, 100]},  # linear kernel    
        {'svm__kernel': ['rbf'], 'svm__C': [1, 10, 100], 'svm__gamma': [0.3, 1.0, 3.0]},  # rbf kernel rank#1 C=100, gamma=3.0
        {'svm__kernel': ['poly'], 'svm__C': [1, 10, 100], 'svm__gamma': [0.3, 1.0, 3.0], 'svm__degree': [2, 3]}  # poly kernel 
    ]


    # Performing Grid Search with cross-validation
    grid_search_ksi = GridSearchCV(estimator=pipe_svm_ksi, param_grid=param_grid_svm, scoring='accuracy', refit=True, verbose=3)

    return grid_search_ksi

def train_and_evaluate_model(model_name, grid_search, X_train, y_train, X_test, y_test, unseen_features, unseen_labels):
    try:
        # Fit the model
        print(f"\n===================== {model_name.upper()} GRID SEARCH FIT =====================")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predict using the best model
        y_pred = best_model.predict(X_test)

        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Assuming ModelPerformance is a custom class you've defined to handle the confusion matrix
        modelPerformance = ModelPerformance(best_model, X_test, y_test)
        modelPerformance.conf_matrix("./images/confusion_matrix.png")

        print("\n===================== CONFUSION MATRIX =====================")
        print("\nConfusion Matrix:\n", cm)

        # Print classification report for detailed performance metrics
        print("\n===================== CLASSIFICATION REPORT =====================")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

        #Precison, Recall, F1-Score
        print("\n===================== PRECISION, RECALL, F1-SCORE =====================")
        print("\nPrecision:", precision_score(y_test, y_pred, average='weighted'))
        print("Recall:", recall_score(y_test, y_pred, average='weighted'))
        print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_pred, average='weighted'))

        # Save the best model
        print("\n===================== BEST MODEL METRICS =====================")
        print("\nBest Parameters:", grid_search.best_params_)
        print("Best Estimator:", grid_search.best_estimator_)
        print("Best Training Accuracy:", grid_search.best_score_)

        accuracy = best_model.score(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Handling unseen data
        label_map = {'Fatal': 0, 'Non-Fatal Injury': 1}
        unseen_labels_numeric = unseen_labels.map(label_map)

        # Fit the model on unseen data and make predictions
        best_model.fit(unseen_features, unseen_labels_numeric)
        unseen_predictions = best_model.predict(unseen_features)
        unseen_accuracy = best_model.score(unseen_features, unseen_labels_numeric)

        print("\n===================== UNSEEN DATA METRICS =====================")
        print("\nUnseen Predictions:", unseen_predictions)
        for i in range(len(unseen_features)):
            print(f"Predicted: {unseen_predictions[i]} Actual: {unseen_labels_numeric.iloc[i]}")

        print(f"Unseen Data Accuracy: {unseen_accuracy:.4f}")

        # Save the model using joblib
        joblib.dump(best_model, f"./model pickel/best_{model_name.lower()}_model.pkl")
        joblib.dump(grid_search, f"./model pickel/pipe_{model_name.lower()}_grid_search.pkl")
        print("\n===================== BEST MODEL SAVED =====================")

    except Exception as e:
        print("\n===================== ERROR =====================")
        print(f"An error occurred in training the model: {e}")