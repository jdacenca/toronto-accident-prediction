import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imPipeline
from model_performance import ModelPerformance
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import csv

# ===================== DATA OVERVIEW =====================
def data_overview(df):
    """Displays an overview of the dataset."""
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 100)

    print("\n===================== DATA OVERVIEW =====================")
    print("\nFirst 3 Records:\n", df.head(3))
    print("\nShape of the DataFrame:", df.shape)
    print("\nData Types:\n", df.dtypes)

    print("\n===================== DATA DESCRIPTION =====================")
    print("\nStatistical Summary:\n", df.describe())

    print("\n===================== COLUMN INFORMATION =====================")
    df.info()

    print("\n===================== MISSING VALUES =====================")
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_data / len(df)) * 100
    print(pd.concat([missing_data, missing_percent], axis=1, keys=['Total Missing', 'Percent Missing']))

    print("\n===================== UNIQUE VALUES =====================")
    for column in df.columns:
        print(f"\nUnique values in {column}:", df[column].unique())


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

    # # aggregate rows with same ACCNUM, DATE, TIME, LATITUDE, LONGITUDE
    df2 = df2.groupby(['ACCNUM'], as_index=False).apply(aggregate_rows).reset_index(drop=True)
    df2 = df2.groupby(['DATE', 'TIME', 'LATITUDE', 'LONGITUDE'], as_index=False).apply(aggregate_rows).reset_index(drop=True)

    df2.drop(columns=['ACCNUM'], inplace=True)
    # Format date and time
    df2["DATE"] = pd.to_datetime(df2["DATE"]).dt.to_period("D").astype(str)

    # Extract date components
    df2['MONTH'] = pd.to_datetime(df2['DATE']).dt.month
    df2['DAY'] = pd.to_datetime(df2['DATE']).dt.day
    df2['WEEK'] = pd.to_datetime(df2['DATE']).dt.isocalendar().week
    df2['DAYOFWEEK'] = pd.to_datetime(df2['DATE']).dt.dayofweek

    # Extract hour from TIME
    df2['HOUR'] = df2['TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))

    # Drop the original DATE, TIME column
    df2.drop(columns=['DATE','TIME'], inplace=True)

    # Replace specific values
    df2['ROAD_CLASS'] = df2['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False)

    # Fill missing values
    unknown_columns = ['PEDCOND','CYCCOND','DISTRICT']
    other_columns = ['ROAD_CLASS', 'ACCLOC', 'VISIBILITY', 'LIGHT', 'RDSFCOND','INVAGE','TRAFFCTL','INVTYPE', 'IMPACTYPE',]
    boolean_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH',
                       'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY','EMERG_VEH']

    df2[other_columns] = df2[other_columns].fillna("Other")
    df2[unknown_columns] = df2[unknown_columns].fillna("NA")
    df2[boolean_columns] = df2[boolean_columns].fillna("No")

    # Convert boolean columns to numeric
    df2[boolean_columns] = df2[boolean_columns].replace({'Yes': 1, 'No': 0}).astype(float)

    # Handle age column
    df2['INVAGE'] = df2['INVAGE'].replace('unknown', np.nan)
    df2['INVAGE'] = df2['INVAGE'].str.replace('OVER 95', '95 to 100')
    df2[['min_age', 'max_age']] = df2['INVAGE'].str.split(' to ', expand=True)
    df2['min_age'] = pd.to_numeric(df2['min_age'], errors='coerce')
    df2['max_age'] = pd.to_numeric(df2['max_age'], errors='coerce')
    df2['AVG_AGE'] = df2[['min_age', 'max_age']].mean(axis=1).astype(float)
    df2.drop(columns=['INVAGE','min_age', 'max_age'], inplace=True)
    df2['INVAGE'] = df2['AVG_AGE'].fillna(df2['AVG_AGE'].mean()).astype(float)
    df2.drop(columns=['AVG_AGE'], inplace=True)

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
    if group.nunique().eq(1).all():
        return group.iloc[0]
    else:
        aggregated = group.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean() if x.dtype != 'O' else None)
        return aggregated

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


# ===================== DATA PREPROCESSING =====================
def data_preprocessing_svm(features, smote=False):
    """Prepares the data for SVM by applying preprocessing and optional SMOTE."""
    num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
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

    if smote:
        pipe_svm_ksi = imPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=17)),
            ('svm', SVC(random_state=17))
        ])
    else:
        pipe_svm_ksi = Pipeline([
            ('preprocessor', preprocessor),
            ('svm', SVC(random_state=17))
        ])

    return pipe_svm_ksi


# ===================== GRID SEARCH =====================
def grid_search_svm(pipe_svm_ksi, param_grid_svm):
    """Performs Grid Search with cross-validation."""
    return GridSearchCV(estimator=pipe_svm_ksi, param_grid=param_grid_svm, scoring='accuracy', refit=True, verbose=3)


# ===================== MODEL TRAINING AND EVALUATION =====================
# Global list to store results
results = []

def train_and_evaluate_model(model_name, grid_search, X_train, y_train, X_test, y_test, unseen_features, unseen_labels, class_imb, smote):
    """
    Trains and evaluates the model and logs results.

    Parameters:
    - model_name: Name of the model.
    - grid_search: GridSearchCV object.
    - X_train, y_train: Training data.
    - X_test, y_test: Test data.
    - unseen_features, unseen_labels: Unseen data for evaluation.
    - class_imb: Class imbalance method used.
    - smote: Whether SMOTE was applied.
    """
    print(f"\n===================== {model_name.upper()} GRID SEARCH FIT =====================")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    unseen_pred = best_model.predict(unseen_features)

    # Calculate metrics
    train_acc = best_model.score(X_train, y_train) * 100
    test_acc = best_model.score(X_test, y_test) * 100
    unseen_acc = best_model.score(unseen_features, unseen_labels) * 100
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    roc = roc_auc_score(y_test, y_pred, average='weighted')

    # Log results
    results.append({
        "Model": model_name,
        "Kernel": best_model.named_steps['svm'].kernel,
        "Train Acc.%": f"{train_acc:.2f}",
        "Test Acc.%": f"{test_acc:.2f}",
        "Unseen Acc.%": f"{unseen_acc:.2f}",
        "Parameters": grid_search.best_params_,
        "Precision": f"{precision:.4f}",
        "F1-Score": f"{f1:.4f}",
        "Recall": f"{recall:.4f}",
        "ROC Score": f"{roc:.4f}",
        "Class Imbalance": class_imb,
        "SMOTE": smote
    })

    # Confusion Matrix and ROC Curve
    model_performance = ModelPerformance(best_model, X_test, y_test)
    model_performance.conf_matrix(f"./insights/svm_tuning/{model_name}/confusion_matrix.png")
    model_performance.roc_cur(f"./insights/svm_tuning/roc_curve.png")
    model_performance.classification_report_heatmap(f"./insights/svm_tuning/classification_report.png")

    print("\n===================== CLASSIFICATION REPORT =====================")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("\n===================== METRICS =====================")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Unseen Accuracy: {unseen_acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC Score: {roc:.4f}")

    # Save the model
    os.makedirs("./model_pickles", exist_ok=True)
    joblib.dump(best_model, f"./model_pickles/best_{model_name.lower()}.pkl")
    print("\nModel saved successfully.")

    return best_model

# ===================== SAVE RESULTS TO MD =====================
def save_results_to_md(file_path):
    """
    Saves the results to a Markdown file.

    Parameters:
    - file_path: Path to the Markdown file.
    """
    with open(file_path, "w") as f:
        # Write table header
        f.write("| Model | Kernel | Train Acc.% | Test Acc.% | Unseen Acc.% | Parameters | Precision | F1-Score | Recall | ROC Score | Class Imbalance | SMOTE |\n")
        f.write("|-------|--------|-------------|------------|--------------|------------|-----------|----------|--------|-----------|----------------|-------|\n")

        # Write table rows
        for result in results:
            f.write(f"| {result['Model']} | {result['Kernel']} | {result['Train Acc.%']} | {result['Test Acc.%']} | {result['Unseen Acc.%']} | {result['Parameters']} | {result['Precision']} | {result['F1-Score']} | {result['Recall']} | {result['ROC Score']} | {result['Class Imbalance']} | {result['SMOTE']} |\n")

    print(f"\nResults saved to {file_path}")

# ===================== SAVE RESULTS TO CSV =====================
def save_results_to_csv(file_path):
    """
    Saves the results to a CSV file.

    Parameters:
    - file_path: Path to the CSV file.
    """
    with open(file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Model", "Kernel", "Train Acc.%", "Test Acc.%", "Unseen Acc.%", 
            "Parameters", "Precision", "F1-Score", "Recall", "ROC Score", 
            "Class Imbalance", "SMOTE"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {file_path}")