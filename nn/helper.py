from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np

def data_description(df):
    # Understand the data 
    print("Data Description:")
    print(df.head(5))
    print(df.info())
    print(df.describe())

    print("\nMissing Data:")
    print(df.isnull().sum())

def unique_values(df):
    categorical_columns = df.select_dtypes(include=[object, 'category']).columns.tolist()
    
    # Check all the unique data
    for x in categorical_columns:
        print(f"\nUnique values in column {x}:")
        print(df[x].unique().tolist())

def convert_to_time(value):
    hours = value // 100
    minutes = value % 100
    return f"{hours:02d}:{minutes:02d}"

def clean_dataset(df, drop_fields):
    categorical_columns = df.select_dtypes(include=[object, 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Change all values to UpperCase
    df[categorical_columns] = df[categorical_columns].apply(lambda col: col.str.upper())

    # Fields to be dropped depending on the dataset
    df.drop(drop_fields, axis=1, inplace=True)

    # Team agreed to drop the entry with missing label
    df.dropna(subset=['ACCLASS'], inplace=True)

    # Dropped ACCLASS with Property Damage : 10 Entries in the dataset 
    df.drop(df[df['ACCLASS'] == 'PROPERTY DAMAGE O'].index, inplace=True)

    df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m') # Update date to per month
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m')
    df['TIME'] = df['TIME'].apply(convert_to_time)
    df['TIME'] = pd.to_datetime(df['TIME']).dt.hour # Update time to per hour
    df['ROAD_CLASS'] = df['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False) # Update the incorrect Road Class with space

    # Pedestrian and Passenger falls under 'Other' in INVTYPE
    #df['VEHTYPE'] = np.where(((df['INVTYPE'] == 'Pedestrian') | (df['INVTYPE'] == 'Passenger')) & (df['VEHTYPE'] != '') , 'Other', df['INVTYPE'])

    # Fill in empty fields for boolean columns
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
    #'VEHTYPE',
    'TRAFFCTL'
]

    other_column =  [
        'ROAD_CLASS',
        'VISIBILITY',
        'LIGHT',
        'RDSFCOND',
        'IMPACTYPE'
    ]

    df[unknown_column] = df[unknown_column].fillna("Unknown")
    df[other_column] = df[other_column].fillna("Other")
    df[boolean_columns] = df[boolean_columns].fillna("No")


    return df

def timer(start_time=None):
    if not start_time:
        start_time=datetime.now()
        return start_time
    elif start_time:
        thour,temp_sec=divmod((datetime.now()-start_time).total_seconds(),3600)
        tmin,tsec=divmod(temp_sec,60)
        print('\n Time taken: %i hours %i minutes and %s seconds.'%(thour,tmin,round(tsec,2)))


def runGridSearchCV(model, param_grid, X_train, y_train, X_test, y_test):

    # Using 5 fold
    tuning_model = GridSearchCV(model, param_grid=param_grid, cv=5, error_score='raise', verbose=3)

    start_time = timer(None)
    tuning_model.fit(X_train, y_train)
    timer(start_time)
    print("Best Parameters: ", tuning_model.best_params_)
    print("Best Score: ", tuning_model.best_score_)
    print("Test Score: ", tuning_model.score(X_test, y_test))

    return tuning_model

def analysis(model, model_name, X_train, y_train, X_test, y_test):

    print("\n")
    print("="*70)
    print(model_name)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("Train Data:")
    generateMetrics(model, X_train, y_train, train_predictions)

    print("Test Data:")
    generateMetrics(model, X_test, y_test, test_predictions)

    print("\n")
    print("="*70)

def generateMetrics(model, x, y, predictions):
    fold = KFold(n_splits=3, shuffle=True, random_state=1)
    scores1 = cross_val_score(model, x, y, cv=fold, scoring="accuracy")
    accuracy1 = accuracy_score(y, predictions)
    cm1 = confusion_matrix(y, predictions)

    y_train_pred = cross_val_predict(model, x, y, cv=fold)
    ConfusionMatrixDisplay.from_predictions(y, y_train_pred, normalize="true", values_format=".0%")
    plt.show()
    
    target = ['0', '1']
    labels = np.arange(2)
    report = classification_report(y, y_train_pred, labels=labels, target_names=target, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    plt.show()

    # Get the ROC
    y_probability = model.predict_proba(x)[:, 1]

    fpr, tpr, threshold = roc_curve(y, y_probability)
    roc_auc = roc_auc_score(y, y_probability)

    # Plot for the ROC Score
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print('\n' + '-'*50)
    print(f"Accuracy: {accuracy1}")
    print(f"Scores: {scores1}")
    print(f"Mean Score: {str(scores1.mean())}")
    print(f"Max Score: {str(scores1.max())}")
    print(f"Confusion Matrix:\n{cm1}")
    print(f"\nClassification report:\n")
    print(classification_report(y, y_train_pred))