from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
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

    print("Number of Unique Counts", df.nunique())

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
    # Team agreed to update the entry with missing label
    df['ACCLASS'] = df['ACCLASS'].fillna("Fatal")
    
    # Dropped ACCLASS with Property Damage : 10 Entries in the dataset 
    df.drop(df[df['ACCLASS'] == 'PROPERTY DAMAGE O'].index, inplace=True)

    df['MONTH'] = pd.to_datetime(df['DATE']).dt.month
    df['DAY'] = pd.to_datetime(df['DATE']).dt.day
    df['WEEK'] = pd.to_datetime(df['DATE']).dt.isocalendar().week
    df['DAYOFWEEK'] = pd.to_datetime(df['DATE']).dt.dayofweek
    df.drop(['DATE'], axis=1, inplace=True)

    df['TIME'] = df['TIME'].apply(convert_to_time)
    df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M').dt.hour # Update time to per hour
    df.drop(['TIME'], axis=1, inplace=True)
    df['ROAD_CLASS'] = df['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False) # Update the incorrect Road Class with space


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

    other_column =  [
        'ACCLOC',
        'TRAFFCTL',
        'VISIBILITY',
        'LIGHT',
        'RDSFCOND',
        'IMPACTYPE',
        'INVTYPE',
        'ROAD_CLASS',
        'DISTRICT'
    ]

    na_column =  [
        'PEDCOND',
        'CYCCOND'
    ]

    df[other_column] = df[other_column].fillna("Other")
    df[na_column] = df[na_column].fillna("na")
    df[boolean_columns] = df[boolean_columns].fillna("No")

    categorical_columns = df.select_dtypes(include=[object, 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Change all values to UpperCase
    df[categorical_columns] = df[categorical_columns].apply(lambda col: col.str.upper())

    # Fields to be dropped depending on the dataset
    df_drop = df.drop(drop_fields, axis=1)

    return df, df_drop

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


def custom_permutation_importance(model, X_train, y_train):
    # Compute for the permutation importance for feature ranking
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=32)
    print(f"Permutation Importance: {perm_importance.importances_mean}")

    importance_mean = perm_importance.importances_mean
    importance_std = perm_importance.importances_std

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance_mean)), importance_mean, yerr=importance_std)
    plt.xticks(range(len(importance_mean)), X_train.columns, rotation=90)
    plt.title("Permutation Importance")
    plt.ylabel("Decrease in Acuracy")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()

def select_best_features(X, y, col_names):
    selector = SelectKBest(score_func=f_classif, k=30)
    selected_features = selector.fit(X, y)

    scores = selector.scores_
    sorted_indices = np.argsort(scores)[::-1]

    plt.figure(figsize=(20, 8))
    plt.bar(range(len(scores)), scores[sorted_indices], color='skyblue')
    plt.xticks(range(len(scores)), col_names[sorted_indices], rotation=90)
    plt.title("Feature Importance")
    plt.ylabel("Scores")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.savefig("./output/beast_features.png")

def recursive_feature_elimination(X, y):
    model = RandomForestClassifier()

    rfe = RFE(model, n_features_to_select=30)
    rfe.fit(X, y)

    ranking = rfe.ranking_
    selected_features = X.columns[rfe.support_]

    print("Selected Features:", selected_features)

    ranking_df = pd.DataFrame({'Feature': X.columns, 'Ranking': ranking})
    ranking_df.sort_values(by='Ranking', inplace=True)

    plt.figure(figsize=(12, 6))
    plt.bar(ranking_df['Feature'], ranking_df['Ranking'], color='skyblue')
    plt.title("Feature Rankings from RFE")
    plt.xlabel("Features")
    plt.ylabel("Ranking (Lower is better)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("./output/recursive_feature_elimination.png")

def variance_threshold(X, y):
    selector = VarianceThreshold()
    fit_selector = selector.fit_transform(X)

    print("Reduced Dataset:")
    print(fit_selector)

    variances = selector.variances_
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.bar(features, variances, color='skyblue')
    plt.axhline(y=0.5, color="red", linestyle="--", label="Threshold")
    plt.title("Variance of Features")
    plt.xlabel("Features")
    plt.ylabel("Variance")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("./output/variance_threshold.png")
