""" Logistic Regression Model"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from LR_preprocessor import LRPreprocessor
from LR_evaluation import LREvaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV


import matplotlib.pyplot as plt

# %%
# Load the data
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
# To explore the data
def data_exploration(df_ksi):
    pass

# %%
# To visualize the data
def data_visualization(df_ksi):
    pass

# %%
def split_data(df_ksi):
    X_train, X_test, y_train, y_test  = train_test_split(df_ksi.drop('ACCLASS', axis=1),
                                    df_ksi['ACCLASS'],
                                    test_size=0.2, random_state=54)

    return X_train, X_test, y_train, y_test

# %%

def build_model(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=54)
    # log_reg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=10000,
    #                              class_weight='balanced', random_state=54)
    log_reg = LogisticRegression(penalty='l2', C=0.1, solver='saga', max_iter=10000,
                                 class_weight='balanced', random_state=54)

    # X_train, X_test, y_train, y_test = train_test_split(df.drop('ACCLASS', axis=1),
    #                                                     df['ACCLASS'],
    #                                                     test_size=0.2, random_state=54)
    log_reg.fit(X_train, y_train)
    cv_score_default = cross_val_score(log_reg, X_train, y_train, cv=skf, scoring='f1')
    print(cv_score_default)
    print(cv_score_default.mean())
    return log_reg, X_train, X_test, y_train, y_test
# %%

def build_model_default(X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=54)
    # log_reg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=10000,
    #                              class_weight='balanced', max_iter=100000, random_state=54)
    # log_reg = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=54)
    log_reg = LogisticRegression(max_iter=10000, random_state=54)

    # X_train, X_test, y_train, y_test = train_test_split(df.drop('ACCLASS', axis=1),
    #                                                     df['ACCLASS'],
    #                                                     test_size=0.2, stratify=df['ACCLASS'],
    #                                                     random_state=54)
    log_reg.fit(X_train, y_train)
    cv_score_default = cross_val_score(log_reg, X_train, y_train, cv=skf, scoring='f1')
    print(cv_score_default)
    print(cv_score_default.mean())
    # return log_reg, X_train, X_test, y_train, y_test
    return log_reg

# %%

def build_model_optimized(X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=54)
    # log_reg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=10000,
    #                              class_weight='balanced', max_iter=100000, random_state=54)
    # log_reg = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=54)
    log_reg = LogisticRegression( solver='saga', C=0.1, class_weight=None, max_iter = 10000,
                                  penalty= 'l1', random_state=54)
    log_reg.fit(X_train, y_train)

    # X_train, X_test, y_train, y_test = train_test_split(df.drop('ACCLASS', axis=1),
    #                                                     df['ACCLASS'],
    #                                                     test_size=0.2, stratify=df['ACCLASS'],
    #                                                     random_state=54)
    log_reg.fit(X_train, y_train)
    cv_score_default = cross_val_score(log_reg, X_train, y_train, cv=skf, scoring='f1')
    print(cv_score_default)
    print(cv_score_default.mean())
    # return log_reg, X_train, X_test, y_train, y_test
    return log_reg

# %%
############################
# Hyperparameter tuning

def LR_tuning(df):
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 54)
    log_reg = LogisticRegression() # Base model

    X_train, X_test, y_train, y_test = train_test_split(df.drop('ACCLASS', axis=1),
                                                        df['ACCLASS'],
                                                        test_size=0.2, random_state=54)
    param_grid = {
           'C': [0.1, 1, 10, 100],
           'penalty': ['l1', 'l2'],
           'solver': ['lbfgs','liblinear','newton-cg','saga'],
           # 'solver': ['liblinear'],
           'max_iter': [300, 1000],
           'class_weight': ['balanced', None],
           # 'class_weight': ['balanced'],
            'random_state' :[54]
       }

    grid_search = GridSearchCV(estimator= log_reg, param_grid = param_grid,
                               cv = skf, scoring = 'f1', refit = 'f1', n_jobs = -1, verbose = True)

    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.score(X_test, y_test))
    print(grid_search.best_estimator_)

# %%
############################
# Apply Sampling Technique



# %%
############################
## Main

from random import randint
from collections import Counter

if __name__ == '__main__':
    df_ksi = load_data()
    # data_exploration(df_ksi)
    # data_visualization(df_ksi)
    X_train, X_test, y_train, y_test = split_data(df_ksi)

    lr_preprocessor = LRPreprocessor(level=1)
    df_new_train = lr_preprocessor.fit_transform(pd.concat([X_train, y_train], axis=1))
    # X_new_train, y_new_train = df_new_train.iloc[:, :-1], df_new_train.iloc[:, -1]
    X_new_train, y_new_train = df_new_train.drop('ACCLASS', axis = 1), df_new_train['ACCLASS']
    print(df_new_train.head(5))
    print(df_new_train.info())

    # LR_tuning(df_new)


    # log_reg  = build_model_default(X_new_train, y_new_train)
    log_reg  = build_model_optimized(X_new_train, y_new_train)

    print(f"Class Distribution: {Counter(df_new_train['ACCLASS'])}, y_new_train: {Counter(y_new_train)}, y_test: {Counter(y_test)}")

    lr_preprocessor.level = 2   # Do not perform SMOTENC on transform
    evaluation = LREvaluation(log_reg, lr_preprocessor, X_test, y_test)


    round = randint(1,100)
    evaluation.confusion_matrix(fr'./insights/confusion_matrix{round}.png')
    evaluation.roc_auc(fr'./insights/roc_auc{round}.png')
    evaluation.precision_recall_auc(fr'./insights/precision_recall_auc{round}.png')
    evaluation.classification_report(fr'./insights/classification_report{round}.png')





"""
    ###############
    # Debug
    from LR_preprocessor import LRPreprocessor
    lr_preprocessor = LRPreprocessor()
    lr_preprocessor._init_clean(df_ksi)
    lr_preprocessor._impute(df_ksi)
"""