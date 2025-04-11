from sklearn.neural_network import MLPClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from helper import runGridSearchCV, analysis, custom_permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shap

from collections import Counter

def mlp_classifier(desc, X_train, X_test, y_train, y_test, unseen_fatal, unseen_notfatal):
    
    print(f'Neural Networks {desc}')
    count = Counter(y_train)
    print(f"Data Distribution: {count}")

    # Define the hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(20, 10, 1), (20, 15, 10, 1), (15, 10, 1)],  # Number of neurons in hidden layers
        'activation': ['relu', 'tanh', 'logistic', 'identity'],      # Activation functions
        'solver': ['adam'],                       # Solvers for weight optimization
        'alpha': [0.0001, 0.001, 0.01, 0.1],                  # Regularization parameter
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],       # Learning rate schedule
        'max_iter': [1000],
        #'batch_size': ['auto', 32, 64, 128]
    }

    # Create a MLP classifier
    model = MLPClassifier()
    '''
    categorical_transformer = Pipeline(
        steps=[
            ('label', LabelEncoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('label', categorical_transformer, column_le)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('mlp', MLPClassifier(random_state=32))
        ]
    )
    '''

    # Find the best hyperparameter
    tuning_model = runGridSearchCV(model, param_grid, X_train, y_train, X_test, y_test)

    # Check the permutation importance
    #custom_permutation_importance(model, X_train, y_train)

    # Metrics
    analysis(tuning_model.best_estimator_, "Metrics for Neural Networks", X_train, y_train, X_test, y_test, desc)

    ##
    efs = ExhaustiveFeatureSelector(tuning_model.best_estimator_, min_features=1, max_features=5, scoring='accuracy', cv=5)
    efs.fit(X_train, y_train)

    print("Best Features (Exhaustive Feature):", efs.best_feature_names_)

    explainer = shap.KernelExplainer(tuning_model.best_estimator_.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    print("Best Features (SHAP):", shap_values)

    # Summary plot of feature importance
    shap.summary_plot(shap_values, X_test, plot='bar')

    with open('./output/mlp_model.pkl', 'wb') as file:
        pickle.dump(tuning_model.best_estimator_, file)

    print("Model saved as mlp_model.pkl")   

    #------------------------------------------------------------------------------------
    # Fatal Data

    # Test for unseen data
    features = unseen_fatal.drop(columns=["ACCLASS"], axis=1)
    target = unseen_fatal["ACCLASS"]

    y_pred = tuning_model.best_estimator_.predict(features)
    y_pred = pd.DataFrame(y_pred)
    print(f"Fatal Real Data: {target}")
    print(f"Fatal Predicted Data: {y_pred}")
    #------------------------------------------------------------------------------------
    # Not Fatal Data
    # Test for unseen data
    features = unseen_notfatal.drop(columns=["ACCLASS"], axis=1)
    target = unseen_notfatal["ACCLASS"]
    
    y_pred = tuning_model.best_estimator_.predict(features)
    y_pred = pd.DataFrame(y_pred)
    print(f"Not Fatal Real Data: {target}")
    print(f"Not Fatal Predicted Data: {y_pred}")

