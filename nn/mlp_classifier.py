from sklearn.neural_network import MLPClassifier
from helper import runGridSearchCV, analysis
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def mlp_classifier(X_train, X_test, y_train, y_test):

    column_le = [
        'DATE',
        'ROAD_CLASS',
        'TRAFFCTL',
        'VISIBILITY',
        'LIGHT',
        'RDSFCOND',
        'IMPACTYPE',
        'VEHTYPE',
        'NEIGHBOURHOOD_158'
    ]
    
    le = LabelEncoder()

    for column in column_le:
        X_train[column] = le.fit_transform(X_train[column])
        X_test[column] = le.fit_transform(X_test[column])

    print("Neural Networks")

    # Define the hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(20, 10, 1), (20, 15, 1), (15, 10, 1)],  # Number of neurons in hidden layers
        'activation': ['relu', 'tanh', 'logistic', 'identity'],      # Activation functions
        'solver': ['adam', 'sgd'],                       # Solvers for weight optimization
        'alpha': [0.0001, 0.001, 0.01],                  # Regularization parameter
        'learning_rate': ['constant', 'adaptive', 'invscaling'],       # Learning rate schedule
        'max_iter': [1000]
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
    tuning_model = runGridSearchCV(model, param_grid, X_train, y_train, X_test, y_test)

    #------------------------------------------------------------------------------------
    # Training Data
    y_pred = tuning_model.best_estimator_.predict(X_train)
    y_pred = pd.DataFrame(y_pred)
    train_predictions = pd.DataFrame(y_pred)
    #------------------------------------------------------------------------------------
    # Test Data
    y_pred = tuning_model.best_estimator_.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    test_predictions = pd.DataFrame(y_pred)

    analysis(tuning_model.best_estimator_, "Metrics for Neural Networks", X_train, y_train, X_test, y_test)