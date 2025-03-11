from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from helper import runGridSearchCV, generateMetrics
import pandas as pd

def mlp_classifier(X_train, X_test, y_train, y_test):
    print("Neural Networks")

    # Define the hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Number of neurons in hidden layers
        'activation': ['relu', 'tanh', 'logistic'],      # Activation functions
        'solver': ['adam', 'sgd'],                       # Solvers for weight optimization
        'alpha': [0.0001, 0.001, 0.01],                  # Regularization parameter
        'learning_rate': ['constant', 'adaptive'],       # Learning rate schedule
    }


    # Create a MLP classifier
    model = MLPClassifier(max_iter=500)
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

    generateMetrics("Metrics for Neural Networks", y_train, train_predictions, y_test, test_predictions)