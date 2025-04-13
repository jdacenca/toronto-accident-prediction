import joblib
import pandas as pd
import numpy as np

from transform import apply_binary_mapping, apply_target_mapping, apply_label_encoding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.neural_network import MLPClassifier


def load_model(path):
    try:
        with open(path, 'rb') as file:
            data = joblib.load(file)  # Load the pickled data
            return data
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        return None
    except ModuleNotFoundError as e:
        print(f"Error loading the model: {e}")
        print("It seems the module containing the custom functions for the pipeline is not found.")
        #print(f"Please ensure '{transform.__name__}' is importable in this environment.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None
    # Display the data
    print(data)
    return data



