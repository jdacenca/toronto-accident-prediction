
import logging
import numpy as np
from typing import Union, Tuple
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from utils.config import RANDOM_STATE, TARGET

def _validate_and_convert_types(X_train: Union[pd.DataFrame, np.ndarray], 
                              y_train: np.ndarray) -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
    """Validate and convert data types of input features and labels.
    
    Args:
        X_train: Training features (DataFrame or ndarray)
        y_train: Training labels (ndarray)
        
    Returns:
        tuple: Processed X_train and y_train with appropriate data types
    """
    # if isinstance(X_train, pd.DataFrame):
    #     logging.info("X_train data types before conversion:")
    #     for col in X_train.columns:
    #         logging.info(f"{col}: {X_train[col].dtype}")
    #         # Check for any non-finite values
    #         non_finite = X_train[col].isna().sum()
    #         if non_finite > 0:
    #             logging.warning(f"Column {col} has {non_finite} non-finite values")
    
    # Convert y_train to numpy array if it's not already
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    y_train = y_train.astype(np.int32)
    
    # Convert X_train to appropriate types if it's a DataFrame
    if isinstance(X_train, pd.DataFrame):
        # Identify numerical columns (those that should be float64)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        # Convert each column separately with error handling
        for col in X_train.columns:
            try:
                if col in numerical_cols:
                    # Convert numerical columns to float64
                    X_train[col] = X_train[col].astype(np.float64)
                else:
                    # Preserve integer type for categorical and binary columns
                    X_train[col] = X_train[col].astype(np.int32)
            except Exception as e:
                logging.error(f"Error converting column {col}: {str(e)}")
                logging.error(f"Unique values in {col}: {X_train[col].unique()}")
                raise
    
    return X_train, y_train

def _convert_resampled_to_dataframe(X_resampled: np.ndarray, 
                                  X_train: Union[pd.DataFrame, np.ndarray],
                                  numerical_cols: pd.Index) -> Union[pd.DataFrame, np.ndarray]:
    """Convert resampled numpy array back to DataFrame if input was DataFrame.
    
    Args:
        X_resampled: Resampled features as numpy array
        X_train: Original training features (DataFrame or ndarray)
        numerical_cols: Columns that should be float64
        
    Returns:
        Resampled features as DataFrame or ndarray
    """
    if isinstance(X_train, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        # Restore original data types
        for col in X_train.columns:
            if col in numerical_cols:
                X_resampled[col] = X_resampled[col].astype(np.float64)
            else:
                X_resampled[col] = X_resampled[col].astype(np.int32)
    return X_resampled

def apply_sampling(X_train: Union[pd.DataFrame, np.ndarray], 
                  y_train: np.ndarray, 
                  method: str = 'smote') -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
    """Apply sampling technique to balance the classes.
    
    Args:
        X_train: Training features (DataFrame or ndarray)
        y_train: Training labels (ndarray)
        method: Sampling method to use. Options:
            - 'smote': Synthetic Minority Over-sampling Technique
            - 'random_over': Random oversampling of minority class
            - 'random_under': Random undersampling of majority class
            - 'smote_tomek': SMOTE followed by Tomek links cleaning
            - 'smote_enn': SMOTE followed by Edited Nearest Neighbors cleaning
        
    Returns:
        tuple: Resampled X_train and y_train
    """
    logging.info(f"Applying {method} sampling technique...")
    
    # Validate and convert data types
    X_train, y_train = _validate_and_convert_types(X_train, y_train)
    
    # Log class distribution before sampling
    unique, counts = np.unique(y_train, return_counts=True)
    logging.info(f"Class distribution before sampling: {dict(zip(unique, counts))}")
    
    # Apply the chosen sampling method
    if method == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif method == 'random_over':
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif method == 'random_under':
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=RANDOM_STATE)
    elif method == 'smote_enn':
        sampler = SMOTEENN(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
            
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Convert back to DataFrame if input was DataFrame
    if isinstance(X_train, pd.DataFrame):
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_resampled = _convert_resampled_to_dataframe(X_resampled, X_train, numerical_cols)
    
    # Log class distribution after sampling
    unique, counts = np.unique(y_resampled, return_counts=True)
    logging.info(f"Class distribution after sampling: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled

def separate_unseen_data(X: Union[pd.DataFrame, np.ndarray], 
                        y: np.ndarray, 
                        n_samples_per_class: int = 5) -> tuple[Union[pd.DataFrame, np.ndarray], 
                                                             np.ndarray, 
                                                             Union[pd.DataFrame, np.ndarray], 
                                                             np.ndarray]:
    """Separate unseen data with balanced classes from the dataset.
    
    Args:
        X: Features (DataFrame or ndarray)
        y: Labels (ndarray)
        n_samples_per_class: Number of samples to take from each class
        
    Returns:
        Tuple containing:
        - X_train: Training features
        - y_train: Training labels
        - X_unseen: Unseen features
        - y_unseen: Unseen labels
    """
    # Convert to DataFrame if not already
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    
    # Create a DataFrame with features and labels
    data = X_df.copy()
    data[TARGET] = y
    
    # Get samples from each class
    unseen_data = []
    for label in np.unique(y):
        class_samples = data[data[TARGET] == label].sample(n=n_samples_per_class, 
                                                          random_state=RANDOM_STATE)
        unseen_data.append(class_samples)
    
    # Combine unseen samples
    unseen_data = pd.concat(unseen_data)
    
    # Remove unseen samples from training data
    train_data = data.drop(unseen_data.index)
    
    # Split back into X and y
    X_train = train_data.drop(TARGET, axis=1)
    y_train = train_data[TARGET].values
    
    X_unseen = unseen_data.drop(TARGET, axis=1)
    y_unseen = unseen_data[TARGET].values
    
    # Convert back to original type if input was numpy array
    if isinstance(X, np.ndarray):
        X_train = X_train.values
        X_unseen = X_unseen.values
    
    logging.info(f"Separated {n_samples_per_class} samples per class for unseen data")
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Unseen data shape: {X_unseen.shape}")
    
    return X_train, y_train, X_unseen, y_unseen