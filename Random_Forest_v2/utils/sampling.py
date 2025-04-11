"""
Utility module for data sampling techniques to handle class imbalance.
This module provides various sampling methods to address class imbalance in datasets.
"""

import logging
import numpy as np
from typing import Union
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from utils.config import RANDOM_STATE


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

    # Check if the dataset is large enough for SMOTE-based methods
    if method.startswith('smote'):
        # Count samples in minority class
        class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
        min_class_count = min(class_counts.values())

        if min_class_count < 6:  # SMOTE needs at least 6 samples in minority class by default
            # Fall back to random oversampling for small datasets
            logging.warning(f"Not enough samples for {method} (needs at least 6 in minority class). "
                            f"Falling back to random oversampling.")
            sampler = RandomOverSampler(random_state=RANDOM_STATE)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    # Log class distribution after sampling
    unique, counts = np.unique(y_resampled, return_counts=True)
    logging.info(f"Class distribution after sampling: {dict(zip(unique, counts))}")

    return X_resampled, y_resampled