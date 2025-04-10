import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score, accuracy_score)
from sklearn.inspection import permutation_importance

# SHAP for model interpretability
try:
    import shap
except ImportError:
    logging.warning("SHAP library not found. SHAP importance calculations will be skipped. Install with: pip install shap")
    shap = None # Set shap to None if not installed

# tqdm for progress bars
from tqdm.auto import tqdm


# --- Configuration ---
# Define output directories
PERFORMANCE_OUTPUT_DIR = 'insights/Refined_Random_Forest_Performance' # New folder for this run's performance results
ARTIFACT_DIR = 'insights/serialized_artifacts' # Standard folder for reusable artifacts
DATA_PATH = 'data/TOTAL_KSI_6386614326836635957.csv' # Path to your data file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')

# --- Preprocessing Pipeline Definitions ---
# (Keep the DataCleaner, FeatureEngineer, and create_preprocessing_pipeline functions exactly as they were in the previous version)
class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning the accident data (excluding Property Damage Only)"""
    # Define columns to drop during initial cleaning
    COLUMNS_TO_DROP = [
        'INDEX', 'OBJECTID', 'FATAL_NO', 'HOOD_140',
        'NEIGHBOURHOOD_140', 'HOOD_158', # Strong correlation with deprecated HOOD_140
    ]

    def __init__(self):
        self.binary_cols = None
        self.ordinal_encoders = {}
        self.numerical_medians = {}
        self.categorical_modes = {}
        self.feature_names_ = None # To store column names after cleaning (excluding target)

    def _clean_initial_data(self, df):
        """Removes 'Property Damage O' records and specified columns."""
        df_copy = df.copy()
        # Keep only rows where ACCLASS indicates injury or fatal (KSI)
        index_range_to_keep = df_copy['ACCLASS'] != 'Property Damage O'
        df_cleaned = df_copy[index_range_to_keep]
        # Drop specified columns, ignoring errors if columns don't exist
        df_cleaned = df_cleaned.drop(columns=self.COLUMNS_TO_DROP, errors='ignore')
        return df_cleaned

    def _identify_binary_columns(self, df_cleaned):
        """Identify columns with binary ('Yes'/'No') values."""
        binary_cols = df_cleaned.select_dtypes(include=['object']).apply(
            lambda x: x.nunique() <= 2 and set(x.unique()).issubset({'Yes', 'No', np.nan})
        )
        return binary_cols[binary_cols].index.tolist()

    def _store_numerical_medians(self, df_cleaned):
        """Store median values for numerical columns for imputation."""
        numerical_cols = df_cleaned.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            self.numerical_medians[col] = df_cleaned[col].median()

    def _initialize_categorical_encoders(self, df_cleaned):
        """Initialize ordinal encoders and store modes for non-binary categorical columns."""
        # Select object columns, exclude target ('ACCLASS') and already identified binary cols
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'ACCLASS' and col not in self.binary_cols]

        for col in categorical_cols:
            # Store mode for imputation
            mode_val = df_cleaned[col].mode()
            self.categorical_modes[col] = mode_val[0] if not mode_val.empty else 'Unknown' # Handle empty mode

            # Initialize and fit OrdinalEncoder on unique non-null values + mode
            self.ordinal_encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # Combine unique values and mode to ensure mode is encoded
            fit_values = pd.concat([df_cleaned[col].dropna(), pd.Series([self.categorical_modes[col]])]).unique()
            self.ordinal_encoders[col].fit(fit_values.reshape(-1, 1))


    def _transform_binary_columns(self, X):
        """Transform binary columns to 0/1, filling NaNs with 'No' (0)."""
        X_copy = X.copy()
        for col in self.binary_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna('No')
                X_copy[col] = (X_copy[col] == 'Yes').astype(int)
        return X_copy

    def _transform_numerical_columns(self, X):
        """Fill missing values in numerical columns using stored medians."""
        X_copy = X.copy()
        for col, median in self.numerical_medians.items():
            if col in X_copy.columns and X_copy[col].isnull().any():
                 # Check if median is valid (not NaN)
                 if pd.notna(median):
                     X_copy[col] = X_copy[col].fillna(median)
                 else:
                     # Fallback if median itself was NaN (e.g., all values were NaN)
                     X_copy[col] = X_copy[col].fillna(0)
                     logging.warning(f"Median for numerical column '{col}' was NaN. Filled NaNs with 0.")
        return X_copy

    def _transform_categorical_columns(self, X):
        """Transform non-binary categorical columns using stored ordinal encoders."""
        X_copy = X.copy()
        for col, encoder in self.ordinal_encoders.items():
            if col in X_copy.columns:
                # Impute missing values with the stored mode before transforming
                fill_value = self.categorical_modes.get(col, 'Unknown')
                X_copy[col] = X_copy[col].fillna(fill_value)

                # Reshape for the encoder and transform
                try:
                    encoded_values = encoder.transform(X_copy[col].values.reshape(-1, 1))
                    X_copy[col] = encoded_values.flatten() # Flatten back to 1D
                except ValueError as e:
                     logging.error(f"Error transforming categorical column '{col}': {e}")
                     # Handle potential new categories not seen during fit (should be handled by 'use_encoded_value')
                     # If error persists, might need to investigate further or assign default value
                     X_copy[col] = -1 # Assign unknown value
        return X_copy

    def fit(self, df, y=None): # y is ignored but needed for Pipeline compatibility
        """Fits the cleaner by identifying column types and storing imputation values."""
        logging.info("DataCleaner: Fitting...")
        df_cleaned = self._clean_initial_data(df)
        self.binary_cols = self._identify_binary_columns(df_cleaned)
        self._store_numerical_medians(df_cleaned)
        self._initialize_categorical_encoders(df_cleaned)
        # Store the feature names *after* cleaning and dropping, excluding the target
        self.feature_names_ = df_cleaned.drop('ACCLASS', axis=1, errors='ignore').columns.tolist()
        logging.info(f"DataCleaner: Fit complete. Features identified: {len(self.feature_names_)}")
        return self

    def transform(self, df):
        """Transforms the data by cleaning, imputing, and encoding."""
        logging.info("DataCleaner: Transforming...")
        df_cleaned = self._clean_initial_data(df)
        # Select only the features identified during fit (excluding target)
        # Important: Use self.feature_names_ to ensure consistency
        X = df_cleaned.drop('ACCLASS', axis=1, errors='ignore')

        # Ensure X contains only the columns expected from fitting
        # Handle cases where columns might be missing in the input df for transform
        cols_to_keep = [col for col in self.feature_names_ if col in X.columns]
        if len(cols_to_keep) < len(self.feature_names_):
             missing_cols = list(set(self.feature_names_) - set(cols_to_keep))
             logging.warning(f"DataCleaner Transform: Input missing columns expected from fit: {missing_cols}. They won't be in the output.")
        X = X[cols_to_keep]


        # Apply transformations in order
        X = self._transform_numerical_columns(X) # Impute numerical first
        X = self._transform_binary_columns(X)   # Then encode binary
        X = self._transform_categorical_columns(X) # Then encode other categoricals

        # Final check: Reindex to ensure columns match feature_names_ exactly, adding missing ones if needed
        # This ensures the output shape is consistent, filling potential missing cols with NaN (or handle differently)
        # However, imputation should have handled NaNs. This reindex mainly ensures column order and presence.
        X = X.reindex(columns=self.feature_names_, fill_value=np.nan) # Or maybe fill with 0 or median/mode?
        # Check if NaNs were introduced by reindexing (shouldn't happen if imputation is correct)
        if X.isnull().sum().sum() > 0:
             logging.warning(f"DataCleaner Transform: NaNs detected after final reindex. Review imputation steps. Columns with NaNs: {X.columns[X.isnull().any()].tolist()}")
             # Applying imputation again as a safety net
             X = self._transform_numerical_columns(X)
             # For categorical/binary, NaN implies missing column, potentially fill with 0 or mode encoded value?
             # Let's fill remaining NaNs with 0 for simplicity, assuming they are numeric after encoding.
             X.fillna(0, inplace=True)


        logging.info(f"DataCleaner: Transform complete. Output shape: {X.shape}")
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.location_clusters = None
        self.time_period_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.feature_names_in_ = None # Store feature names seen during fit
        self.feature_names_out_ = None # Store feature names after transformation

    def fit(self, X, y=None):
        logging.info("FeatureEngineer: Fitting...")
        self.feature_names_in_ = X.columns.tolist()

        # Fit location clusters if latitude and longitude are available
        if all(col in self.feature_names_in_ for col in ['LATITUDE', 'LONGITUDE']):
            coords = X[['LATITUDE', 'LONGITUDE']].copy()
            # Handle potential NaNs before fitting KMeans
            coords_filled = coords.fillna(coords.median())
            # Ensure there are enough valid samples to fit KMeans
            if coords_filled.shape[0] >= self.n_clusters:
                 self.location_clusters = KMeans(n_clusters=self.n_clusters, random_state=48, n_init=10)
                 self.location_clusters.fit(coords_filled)
                 logging.info(f"FeatureEngineer: KMeans fitted with {self.n_clusters} clusters.")
            else:
                 logging.warning(f"FeatureEngineer: Not enough samples ({coords_filled.shape[0]}) to fit KMeans with {self.n_clusters} clusters. Skipping location clustering.")
                 self.location_clusters = None # Ensure it's None if not fitted
        else:
            logging.warning("FeatureEngineer: LATITUDE or LONGITUDE columns not found. Skipping location clustering.")
            self.location_clusters = None

        # Fit time period encoder
        time_periods = [
            ['Night (0-5)'], ['Morning (6-11)'],
            ['Afternoon (12-16)'], ['Evening (17-20)'],
            ['Night (21-23)'], ['Unknown'] # Add Unknown category
        ]
        self.time_period_encoder.fit(time_periods)
        logging.info("FeatureEngineer: TimePeriod encoder fitted.")

        # Determine output feature names
        self._set_output_feature_names(X)
        logging.info(f"FeatureEngineer: Fit complete. Expected output features: {self.feature_names_out_}")
        return self

    def transform(self, X):
        logging.info("FeatureEngineer: Transforming...")
        X_transformed = X.copy()

        # Time-based features
        hour_col_created = False
        if 'TIME' in self.feature_names_in_:
            try:
                # Convert TIME to hour (robustly handling potential non-numeric values)
                time_str = X_transformed['TIME'].fillna(-1).astype(int).astype(str).str.zfill(4)
                X_transformed['Hour'] = pd.to_numeric(time_str.str[:2], errors='coerce')
                # Impute any NaNs in Hour (e.g., from original -1 or conversion errors) with a reasonable default like median
                median_hour = X_transformed['Hour'].median()
                if pd.isna(median_hour): median_hour = 12 # Fallback median
                X_transformed['Hour'] = X_transformed['Hour'].fillna(median_hour).astype(int)
                hour_col_created = True
            except Exception as e:
                logging.warning(f"FeatureEngineer: Could not process TIME column: {e}. Skipping Hour/TimePeriod features derived from TIME.")

        if 'DATE' in self.feature_names_in_ and hour_col_created:
            try:
                # Use 'Hour' to create 'TimePeriod'
                X_transformed['TimePeriod'] = pd.cut(
                    X_transformed['Hour'],
                    bins=[-1, 5, 11, 16, 20, 23],
                    labels=['Night (0-5)', 'Morning (6-11)',
                           'Afternoon (12-16)', 'Evening (17-20)',
                           'Night (21-23)'],
                    right=True # Include right edge (e.g., hour 5 is in Night (0-5))
                )
                # Handle potential NaNs introduced by cut (if Hour had issues despite imputation)
                # Add 'Unknown' category if not present and fill NaNs
                if 'Unknown' not in X_transformed['TimePeriod'].cat.categories:
                     X_transformed['TimePeriod'] = X_transformed['TimePeriod'].cat.add_categories('Unknown')
                X_transformed['TimePeriod'] = X_transformed['TimePeriod'].fillna('Unknown')

                # Encode TimePeriod
                encoded_values = self.time_period_encoder.transform(
                    X_transformed['TimePeriod'].astype(str).values.reshape(-1, 1) # Ensure string type for encoder
                )
                X_transformed['TimePeriodEncoded'] = encoded_values.flatten() # Use new column name
            except Exception as e:
                logging.warning(f"FeatureEngineer: Error creating/encoding TimePeriod: {e}. Assigning default value.")
                X_transformed['TimePeriodEncoded'] = -1 # Assign default unknown value
        elif 'DATE' in self.feature_names_in_:
             # If DATE exists but Hour could not be created
             logging.warning("FeatureEngineer: DATE exists but Hour could not be created. Assigning default TimePeriodEncoded.")
             X_transformed['TimePeriodEncoded'] = -1
        # else: TimePeriodEncoded column won't be created if DATE/TIME/Hour processing failed


        # Location-based features
        if self.location_clusters is not None and all(col in self.feature_names_in_ for col in ['LATITUDE', 'LONGITUDE']):
            try:
                coords = X_transformed[['LATITUDE', 'LONGITUDE']].copy()
                coords_filled = coords.fillna(coords.median()) # Impute NaNs for prediction/distance calc

                # Predict cluster
                X_transformed['LOCATION_CLUSTER'] = self.location_clusters.predict(coords_filled)

                # Calculate distance to assigned cluster center
                cluster_centers = self.location_clusters.cluster_centers_
                assigned_centers = cluster_centers[X_transformed['LOCATION_CLUSTER']]
                X_transformed['DISTANCE_TO_CLUSTER_CENTER'] = np.sqrt(
                    ((coords_filled.values - assigned_centers) ** 2).sum(axis=1)
                )
            except Exception as e:
                 logging.warning(f"FeatureEngineer: Error processing location features: {e}. Assigning default values.")
                 X_transformed['LOCATION_CLUSTER'] = -1
                 X_transformed['DISTANCE_TO_CLUSTER_CENTER'] = -1
        # else: Location columns won't be created if clustering wasn't fitted or LAT/LON missing


        # Drop intermediate/original columns that were replaced or are no longer needed
        cols_to_drop = ['TIME', 'DATE', 'Hour', 'TimePeriod', 'LATITUDE', 'LONGITUDE']
        X_transformed = X_transformed.drop(columns=[col for col in cols_to_drop if col in X_transformed.columns], errors='ignore')

        # Ensure output columns match the expected feature names out
        # Add any missing columns expected in output (e.g., if conditional features weren't created)
        if self.feature_names_out_: # Check if feature_names_out_ is set
             for col in self.feature_names_out_:
                 if col not in X_transformed.columns:
                     logging.warning(f"FeatureEngineer: Expected output column '{col}' not generated. Adding with default value -1.")
                     X_transformed[col] = -1 # Assign a default value
        else:
            logging.error("FeatureEngineer: feature_names_out_ not set during transform. Cannot ensure output columns.")
            # Fallback? Or maybe let it fail later? For now, just log.


        # Return only the expected output columns in the correct order
        if self.feature_names_out_:
            # Ensure all expected columns exist before indexing
            missing_final_cols = list(set(self.feature_names_out_) - set(X_transformed.columns))
            if missing_final_cols:
                 logging.error(f"FeatureEngineer: Final output columns missing after processing: {missing_final_cols}")
                 # Decide how to handle: error out, or return what exists?
                 # Let's return what exists but log error
                 existing_cols = [col for col in self.feature_names_out_ if col in X_transformed.columns]
                 final_X = X_transformed[existing_cols]
            else:
                 final_X = X_transformed[self.feature_names_out_]
        else:
             logging.error("FeatureEngineer: Cannot determine final output columns. Returning processed DataFrame.")
             final_X = X_transformed


        logging.info(f"FeatureEngineer: Transform complete. Output shape: {final_X.shape}")
        return final_X

    def _set_output_feature_names(self, X_fit):
        """Determines the feature names that will be output by the transform method."""
        # Start with input features
        current_features = self.feature_names_in_.copy()
        # Remove features that will be dropped
        features_to_remove = ['TIME', 'DATE', 'LATITUDE', 'LONGITUDE'] # Keep Hour/TimePeriod temporarily if needed
        current_features = [f for f in current_features if f not in features_to_remove]
        # Add new features conditionally
        if 'DATE' in self.feature_names_in_ and 'TIME' in self.feature_names_in_:
            current_features.append('TimePeriodEncoded')
        if self.location_clusters is not None and all(col in self.feature_names_in_ for col in ['LATITUDE', 'LONGITUDE']):
            current_features.append('LOCATION_CLUSTER')
            current_features.append('DISTANCE_TO_CLUSTER_CENTER')
        # Remove intermediate features if they were added implicitly
        current_features = [f for f in current_features if f not in ['Hour', 'TimePeriod']]

        self.feature_names_out_ = list(dict.fromkeys(current_features)) # Keep unique and preserve order


    def get_feature_names_out(self, input_features=None):
        """Return feature names for output"""
        # If fit hasn't been called, we can't know the output names
        if self.feature_names_out_ is None:
             logging.warning("get_feature_names_out called before fit or output names not set.")
             # Try to determine based on input_features if provided? Risky.
             # Return empty list as a safe default.
             return []
        return self.feature_names_out_


def create_preprocessing_pipeline(n_clusters=10):
    """Create the main preprocessing pipeline including cleaning and feature engineering."""
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer(n_clusters=n_clusters)),
    ])

# --- Core Modeling Functions ---

def setup_directories():
    """Create necessary directories for outputs using configured paths"""
    # Use the globally defined variables
    dirs = [ARTIFACT_DIR, PERFORMANCE_OUTPUT_DIR]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directories ensured: {dirs}")

def load_and_preprocess_data(data_path=DATA_PATH): # Use configured path
    """Load and preprocess the data using the defined pipeline"""
    logging.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        if df.empty:
             logging.error(f"Data file loaded but is empty: {data_path}")
             return None, None, None, None
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {data_path}")
        return None, None, None, None
    except pd.errors.EmptyDataError:
         logging.error(f"Error: Data file is empty: {data_path}")
         return None, None, None, None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None, None, None

    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline()

    logging.info("Fitting preprocessing pipeline...")
    try:
        # Fit the pipeline on the whole dataset to learn transformations
        pipeline.fit(df) # Fit requires the original df including ACCLASS for DataCleaner logic
    except Exception as e:
        logging.error(f"Error fitting pipeline: {e}", exc_info=True) # Log traceback
        return None, None, None, None

    logging.info("Transforming data using pipeline...")
    try:
        # Transform the data (features only)
        X = pipeline.transform(df)
    except Exception as e:
        logging.error(f"Error transforming data with pipeline: {e}", exc_info=True)
        return None, None, None, None

    # Create target variable AFTER cleaning but from original df
    try:
        # Re-apply initial cleaning to get the correct index alignment for the target
        if 'cleaner' in pipeline.named_steps and hasattr(pipeline.named_steps['cleaner'], '_clean_initial_data'):
             df_cleaned_for_target = pipeline.named_steps['cleaner']._clean_initial_data(df.copy())
        else:
             logging.error("Could not find 'cleaner' step or its method in pipeline for target creation.")
             return None, None, None, None

        # Ensure alignment by using the index from the *transformed* X
        valid_index = X.index.intersection(df_cleaned_for_target.index)
        if len(valid_index) != len(X.index):
            logging.warning(f"Index mismatch between transformed X ({len(X.index)}) and cleaned data ({len(valid_index)} overlapping). Re-aligning X and y.")
            X = X.loc[valid_index] # Align X to the common index

        # Create y using the common index
        y = (df_cleaned_for_target.loc[valid_index, 'ACCLASS'] == 'Fatal').astype(int)

        if X.shape[0] != y.shape[0]:
             # This check should ideally not fail if alignment is correct
             logging.error(f"FATAL: Mismatch between feature rows ({X.shape[0]}) and target rows ({y.shape[0]}) after alignment.")
             return None, None, None, None

    except KeyError as e:
         logging.error(f"KeyError creating target variable ('ACCLASS' missing or index issue): {e}")
         return None, None, None, None
    except Exception as e:
         logging.error(f"Unexpected error creating target variable: {e}", exc_info=True)
         return None, None, None, None

    # Check for NaNs in final X (should be handled by pipeline, but check anyway)
    if X.isnull().sum().sum() > 0:
        logging.warning(f"NaN values found in final preprocessed data (X) BEFORE feature name alignment: \n{X.isnull().sum()[X.isnull().sum() > 0]}")
        # Attempt final fill - consider logging more detail or raising error depending on requirements
        logging.warning("Attempting final NaN fill with 0.")
        X.fillna(0, inplace=True)

    # Get final feature names from the pipeline
    feature_names = []
    try:
        # Get feature names from the last step (usually 'engineer')
        if hasattr(pipeline.steps[-1][1], 'get_feature_names_out'):
             feature_names = pipeline.steps[-1][1].get_feature_names_out()
        elif hasattr(pipeline.steps[-1][1], 'feature_names_out_'): # Check direct attribute
             feature_names = pipeline.steps[-1][1].feature_names_out_
        elif hasattr(pipeline.steps[-1][1], 'feature_names_'): # Fallback
             feature_names = pipeline.steps[-1][1].feature_names_

        if not feature_names: # If still empty, use X's columns as last resort
            feature_names = X.columns.tolist()
            logging.warning("Could not retrieve definite feature names from pipeline. Using X.columns.")

    except Exception as e:
        logging.error(f"Error retrieving feature names from pipeline: {e}")
        feature_names = X.columns.tolist() # Fallback

    # Final check: Ensure X columns match feature_names list *exactly*
    if list(X.columns) != feature_names:
        logging.warning(f"Mismatch between final X.columns ({len(X.columns)}) and retrieved feature_names ({len(feature_names)}). Attempting to re-align X columns.")
        try:
            X = X[feature_names] # Reorder/select columns in X to match feature_names
            logging.info("Re-aligned X columns based on retrieved feature_names.")
        except KeyError as e:
            missing_in_X = list(set(feature_names) - set(X.columns))
            extra_in_X = list(set(X.columns) - set(feature_names))
            logging.error(f"KeyError during final column alignment: {e}.")
            logging.error(f"Features in feature_names but missing in X: {missing_in_X}")
            logging.error(f"Features in X but not in feature_names: {extra_in_X}")
            logging.error("Cannot proceed with inconsistent features. Returning None.")
            return None, None, None, None # Critical failure
        except Exception as e:
            logging.error(f"Unexpected error during final column alignment: {e}. Returning None.")
            return None, None, None, None # Critical failure


    logging.info(f"Preprocessing complete. Final feature shape: {X.shape}")
    logging.info(f"Target variable shape: {y.shape}")
    logging.info(f"Number of Fatal accidents (target=1): {y.sum()}")
    logging.info(f"Number of Non-Fatal accidents (target=0): {len(y) - y.sum()}")

    return X, y, pipeline, feature_names


def calculate_feature_importance(X, y, feature_names):
    """Determine important features using methods suitable for Random Forest"""
    if not feature_names or X.shape[1] != len(feature_names):
         logging.error("Feature names invalid or mismatch with X columns in calculate_feature_importance. Aborting.")
         return None

    logging.info(f"Calculating feature importance. Input feature shape: {X.shape}")
    # Use the globally configured performance output directory
    output_dir = PERFORMANCE_OUTPUT_DIR

    # --- Scaling ---
    scaler = StandardScaler()
    try:
         X_scaled_values = scaler.fit_transform(X.values)
         X_scaled_df = pd.DataFrame(X_scaled_values, columns=feature_names, index=X.index)
         logging.info(f"Data scaled for importance calculation. Shape: {X_scaled_df.shape}")
    except Exception as e:
         logging.error(f"Error scaling data for importance: {e}. Aborting importance calculation.")
         return None

    # --- Base Model Training ---
    try:
        rf_base = RandomForestClassifier(n_estimators=100, random_state=48, class_weight='balanced', n_jobs=-1)
        rf_base.fit(X_scaled_df, y)
        logging.info("Base Random Forest model trained for importance calculation.")
    except Exception as e:
        logging.error(f"Error training base RF for importance: {e}. Aborting importance calculation.")
        return None

    importance_results = {}

    # --- 1. MDI Importance ---
    try:
        logging.info("Calculating RF MDI (Mean Decrease in Impurity)...")
        mdi_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_base.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        importance_results['MDI'] = mdi_importance

        logging.info(f"Top 5 features (MDI): {mdi_importance['feature'].head(5).tolist()}")
        mdi_importance.to_csv(f'{output_dir}/rf_mdi_importance.csv', index=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=mdi_importance.head(20), y='feature', x='importance', hue='feature', palette='viridis', dodge=False, legend=False)
        plt.title('Random Forest Feature Importance (MDI)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rf_mdi_importance.png')
        plt.close()
        logging.info("RF MDI importance saved.")
    except Exception as e:
        logging.error(f"Error calculating/plotting MDI importance: {e}")


    # --- 2. Permutation Importance ---
    try:
        logging.info("Calculating Permutation Importance...")
        perm_result = permutation_importance(
            rf_base, X_scaled_df, y, n_repeats=10, random_state=48, n_jobs=-1, scoring='f1'
        )
        perm_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_result.importances_mean,
            'importance_std': perm_result.importances_std
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
        importance_results['Permutation'] = perm_importance_df[['feature', 'importance_mean']].rename(columns={'importance_mean': 'importance'})

        logging.info(f"Top 5 features (Permutation): {perm_importance_df['feature'].head(5).tolist()}")
        perm_importance_df.to_csv(f'{output_dir}/permutation_importance.csv', index=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=perm_importance_df.head(20), y='feature', x='importance_mean', hue='feature', palette='plasma', dodge=False, legend=False)
        plt.title('Permutation Feature Importance (Mean Decrease in F1 Score)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/permutation_importance.png')
        plt.close()
        logging.info("Permutation importance saved.")
    except Exception as e:
        logging.error(f"Error calculating/plotting Permutation importance: {e}")


    # --- 3. SHAP Values (Calculated on a Sample with Progress Bar) ---
    if shap is None: # Check if SHAP was imported successfully
         logging.warning("SHAP library not available. Skipping SHAP importance.")
    else:
        try:
            logging.info("Calculating SHAP values (with progress bar)...")
            explainer = shap.TreeExplainer(rf_base)

            # --- Create a sample of the data ---
            sample_size = min(3000, X_scaled_df.shape[0])
            if sample_size < X_scaled_df.shape[0]:
                 logging.info(f"Using a sample of {sample_size} rows for SHAP calculation.")
                 X_shap_sample = X_scaled_df.sample(sample_size, random_state=48)
            else:
                 logging.info("Using the full dataset for SHAP calculation (dataset size <= sample limit).")
                 X_shap_sample = X_scaled_df

            # --- Calculate SHAP values in chunks with tqdm ---
            chunk_size = 100
            num_samples = X_shap_sample.shape[0]
            shap_values_list_class0 = []
            shap_values_list_class1 = []
            is_binary_shap = False

            pbar = tqdm(range(0, num_samples, chunk_size), desc="Calculating SHAP Chunks")
            for i in pbar:
                start_idx = i
                end_idx = min(i + chunk_size, num_samples)
                X_chunk = X_shap_sample.iloc[start_idx:end_idx]
                if X_chunk.empty: continue

                chunk_shap_values = explainer.shap_values(X_chunk)

                if isinstance(chunk_shap_values, list) and len(chunk_shap_values) == 2:
                     is_binary_shap = True
                     shap_values_list_class0.append(chunk_shap_values[0])
                     shap_values_list_class1.append(chunk_shap_values[1])
                else:
                     shap_values_list_class1.append(chunk_shap_values)

            # Combine results from chunks
            if is_binary_shap:
                 shap_values_class0_combined = np.vstack(shap_values_list_class0)
                 shap_values_class1_combined = np.vstack(shap_values_list_class1)
                 shap_values = [shap_values_class0_combined, shap_values_class1_combined]
                 shap_values_for_summary = shap_values[1]
                 logging.info("Combined binary SHAP values from chunks.")
            else:
                 shap_values_combined = np.vstack(shap_values_list_class1)
                 shap_values = shap_values_combined
                 shap_values_for_summary = shap_values
                 logging.info("Combined single-output SHAP values from chunks.")

            # --- Calculate Importance & Save ---
            mean_abs_shap = np.abs(shap_values_for_summary).mean(axis=0)
            shap_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            importance_results['SHAP'] = shap_importance_df

            logging.info(f"Top 5 features (SHAP): {shap_importance_df['feature'].head(5).tolist()}")
            shap_importance_df.to_csv(f'{output_dir}/shap_importance.csv', index=False)

            # --- Generate Plots ---
            plt.figure()
            shap.summary_plot(shap_values_for_summary, X_shap_sample, feature_names=feature_names, show=False)
            plt.title('SHAP Summary Plot (Impact on Fatal Prediction - Sampled Data)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_summary_plot.png', bbox_inches='tight')
            plt.close()

            plt.figure()
            shap.summary_plot(shap_values_for_summary, X_shap_sample, feature_names=feature_names, plot_type="bar", show=False)
            plt.title('SHAP Mean Absolute Importance (Sampled Data)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_bar_plot.png', bbox_inches='tight')
            plt.close()
            logging.info("SHAP values calculated (chunked w/ progress) and plots saved.")

        except Exception as e:
            logging.error(f"Error calculating or plotting SHAP values: {e}", exc_info=True)
            logging.warning("Skipping SHAP importance due to error.")
            importance_results['SHAP'] = pd.DataFrame({'feature': feature_names, 'importance': 0})


    # --- 4. Comparison ---
    if not importance_results:
        logging.warning("No importance results were successfully generated. Skipping comparison.")
        return None

    try:
        logging.info("Creating comparison of feature importance methods...")
        comparison_df = pd.DataFrame(index=feature_names)
        for method, df in importance_results.items():
            if isinstance(df, pd.DataFrame) and 'importance' in df.columns and 'feature' in df.columns:
                 temp_df = df.set_index('feature')['importance']
                 total_importance = temp_df.sum()
                 if total_importance > 1e-9:
                     normalized_importance = temp_df / total_importance
                 else:
                     normalized_importance = temp_df
                 comparison_df[f'{method}_Normalized'] = normalized_importance
            else:
                 logging.warning(f"Invalid format for importance results of method '{method}'. Skipping.")


        comparison_df.fillna(0, inplace=True)
        comparison_df['Average_Normalized'] = comparison_df.mean(axis=1)
        comparison_df.sort_values('Average_Normalized', ascending=False, inplace=True)

        # Save comparison CSV to the specific performance directory
        comparison_df.to_csv(f'{output_dir}/importance_comparison.csv')

        # Plot comparison (Top N features)
        top_n = min(30, comparison_df.shape[0])
        plt.figure(figsize=(12, 10))
        plot_data = comparison_df.head(top_n).drop('Average_Normalized', axis=1)
        plot_data.plot(kind='bar', width=0.8, ax=plt.gca())
        plt.title(f'Top {top_n} Feature Importance Comparison (Normalized)')
        plt.ylabel('Normalized Importance Score')
        plt.xlabel('Feature')
        plt.xticks(rotation=90)
        plt.legend(title='Method')
        plt.tight_layout()
        # Save plot to the specific performance directory
        plt.savefig(f'{output_dir}/importance_comparison_bar.png')
        plt.close()

        # Heatmap comparison
        plt.figure(figsize=(10, max(8, top_n // 2)))
        sns.heatmap(plot_data, annot=True, cmap='viridis', fmt=".3f")
        plt.title(f'Top {top_n} Feature Importance Heatmap (Normalized)')
        plt.yticks(rotation=0)
        plt.tight_layout()
        # Save plot to the specific performance directory
        plt.savefig(f'{output_dir}/importance_comparison_heatmap.png')
        plt.close()
        logging.info("Feature importance comparison completed.")

        # Save ranked features to the specific performance directory
        top_features = comparison_df.index.tolist()
        with open(f'{output_dir}/top_features_ranked.txt', 'w') as f:
            f.write("Ranked Features (based on average normalized importance):\n")
            for i, feature in enumerate(top_features, 1):
                f.write(f"{i}. {feature}\n")

        return top_features

    except Exception as e:
         logging.error(f"Error during importance comparison: {e}", exc_info=True)
         return None


def train_and_evaluate_random_forest(X, y, feature_names):
    """Train, tune (refined), and evaluate the RandomForestClassifier model"""
    if not feature_names or X.shape[1] != len(feature_names):
         logging.error("Feature names invalid or mismatch with X columns in train_and_evaluate. Aborting.")
         return None, None # Return None for model and scaler

    logging.info("Starting Random Forest training and evaluation...")
    # Use the globally configured directories
    output_dir = PERFORMANCE_OUTPUT_DIR
    artifact_dir = ARTIFACT_DIR

    # --- 1. Scale Features ---
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X.values)
        joblib.dump(scaler, f'{artifact_dir}/scaler_rf.pkl') # Save to artifact dir
        logging.info(f"Features scaled using StandardScaler. Scaler saved to {artifact_dir}/scaler_rf.pkl")
    except Exception as e:
        logging.error(f"Error scaling data for training: {e}. Aborting training.")
        return None, None

    # --- 2. Split Data (Stratified) ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=48, stratify=y
        )
        logging.info(f"Data split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")
        logging.info(f"Train target distribution: {np.bincount(y_train)} (0s, 1s)")
        logging.info(f"Test target distribution: {np.bincount(y_test)} (0s, 1s)")
    except Exception as e:
        logging.error(f"Error during train/test split: {e}. Aborting training.")
        return None, None

    # --- 3. Hyperparameter Tuning using GridSearchCV (Refined Grid) ---
    logging.info("Setting up GridSearchCV for Random Forest (Refined Search)...")
    # --- Use the REFINED parameter grid ---
    param_grid = {
        'n_estimators': [250, 300, 400],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [3, 5, 7],
        'max_features': ['sqrt'],
        'class_weight': ['balanced_subsample']
    }
    # --- End of Refined Grid ---

    rf = RandomForestClassifier(random_state=48, n_jobs=-1)
    scoring = {'f1': 'f1', 'roc_auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}

    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5,
        scoring=scoring, refit='f1', n_jobs=-1, verbose=2
    )

    logging.info("Performing refined grid search...")
    start_gs = time.time()
    try:
        grid_search.fit(X_train, y_train)
        end_gs = time.time()
        logging.info(f"Refined grid search completed in {end_gs - start_gs:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error during refined GridSearchCV fitting: {e}. Aborting training.", exc_info=True)
        return None, None

    # --- Log Grid Search Results ---
    logging.info("\n--- Refined Grid Search Results ---")
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    try:
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        # Save CV results to the specific performance directory
        cv_results_df.to_csv(f'{output_dir}/grid_search_cv_results.csv', index=False)
        logging.info(f"Full cross-validation results saved to {output_dir}/grid_search_cv_results.csv")
    except Exception as e:
        logging.warning(f"Could not save CV results to CSV: {e}")


    # --- 4. Evaluate Best Model ---
    best_rf_model = grid_search.best_estimator_
    logging.info("Evaluating the best model from refined search on the test set...")
    try:
        y_pred = best_rf_model.predict(X_test)
        y_prob = best_rf_model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Non-Fatal', 'Fatal'])
        avg_precision = average_precision_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_score = auc(fpr, tpr)

        logging.info("\n--- Test Set Evaluation (Refined Model) ---")
        logging.info(f"Accuracy: {test_accuracy:.4f}")
        logging.info(f"Average Precision (PR Curve): {avg_precision:.4f}")
        logging.info(f"AUC (ROC Curve): {roc_auc_score:.4f}")
        logging.info("\nClassification Report:\n" + report)

        # Save evaluation summary to the specific performance directory
        with open(f'{output_dir}/evaluation_summary.txt', 'w') as f:
            f.write("Random Forest Model Evaluation Summary (Refined Tuning)\n" + "=" * 55 + "\n\n")
            f.write(f"Best Parameters Found:\n{grid_search.best_params_}\n\n")
            f.write(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}\n\n")
            f.write("--- Test Set Metrics ---\n")
            f.write(f"Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"ROC AUC: {roc_auc_score:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

    except Exception as e:
        logging.error(f"Error during refined model evaluation on test set: {e}. Cannot generate plots.", exc_info=True)
        return best_rf_model, scaler


    # --- 5. Create Visualizations ---
    try:
        # Confusion Matrix
        plt.figure(figsize=(7, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fatal', 'Fatal'], yticklabels=['Non-Fatal', 'Fatal'])
        plt.title('Confusion Matrix (Refined Model)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        # Save plot to the specific performance directory
        plt.savefig(f'{output_dir}/confusion_matrix.png')
        plt.close()

        # ROC Curve
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC - Refined Model)')
        plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
        # Save plot to the specific performance directory
        plt.savefig(f'{output_dir}/roc_curve.png')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        no_skill = np.sum(y_test) / len(y_test)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
        plt.plot([0, 1], [no_skill, no_skill], color='red', lw=2, linestyle='--', label=f'No Skill (AP={no_skill:.2f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Refined Model)')
        plt.legend(loc="lower left"); plt.grid(True); plt.tight_layout()
        # Save plot to the specific performance directory
        plt.savefig(f'{output_dir}/precision_recall_curve.png')
        plt.close()
        logging.info(f"Evaluation plots saved to {output_dir}")

    except Exception as e:
        logging.error(f"Error generating evaluation plots: {e}", exc_info=True)

    return best_rf_model, scaler


def save_model_artifacts(model, pipeline, scaler):
    """Save the trained model, preprocessing pipeline, and scaler"""
    # Use the globally configured artifact directory
    artifact_dir = ARTIFACT_DIR
    logging.info(f"Saving model artifacts to {artifact_dir}...")
    try:
        if model:
            model_path = f'{artifact_dir}/random_forest_model.pkl' # Keep standard name unless versioning needed
            joblib.dump(model, model_path)
            logging.info(f"Model saved to {model_path}")
        else:
            logging.warning("Model object is None, skipping model saving.")

        if pipeline:
            pipeline_path = f'{artifact_dir}/preprocessing_pipeline_rf.pkl' # Keep standard name
            joblib.dump(pipeline, pipeline_path)
            logging.info(f"Preprocessing pipeline saved to {pipeline_path}")
        else:
            logging.warning("Pipeline object is None, skipping pipeline saving.")

        # Scaler is already saved in train_and_evaluate to artifact_dir
        if scaler is None:
             logging.warning("Scaler object is None.")

    except Exception as e:
        logging.error(f"Error saving artifacts: {e}", exc_info=True)

# --- Main Execution ---
def main():
    """Main execution function for Random Forest modeling"""
    total_start_time = time.time()

    # Setup directories (uses global vars PERFORMANCE_OUTPUT_DIR, ARTIFACT_DIR)
    setup_directories()

    # Load and preprocess data (uses global DATA_PATH)
    X, y, pipeline, feature_names = load_and_preprocess_data()

    # --- Critical Check after Loading/Preprocessing ---
    if X is None or y is None or pipeline is None or feature_names is None:
        logging.critical("Data loading or preprocessing failed. Cannot proceed. Exiting.")
        return
    if X.empty or y.empty:
         logging.critical("Data is empty after preprocessing. Cannot proceed. Exiting.")
         return
    logging.info("Data loaded and preprocessed successfully.")

    # Calculate feature importance (saves to PERFORMANCE_OUTPUT_DIR)
    try:
        top_features = calculate_feature_importance(X, y, feature_names)
        if top_features:
            logging.info(f"Top features identified (ranked): {top_features[:10]}...")
        else:
             logging.warning("Feature importance calculation did not return ranked features.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during feature importance calculation: {e}", exc_info=True)


    # Train, tune (refined grid) and evaluate (saves results to PERFORMANCE_OUTPUT_DIR, scaler to ARTIFACT_DIR)
    best_model = None
    scaler = None
    try:
        best_model, scaler = train_and_evaluate_random_forest(X, y, feature_names)
        if best_model is None:
             logging.error("Model training/evaluation failed.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during model training/evaluation: {e}", exc_info=True)
        # Try to recover model from grid_search if possible, but evaluation failed
        if 'grid_search' in locals() and hasattr(grid_search, 'best_estimator_'):
            best_model = grid_search.best_estimator_
            logging.warning("Attempting to save best model from GridSearch despite evaluation error.")


    # Save final model, pipeline (saves to ARTIFACT_DIR)
    if best_model and pipeline and scaler:
         save_model_artifacts(best_model, pipeline, scaler)
    else:
         logging.warning("Skipping artifact saving due to missing model, pipeline, or scaler.")


    total_end_time = time.time()
    execution_time = total_end_time - total_start_time
    logging.info(f"--- Random Forest Modeling Workflow Completed ---")
    logging.info(f"Total execution time: {execution_time:.2f} seconds")
    logging.info(f"Performance results saved in '{PERFORMANCE_OUTPUT_DIR}'")
    logging.info(f"Model artifacts saved in '{ARTIFACT_DIR}'")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()