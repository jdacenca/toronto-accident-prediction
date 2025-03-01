# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score,
                            ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle

# Data Loading
df_total_ksi = pd.read_csv('TOTAL_KSI.csv')

# Display information about the dataset
print(df_total_ksi.info())  # Column types and missing values

# Display summary statistics for all columns
print(df_total_ksi.describe(include='all'))  # Summary stats for all columns

# Calculate and display missing data
missing_data = df_total_ksi.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_data/len(df_total_ksi)) * 100
pd.concat([missing_data, missing_percent], axis=1, keys=['Total Missing', 'Percent Missing'])

# Feature Selection
relevant_features = [
    'DATE', 'TIME', 'LATITUDE', 
    'LONGITUDE', 'x', 'y',
    'OFFSET', 'ROAD_CLASS', 'ACCLOC',
    'TRAFFCTL', 'VISIBILITY', 'LIGHT',
    'RDSFCOND', 'IMPACTYPE', 'INVTYPE',
    'INVAGE', 'INJURY', 'INITDIR',  
    'VEHTYPE', 'MANOEUVER', 'DRIVACT', 
    'DRIVCOND', 'PEDTYPE', 'PEDACT', 
    'PEDCOND', 'CYCLISTYPE', 'CYCACT', 
    'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 
    'AUTOMOBILE', 'TRUCK', 'MOTORCYCLE', 
    'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 
    'SPEEDING', 'AG_DRIV', 'REDLIGHT', 
    'ALCOHOL', 'DISABILITY']

# Data Preprocessing
labels = df_total_ksi['ACCLASS']
print(labels.info())
df_features = df_total_ksi[relevant_features]
print(df_features.info())

# Identify numeric and categorical features
numeric_features = []
categorical_features = []

for col in df_features.columns:
    if df_features[col].dtype == 'object':
        categorical_features.append(col)
    else:
        numeric_features.append(col)

print('Numeric Features:', numeric_features)
print('Categorical Features:', categorical_features)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print(labels_encoded.shape)

# Preprocessor pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Model Pipeline with Hyperparameter Tuning
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=48))
])

param_grid = {
    'classifier__max_depth': [3, 5, 7, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features': ['sqrt', 'log2', None],
}

# Insert the last 10 items from df_features and labels_encoded into unseen_features and unseen_labels
unseen_features = df_features.iloc[-10:]
unseen_labels = labels_encoded[-10:]
# Remove the last 10 items from df_features and labels_encoded
df_features = df_features.iloc[:-10]
labels_encoded = labels_encoded[:-10]

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(df_features, labels_encoded, test_size=0.2, random_state=48)


# Model Training with Cross-Validation
grid_search = GridSearchCV(dt_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_dt = grid_search.best_estimator_

# Model Evaluation
y_pred = best_dt.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Estimator: {grid_search.best_estimator_}")
print(f"Training Accuracy: {grid_search.best_score_:.3f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Testing Precision: {precision_score(y_test, y_pred, average="micro"):.3f}")
print(f"Testing Recall: {recall_score(y_test, y_pred, average="micro"):.3f}")
print(f"Testing F1 Score: {f1_score(y_test, y_pred, average="micro"):.3f}")

# Visualization
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
plt.show()

# Plot the tree
plt.figure(figsize=(15,10))
plot_tree(best_dt.named_steps['classifier'], filled=True)
plt.show()

# Unseen Data Prediction
best_dt.fit(unseen_features, unseen_labels)
unseen_predictions = best_dt.predict(unseen_features)
print(f"Unseen Data Accuracy: {accuracy_score(unseen_labels, unseen_predictions):.3f}")

# Plot the tree
plt.figure(figsize=(15,10))
plot_tree(best_dt.named_steps['classifier'], filled=True)
plt.show()

# Model Serialization
with open('dt_model.pkl', 'wb') as f:
    pickle.dump(best_dt, f)