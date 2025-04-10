import numpy as np
import pandas as pd
from data_preprocessor import (
    data_overview,
    data_cleaning,
    sample_and_update_data,
    data_preprocessing,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import csv
from model_performance import ModelPerformance
import os

# Random seed for reproducibility
np.random.seed(17)

# ===================== LOAD AND CLEAN DATA =====================
data_ksi = pd.read_csv("./data/Total_KSI.csv")

# Initial data overview
data_overview(data_ksi)

# Drop unnecessary columns            
columns_to_drop = [ 'OBJECTID', 'INDEX',  # index_id 
    'FATAL_NO', # sequence No. - high missing values
    'OFFSET', #high missing values
    'x', 'y','CYCLISTYPE', 'PEDTYPE', 'PEDACT', # high correlation
    'EMERG_VEH',       # 0 permutation importance 
    'CYCCOND',         # 0 permutation importance 
    "NEIGHBOURHOOD_158","NEIGHBOURHOOD_140","STREET1","STREET2","INJURY" # based on feature importance
]


# ===================== FUNCTION TO STORE METRICS =====================
def store_voting_metrics(classifiers, X_train, y_train, X_test, y_test, unseen_features, unseen_labels, voting_type, class_imb, results):
    """
    Stores metrics for hard and soft voting classifiers with various classifiers and generates performance plots.

    Parameters:
    - classifiers: List of classifiers to evaluate.
    - X_train, y_train: Training data.
    - X_test, y_test: Test data.
    - unseen_features, unseen_labels: Unseen data for evaluation.
    - voting_type: Type of voting ("Hard" or "Soft").
    - class_imb: Class imbalance method ("oversampling", "undersampling", or "original").
    - results: List to store all results.
    """
    print(f"\n===================== {voting_type.upper()} VOTING METRICS ({class_imb.upper()}) =====================")
    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Calculate metrics
        train_acc = classifier.score(X_train, y_train)
        test_acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

        # Evaluate on unseen data
        unseen_pred = classifier.predict(unseen_features)
        unseen_acc = accuracy_score(unseen_labels, unseen_pred)

        if isinstance(classifier, VotingClassifier):
            classifier_name = voting_type+str(classifier).split('(')[0].strip()

        else:
            classifier_name = str(classifier).split('(')[0].strip()

        results.append({
            "Classifier": classifier_name,
            "Class Imbalance": class_imb,
            "Train Accuracy": f"{train_acc:.4f}",
            "Test Accuracy": f"{test_acc:.4f}",
            "Unseen Accuracy": f"{unseen_acc:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "F1 Score": f"{f1:.4f}",
            "ROC AUC": f"{roc_auc:.4f}"
        })

        # Print metrics
        print(f"\nClassifier: {classifier}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Unseen Accuracy: {unseen_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")


        if isinstance(classifier, VotingClassifier):
            
            # Generate and save performance plots
            folder_path = f"./ensemble_performance/{voting_type.lower()}voting_{class_imb.lower()}"
            os.makedirs(folder_path, exist_ok=True)

            #pickle the model
            import joblib
            joblib.dump(classifier, f"ensemble_performance/voting_pickles/{voting_type.lower()}voting_{class_imb.lower()}_model.pkl")

            performance = ModelPerformance(classifier, X_test, y_test)
            performance.conf_matrix(f"ensemble_performance/{voting_type.lower()}voting_{class_imb.lower()}/confusion_matrix.png")
            performance.classification_report_heatmap(f"ensemble_performance/{voting_type.lower()}voting_{class_imb.lower()}/classification_report.png")
            performance.roc_cur(f"ensemble_performance/{voting_type.lower()}voting_{class_imb.lower()}/roc_curve.png")

# ===================== FUNCTION TO SAVE RESULTS =====================
def save_results_to_files(results, csv_file, md_file):
    """
    Saves all results to a single CSV and Markdown file.

    Parameters:
    - results: List of all results.
    - csv_file: Path to save the results as a CSV file.
    - md_file: Path to save the results as a Markdown file.
    """
    # Save results to CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_file}")

    # Save results to Markdown
    with open(md_file, mode='w') as f:
        f.write("| Classifier | Class Imbalance | Train Accuracy | Test Accuracy | Unseen Accuracy | Precision | Recall | F1 Score | ROC AUC |\n")
        f.write("|------------|-----------------|----------------|---------------|-----------------|-----------|--------|----------|---------|\n")
        for result in results:
            f.write(f"| {result['Classifier']} |  {result['Class Imbalance']} | {result['Train Accuracy']} | {result['Test Accuracy']} | {result['Unseen Accuracy']} | {result['Precision']} | {result['Recall']} | {result['F1 Score']} | {result['ROC AUC']} |\n")

    print(f"\nResults saved to {md_file}")

# ===================== TRAIN AND EVALUATE MODELS =====================
def process_and_train(data, columns_to_drop, class_imb, results):
    print(f"\n===================== {class_imb.upper()} =====================")

    # Clean the data
    cleaned_df = data_cleaning(data, columns_to_drop, class_imb=class_imb if class_imb != "original" else None)
    data_overview(cleaned_df)

    # Split the data into features and target
    unseen_features, unseen_labels, cleaned_df, features, target = sample_and_update_data(cleaned_df)

    # Encode the target variable
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)

    # Split the data into train & test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, stratify=target, test_size=0.2, random_state=17
    )

    # Preprocess the data
    preprocessor = data_preprocessing(features)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    unseen_features = preprocessor.transform(unseen_features)

    # Define the classifiers
    log_reg_H = LogisticRegression(max_iter=1400)
    dt_H = DecisionTreeClassifier(criterion='entropy', max_depth=42)
    nn_H = MLPClassifier(activation='tanh', alpha=0.01, hidden_layer_sizes=(15, 10, 1), learning_rate='invscaling', max_iter=1000, solver='adam')
    svm_H = SVC(C=0.1, kernel='poly', degree=3, gamma=0.1)
    svm_soft_H = SVC(C=0.1, kernel='poly', degree=3, gamma=0.1, probability=True)  # For soft voting
    rf_H = RandomForestClassifier(n_estimators=1000, random_state=37, n_jobs=-1, class_weight='balanced')

    # Hard voting
    voting_H = VotingClassifier(estimators=[('lr', log_reg_H), ('rf', rf_H), ('svm', svm_H), ('dt', dt_H), ('nn', nn_H)], voting='hard')
    classifiers_hard = [log_reg_H, rf_H, svm_H, dt_H, nn_H, voting_H]

    store_voting_metrics(
        classifiers_hard,
        X_train,
        y_train,
        X_test,
        y_test,
        unseen_features,
        unseen_labels,
        "Hard",
        class_imb,
        results
    )

    # Soft voting
    voting_S = VotingClassifier(estimators=[('lr', log_reg_H), ('rf', rf_H), ('svm', svm_soft_H), ('dt', dt_H), ('nn', nn_H)], voting='soft')
    classifiers_soft = [log_reg_H, rf_H, svm_soft_H, dt_H, nn_H, voting_S]

    store_voting_metrics(
        classifiers_soft,
        X_train,
        y_train,
        X_test,
        y_test,
        unseen_features,
        unseen_labels,
        "Soft",
        class_imb,
        results
    )

# ===================== MAIN EXECUTION =====================
# Initialize results list
results = []

# Process and train for each class imbalance method
class_imbalance_methods = ["oversampling", "undersampling"]

for method in class_imbalance_methods:
    process_and_train(data_ksi, columns_to_drop, class_imb=method, results=results)

# Save all results to a single CSV and Markdown file
save_results_to_files(results, "./ensemble_performance/results/voting_metrics.csv", "./ensemble_performance/results/voting_metrics.md")