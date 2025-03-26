import numpy as np
import pandas as pd
from data_preprocessor import (
    data_overview,
    data_cleaning,
    sample_and_update_data,
    data_preprocessing_svm,
    train_and_evaluate_model,
    grid_search_svm,
    save_results_to_md,
    save_results_to_csv,
)
from data_visualizer import (
    scatter_plot,
    pie_chart,
    bar_chart,
    heatmap_correlation,
    heatmap_missing_values,
    hist_plot,
    pair_plot,
    spline_plot,
    cat_distribution,
    create_full_correlation_matrices,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Random seed for reproducibility
np.random.seed(17)

# ===================== LOAD AND CLEAN DATA =====================
data_ksi = pd.read_csv("./data/Total_KSI.csv")

# Initial data overview
data_overview(data_ksi)

# Drop unnecessary columns
columns_to_drop = [
    "OBJECTID",
    "INDEX",
    "FATAL_NO",
    "x",
    "y",
    "EMERG_VEH",
    "OFFSET",
    "CYCACT",
    "CYCCOND",
    "STREET1",
    "STREET2",
    "NEIGHBOURHOOD_140",
    "HOOD_140",
]

# Visualizations
visualizations = [
    (scatter_plot, "./insights/data_exploration/scatter_plot.png"),
    (pie_chart, "./insights/data_exploration/pie_chart.png"),
    (bar_chart, "./insights/data_exploration/bar_chart.png"),
    (heatmap_correlation, "./insights/correlation/heatmap_correlation.png"),
    (heatmap_missing_values, "./insights/data_exploration/heatmap_missing_values.png"),
    (hist_plot, "./insights/data_exploration/hist_plot.png"),
    (pair_plot, "./insights/data_exploration/pair_plot.png"),
    (cat_distribution, "./insights/data_exploration/cat_distribution.png"),
    (spline_plot, "./insights/data_exploration/spline_plot.png"),
]

for viz_func, save_path in visualizations:
    viz_func(data_ksi, save_path)

create_full_correlation_matrices(data_ksi)

# ===================== DEFINE PARAMETER GRIDS =====================
param_grids = {
    "svm_linear": [
        {"svm__kernel": ["linear"], "svm__C": [1]},
    ],
    "svm_rbf": [
        {"svm__kernel": ["rbf"], "svm__C": [1], "svm__gamma": [3.0]},
    ],
    "svm_poly": [
        {"svm__kernel": ["poly"], "svm__C": [10], "svm__gamma": [1.0], "svm__degree": [3]},
    ],
}

# ===================== TRAIN AND EVALUATE MODELS =====================
def process_and_train(data, columns_to_drop, param_grids, class_imb="original"):
    """
    Cleans the data, preprocesses it, and trains models based on the class imbalance method.

    Parameters:
    - data: The input dataset.
    - columns_to_drop: Columns to drop from the dataset.
    - param_grids: Parameter grids for GridSearchCV.
    - class_imb: Class imbalance handling method ("original", "oversampling", "undersampling").
    """
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

    # If class_imb is "original", test with both smote=True and smote=False
    smote_values = [True, False] if class_imb == "original" else [False]

    for smote in smote_values:
        print(f"\nTesting with SMOTE={smote} for class imbalance method: {class_imb.upper()}")

        # Preprocess the data
        pipeline_svm = data_preprocessing_svm(features, smote=smote)

        # Train and evaluate models
        for model_name, param_grid in param_grids.items():
            grid_search = grid_search_svm(pipeline_svm, param_grid)
            train_and_evaluate_model(
                f"{model_name}_smote"if class_imb == "original" and smote==True else f"{model_name}_{class_imb}",
                grid_search,
                X_train,
                y_train,
                X_test,
                y_test,
                unseen_features,
                unseen_labels,
                class_imb,  
                smote 
            )


# Process and train for each class imbalance method
class_imbalance_methods = ["original", "oversampling", "undersampling"]

for method in class_imbalance_methods:
    process_and_train(data_ksi, columns_to_drop, param_grids, class_imb=method)

# Save results to Markdown file & CSV
save_results_to_md("./insights/results/model_results.md")
save_results_to_csv("./insights/results/model_results.csv")