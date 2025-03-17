# Decision Tree Tuning Results 

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 91.14 | 92.95 | 0.7298 | 0.7940 | 0.7605 | original |
| DT_gini_original | 91.14 | 92.95 | 0.7298 | 0.7940 | 0.7605 | original |
| DT_entropy_original | 91.94 | 93.64 | 0.7584 | 0.8052 | 0.7811 | original |
| DT_weighted_original | 91.47 | 93.69 | 0.7726 | 0.7828 | 0.7777 | original |
| DT_basic_oversampling | 78.04 | 79.33 | 0.3946 | 0.8727 | 0.5434 | oversampling |
| DT_gini_oversampling | 78.04 | 79.33 | 0.3946 | 0.8727 | 0.5434 | oversampling |
| DT_entropy_oversampling | 78.18 | 79.12 | 0.3908 | 0.8614 | 0.5377 | oversampling |
| DT_weighted_oversampling | 77.93 | 79.33 | 0.3946 | 0.8727 | 0.5434 | oversampling |
| DT_basic_undersampling | 78.04 | 79.33 | 0.3946 | 0.8727 | 0.5434 | undersampling |
| DT_gini_undersampling | 78.04 | 79.33 | 0.3946 | 0.8727 | 0.5434 | undersampling |
| DT_entropy_undersampling | 78.18 | 79.12 | 0.3908 | 0.8614 | 0.5377 | undersampling |
| DT_weighted_undersampling | 77.93 | 79.33 | 0.3946 | 0.8727 | 0.5434 | undersampling |
| DT_basic_SMOTE | 91.79 | 88.33 | 0.5719 | 0.6854 | 0.6235 | SMOTE |
| DT_gini_SMOTE | 91.79 | 88.33 | 0.5719 | 0.6854 | 0.6235 | SMOTE |
| DT_entropy_SMOTE | 92.26 | 89.65 | 0.6199 | 0.6873 | 0.6519 | SMOTE |
| DT_weighted_SMOTE | 91.79 | 88.33 | 0.5719 | 0.6854 | 0.6235 | SMOTE |
