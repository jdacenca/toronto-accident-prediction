# Decision Tree Tuning Results (unseen_dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 91.07 | 93.40 | 0.7554 | 0.7865 | 0.7706 | original |
| DT_gini_original | 91.44 | 92.42 | 0.7178 | 0.7622 | 0.7393 | original |
| DT_entropy_original | 92.04 | 93.24 | 0.7555 | 0.7697 | 0.7625 | original |
| DT_weighted_original | 92.11 | 93.11 | 0.7610 | 0.7453 | 0.7531 | original |
| DT_basic_oversampling | 79.73 | 77.47 | 0.3704 | 0.8539 | 0.5167 | oversampling |
| DT_gini_oversampling | 79.61 | 79.03 | 0.3881 | 0.8446 | 0.5318 | oversampling |
| DT_entropy_oversampling | 80.06 | 80.11 | 0.4044 | 0.8670 | 0.5515 | oversampling |
| DT_weighted_oversampling | 79.90 | 77.47 | 0.3704 | 0.8539 | 0.5167 | oversampling |
| DT_basic_undersampling | 79.73 | 77.47 | 0.3704 | 0.8539 | 0.5167 | undersampling |
| DT_gini_undersampling | 79.61 | 79.03 | 0.3881 | 0.8446 | 0.5318 | undersampling |
| DT_entropy_undersampling | 80.06 | 80.11 | 0.4044 | 0.8670 | 0.5515 | undersampling |
| DT_weighted_undersampling | 79.90 | 77.47 | 0.3704 | 0.8539 | 0.5167 | undersampling |
| DT_basic_SMOTE | 92.33 | 90.46 | 0.6434 | 0.7266 | 0.6825 | SMOTE |
| DT_gini_SMOTE | 92.07 | 90.41 | 0.6423 | 0.7228 | 0.6802 | SMOTE |
| DT_entropy_SMOTE | 92.29 | 91.10 | 0.6695 | 0.7285 | 0.6978 | SMOTE |
| DT_weighted_SMOTE | 92.22 | 90.46 | 0.6434 | 0.7266 | 0.6825 | SMOTE |
