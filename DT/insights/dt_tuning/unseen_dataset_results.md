# Decision Tree Tuning Results (unseen_dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 89.72 | 91.18 | 0.6748 | 0.7228 | 0.6980 | original |
| DT_gini_original | 90.02 | 91.31 | 0.6711 | 0.7528 | 0.7096 | original |
| DT_entropy_original | 90.99 | 93.16 | 0.7599 | 0.7528 | 0.7563 | original |
| DT_weighted_original | 90.70 | 92.29 | 0.7224 | 0.7360 | 0.7291 | original |
| DT_basic_oversampling | 78.19 | 77.84 | 0.3726 | 0.8352 | 0.5153 | oversampling |
| DT_gini_oversampling | 78.09 | 78.53 | 0.3796 | 0.8240 | 0.5198 | oversampling |
| DT_entropy_oversampling | 79.01 | 80.80 | 0.4128 | 0.8558 | 0.5570 | oversampling |
| DT_weighted_oversampling | 77.97 | 77.84 | 0.3726 | 0.8352 | 0.5153 | oversampling |
| DT_basic_undersampling | 78.19 | 77.84 | 0.3726 | 0.8352 | 0.5153 | undersampling |
| DT_gini_undersampling | 78.09 | 78.53 | 0.3796 | 0.8240 | 0.5198 | undersampling |
| DT_entropy_undersampling | 79.01 | 80.80 | 0.4128 | 0.8558 | 0.5570 | undersampling |
| DT_weighted_undersampling | 77.97 | 77.84 | 0.3726 | 0.8352 | 0.5153 | undersampling |
| DT_basic_SMOTE | 92.50 | 90.15 | 0.6367 | 0.7022 | 0.6679 | SMOTE |
| DT_gini_SMOTE | 92.42 | 90.12 | 0.6434 | 0.6723 | 0.6575 | SMOTE |
| DT_entropy_SMOTE | 92.61 | 90.54 | 0.6571 | 0.6891 | 0.6728 | SMOTE |
| DT_weighted_SMOTE | 92.49 | 90.15 | 0.6367 | 0.7022 | 0.6679 | SMOTE |
