# Decision Tree Tuning Results (unseen_dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 90.77 | 93.53 | 0.7711 | 0.7697 | 0.7704 | original |
| DT_gini_original | 90.11 | 92.58 | 0.7148 | 0.7884 | 0.7498 | original |
| DT_entropy_original | 91.40 | 92.79 | 0.7421 | 0.7491 | 0.7456 | original |
| DT_weighted_original | 90.89 | 93.66 | 0.7784 | 0.7697 | 0.7740 | original |
| DT_basic_oversampling | 78.44 | 78.10 | 0.3780 | 0.8558 | 0.5244 | oversampling |
| DT_gini_oversampling | 78.42 | 79.40 | 0.3917 | 0.8333 | 0.5329 | oversampling |
| DT_entropy_oversampling | 78.89 | 79.95 | 0.4005 | 0.8483 | 0.5441 | oversampling |
| DT_weighted_oversampling | 78.63 | 78.10 | 0.3780 | 0.8558 | 0.5244 | oversampling |
| DT_basic_undersampling | 78.44 | 78.10 | 0.3780 | 0.8558 | 0.5244 | undersampling |
| DT_gini_undersampling | 78.42 | 79.40 | 0.3917 | 0.8333 | 0.5329 | undersampling |
| DT_entropy_undersampling | 78.89 | 79.95 | 0.4005 | 0.8483 | 0.5441 | undersampling |
| DT_weighted_undersampling | 78.63 | 78.10 | 0.3780 | 0.8558 | 0.5244 | undersampling |
| DT_basic_SMOTE | 92.61 | 90.10 | 0.6314 | 0.7154 | 0.6708 | SMOTE |
| DT_gini_SMOTE | 92.42 | 90.17 | 0.6406 | 0.6910 | 0.6649 | SMOTE |
| DT_entropy_SMOTE | 92.04 | 89.67 | 0.6190 | 0.6966 | 0.6555 | SMOTE |
| DT_weighted_SMOTE | 92.69 | 90.10 | 0.6314 | 0.7154 | 0.6708 | SMOTE |
