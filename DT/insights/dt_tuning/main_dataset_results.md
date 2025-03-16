# Decision Tree Tuning Results (main_dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 89.83 | 91.87 | 0.6989 | 0.7434 | 0.7205 | original |
| DT_gini_original | 89.90 | 92.61 | 0.7220 | 0.7734 | 0.7468 | original |
| DT_entropy_original | 91.00 | 92.98 | 0.7384 | 0.7772 | 0.7573 | original |
| DT_weighted_original | 90.84 | 93.32 | 0.7656 | 0.7584 | 0.7620 | original |
| DT_basic_oversampling | 79.17 | 77.80 | 0.3724 | 0.8390 | 0.5158 | oversampling |
| DT_gini_oversampling | 78.82 | 77.69 | 0.3694 | 0.8240 | 0.5101 | oversampling |
| DT_entropy_oversampling | 77.74 | 79.86 | 0.3988 | 0.8446 | 0.5417 | oversampling |
| DT_weighted_oversampling | 78.98 | 77.80 | 0.3724 | 0.8390 | 0.5158 | oversampling |
| DT_basic_undersampling | 79.17 | 77.80 | 0.3724 | 0.8390 | 0.5158 | undersampling |
| DT_gini_undersampling | 78.82 | 77.69 | 0.3694 | 0.8240 | 0.5101 | undersampling |
| DT_entropy_undersampling | 77.74 | 79.86 | 0.3988 | 0.8446 | 0.5417 | undersampling |
| DT_weighted_undersampling | 78.98 | 77.80 | 0.3724 | 0.8390 | 0.5158 | undersampling |
| DT_basic_SMOTE | 92.25 | 88.07 | 0.5659 | 0.6592 | 0.6090 | SMOTE |
| DT_gini_SMOTE | 92.05 | 88.60 | 0.5870 | 0.6442 | 0.6143 | SMOTE |
| DT_entropy_SMOTE | 92.43 | 88.89 | 0.5963 | 0.6554 | 0.6244 | SMOTE |
| DT_weighted_SMOTE | 92.25 | 88.07 | 0.5659 | 0.6592 | 0.6090 | SMOTE |
