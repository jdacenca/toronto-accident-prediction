# Decision Tree Tuning Results (main_dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 90.27 | 93.74 | 0.7735 | 0.7865 | 0.7799 | original |
| DT_gini_original | 90.16 | 91.50 | 0.6710 | 0.7790 | 0.7210 | original |
| DT_entropy_original | 90.69 | 93.72 | 0.7824 | 0.7678 | 0.7750 | original |
| DT_weighted_original | 91.15 | 92.93 | 0.7401 | 0.7678 | 0.7537 | original |
| DT_basic_oversampling | 78.46 | 79.86 | 0.3998 | 0.8558 | 0.5450 | oversampling |
| DT_gini_oversampling | 77.60 | 79.22 | 0.3920 | 0.8596 | 0.5384 | oversampling |
| DT_entropy_oversampling | 79.35 | 79.91 | 0.4014 | 0.8652 | 0.5484 | oversampling |
| DT_weighted_oversampling | 78.25 | 79.86 | 0.3998 | 0.8558 | 0.5450 | oversampling |
| DT_basic_undersampling | 78.46 | 79.86 | 0.3998 | 0.8558 | 0.5450 | undersampling |
| DT_gini_undersampling | 77.60 | 79.22 | 0.3920 | 0.8596 | 0.5384 | undersampling |
| DT_entropy_undersampling | 79.35 | 79.91 | 0.4014 | 0.8652 | 0.5484 | undersampling |
| DT_weighted_undersampling | 78.25 | 79.86 | 0.3998 | 0.8558 | 0.5450 | undersampling |
| DT_basic_SMOTE | 91.99 | 89.07 | 0.5946 | 0.7060 | 0.6455 | SMOTE |
| DT_gini_SMOTE | 91.64 | 89.23 | 0.6050 | 0.6798 | 0.6402 | SMOTE |
| DT_entropy_SMOTE | 92.08 | 88.54 | 0.5812 | 0.6704 | 0.6226 | SMOTE |
| DT_weighted_SMOTE | 91.99 | 89.07 | 0.5946 | 0.7060 | 0.6455 | SMOTE |
