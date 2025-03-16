# Decision Tree Tuning Results (Main dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 100.00 | 93.74 | 0.7735 | 0.7865 | 0.7799 | original |
| DT_gini_original | 98.98 | 91.50 | 0.6693 | 0.7846 | 0.7224 | original |
| DT_entropy_original | 92.28 | 91.55 | 0.8615 | 0.4775 | 0.6145 | original |
| DT_weighted_original | 100.00 | 93.03 | 0.7419 | 0.7753 | 0.7582 | original |
| DT_basic_oversampling | 100.00 | 79.86 | 0.3998 | 0.8558 | 0.5450 | oversampling |
| DT_gini_oversampling | 98.43 | 79.22 | 0.3920 | 0.8596 | 0.5384 | oversampling |
| DT_entropy_oversampling | 81.55 | 73.28 | 0.3195 | 0.7921 | 0.4553 | oversampling |
| DT_weighted_oversampling | 100.00 | 79.86 | 0.3998 | 0.8558 | 0.5450 | oversampling |
| DT_basic_undersampling | 100.00 | 79.86 | 0.3998 | 0.8558 | 0.5450 | undersampling |
| DT_gini_undersampling | 98.43 | 79.22 | 0.3920 | 0.8596 | 0.5384 | undersampling |
| DT_entropy_undersampling | 81.55 | 73.28 | 0.3195 | 0.7921 | 0.4553 | undersampling |
| DT_weighted_undersampling | 100.00 | 79.86 | 0.3998 | 0.8558 | 0.5450 | undersampling |
| DT_basic_SMOTE | 100.00 | 89.07 | 0.5946 | 0.7060 | 0.6455 | SMOTE |
| DT_gini_SMOTE | 99.26 | 89.23 | 0.6050 | 0.6798 | 0.6402 | SMOTE |
| DT_entropy_SMOTE | 85.12 | 83.42 | 0.4386 | 0.6292 | 0.5169 | SMOTE |
| DT_weighted_SMOTE | 100.00 | 89.07 | 0.5946 | 0.7060 | 0.6455 | SMOTE |
