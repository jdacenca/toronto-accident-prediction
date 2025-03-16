# Decision Tree Tuning Results (main_dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 91.14 | 92.95 | 0.7298 | 0.7940 | 0.7605 | original |
| DT_gini_original | 90.66 | 92.53 | 0.7153 | 0.7809 | 0.7466 | original |
| DT_entropy_original | 91.77 | 93.66 | 0.7682 | 0.7884 | 0.7782 | original |
| DT_weighted_original | 91.47 | 93.69 | 0.7726 | 0.7828 | 0.7777 | original |
| DT_basic_oversampling | 78.04 | 79.33 | 0.3946 | 0.8727 | 0.5434 | oversampling |
| DT_gini_oversampling | 77.93 | 79.78 | 0.3997 | 0.8652 | 0.5467 | oversampling |
| DT_entropy_oversampling | 77.88 | 79.59 | 0.3962 | 0.8539 | 0.5412 | oversampling |
| DT_weighted_oversampling | 77.93 | 79.33 | 0.3946 | 0.8727 | 0.5434 | oversampling |
| DT_basic_undersampling | 78.04 | 79.33 | 0.3946 | 0.8727 | 0.5434 | undersampling |
| DT_gini_undersampling | 77.93 | 79.78 | 0.3997 | 0.8652 | 0.5467 | undersampling |
| DT_entropy_undersampling | 77.88 | 79.59 | 0.3962 | 0.8539 | 0.5412 | undersampling |
| DT_weighted_undersampling | 77.93 | 79.33 | 0.3946 | 0.8727 | 0.5434 | undersampling |
| DT_basic_SMOTE | 91.79 | 88.33 | 0.5719 | 0.6854 | 0.6235 | SMOTE |
| DT_gini_SMOTE | 91.72 | 88.52 | 0.5815 | 0.6610 | 0.6188 | SMOTE |
| DT_entropy_SMOTE | 92.18 | 89.57 | 0.6213 | 0.6667 | 0.6432 | SMOTE |
| DT_weighted_SMOTE | 91.79 | 88.33 | 0.5719 | 0.6854 | 0.6235 | SMOTE |
