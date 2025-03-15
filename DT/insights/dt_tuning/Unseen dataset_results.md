# Decision Tree Tuning Results (Unseen dataset)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| DT_basic_original | 100.00 | 93.56 | 0.7695 | 0.7753 | 0.7724 | original |
| DT_gini_original | 99.00 | 92.45 | 0.7095 | 0.7865 | 0.7460 | original |
| DT_entropy_original | 92.48 | 91.65 | 0.9291 | 0.4419 | 0.5990 | original |
| DT_weighted_original | 100.00 | 93.71 | 0.7772 | 0.7772 | 0.7772 | original |
| DT_basic_oversampling | 100.00 | 78.10 | 0.3780 | 0.8558 | 0.5244 | oversampling |
| DT_gini_oversampling | 98.27 | 79.40 | 0.3917 | 0.8333 | 0.5329 | oversampling |
| DT_entropy_oversampling | 81.96 | 78.05 | 0.3644 | 0.7472 | 0.4899 | oversampling |
| DT_weighted_oversampling | 100.00 | 78.10 | 0.3780 | 0.8558 | 0.5244 | oversampling |
| DT_basic_undersampling | 100.00 | 78.10 | 0.3780 | 0.8558 | 0.5244 | undersampling |
| DT_gini_undersampling | 98.27 | 79.40 | 0.3917 | 0.8333 | 0.5329 | undersampling |
| DT_entropy_undersampling | 81.96 | 78.05 | 0.3644 | 0.7472 | 0.4899 | undersampling |
| DT_weighted_undersampling | 100.00 | 78.10 | 0.3780 | 0.8558 | 0.5244 | undersampling |
| DT_basic_SMOTE | 100.00 | 89.96 | 0.6266 | 0.7135 | 0.6673 | SMOTE |
| DT_gini_SMOTE | 99.36 | 90.20 | 0.6432 | 0.6854 | 0.6636 | SMOTE |
| DT_entropy_SMOTE | 84.10 | 79.95 | 0.3760 | 0.6386 | 0.4733 | SMOTE |
| DT_weighted_SMOTE | 100.00 | 89.96 | 0.6266 | 0.7135 | 0.6673 | SMOTE |
