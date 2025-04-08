# Decision Tree Tuning Results 

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| basic | 90.93 | 94.01 | 0.7924 | 0.7790 | 0.7856 | original |
| gini | 90.93 | 94.01 | 0.7924 | 0.7790 | 0.7856 | original |
| entropy | 90.75 | 93.11 | 0.7468 | 0.7734 | 0.7599 | original |
| weighted | 91.01 | 94.85 | 0.8291 | 0.7996 | 0.8141 | original |
| basic oversampling | 74.75 | 76.94 | 0.3661 | 0.8708 | 0.5155 | oversampling |
| gini oversampling | 74.75 | 76.94 | 0.3661 | 0.8708 | 0.5155 | oversampling |
| entropy oversampling | 76.96 | 74.25 | 0.3331 | 0.8258 | 0.4747 | oversampling |
| weighted oversampling | 74.75 | 76.94 | 0.3661 | 0.8708 | 0.5155 | oversampling |
| basic undersampling | 74.75 | 76.94 | 0.3661 | 0.8708 | 0.5155 | undersampling |
| gini undersampling | 74.75 | 76.94 | 0.3661 | 0.8708 | 0.5155 | undersampling |
| entropy undersampling | 76.96 | 74.25 | 0.3331 | 0.8258 | 0.4747 | undersampling |
| weighted undersampling | 74.75 | 76.94 | 0.3661 | 0.8708 | 0.5155 | undersampling |
| basic SMOTE | 88.25 | 83.38 | 0.4302 | 0.5543 | 0.4845 | SMOTE |
| gini SMOTE | 88.25 | 83.38 | 0.4302 | 0.5543 | 0.4845 | SMOTE |
| entropy SMOTE | 88.11 | 83.32 | 0.4246 | 0.5169 | 0.4662 | SMOTE |
| weighted SMOTE | 88.18 | 83.38 | 0.4302 | 0.5543 | 0.4845 | SMOTE |
