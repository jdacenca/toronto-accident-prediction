| Model | Kernel | Train Acc.% | Test Acc.% | Unseen Acc.% | Parameters | Precision | F1-Score | Recall | ROC Score | Class Imbalance | SMOTE |
|-------|--------|-------------|------------|--------------|------------|-----------|----------|--------|-----------|----------------|-------|
| svm_linear_smote | linear | 71.96 | 63.46 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.8063 | 0.6942 | 0.6346 | 0.5719 | original | True |
| svm_rbf_smote | rbf | 89.79 | 87.35 | 100.00 | {'svm__C': 0.01, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.7630 | 0.8145 | 0.8735 | 0.5000 | original | True |
| svm_poly_smote | poly | 100.00 | 78.54 | 90.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.3, 'svm__kernel': 'poly'} | 0.7762 | 0.7808 | 0.7854 | 0.4941 | original | True |
| svm_linear_original | linear | 87.44 | 87.35 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.7630 | 0.8145 | 0.8735 | 0.5000 | original | False |
| svm_rbf_original | rbf | 87.36 | 87.35 | 100.00 | {'svm__C': 0.01, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.7630 | 0.8145 | 0.8735 | 0.5000 | original | False |
| svm_poly_original | poly | 93.34 | 85.32 | 100.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.7726 | 0.8076 | 0.8532 | 0.4953 | original | False |
| svm_linear_oversampling | linear | 73.40 | 71.51 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.7162 | 0.7147 | 0.7151 | 0.7151 | oversampling | False |
| svm_rbf_oversampling | rbf | 100.00 | 99.88 | 100.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.9988 | 0.9988 | 0.9988 | 0.9988 | oversampling | False |
| svm_poly_oversampling | poly | 100.00 | 94.85 | 90.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.3, 'svm__kernel': 'poly'} | 0.9529 | 0.9483 | 0.9485 | 0.9484 | oversampling | False |
| svm_linear_undersampling | linear | 77.17 | 68.15 | 60.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.6816 | 0.6814 | 0.6815 | 0.6815 | undersampling | False |
| svm_rbf_undersampling | rbf | 78.38 | 66.13 | 90.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.6613 | 0.6613 | 0.6613 | 0.6613 | undersampling | False |
| svm_poly_undersampling | poly | 92.12 | 61.69 | 100.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.6171 | 0.6168 | 0.6169 | 0.6169 | undersampling | False |
