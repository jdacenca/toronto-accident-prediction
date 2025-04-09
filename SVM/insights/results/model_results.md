| Model | Kernel | Train Acc.% | Test Acc.% | Unseen Acc.% | Parameters | Precision | F1-Score | Recall | ROC Score | Class Imbalance | SMOTE |
|-------|--------|-------------|------------|--------------|------------|-----------|----------|--------|-----------|----------------|-------|
| svm_linear_smote | linear | 71.73 | 63.97 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.8073 | 0.6982 | 0.6397 | 0.5748 | original | True |
| svm_rbf_smote | rbf | 100.00 | 87.25 | 100.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.7629 | 0.8140 | 0.8725 | 0.4994 | original | True |
| svm_poly_smote | poly | 100.00 | 80.26 | 90.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.7920 | 0.7972 | 0.8026 | 0.5279 | original | True |
| svm_linear_original | linear | 87.44 | 87.35 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.7630 | 0.8145 | 0.8735 | 0.5000 | original | False |
| svm_rbf_original | rbf | 100.00 | 87.35 | 100.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.7630 | 0.8145 | 0.8735 | 0.5000 | original | False |
| svm_poly_original | poly | 100.00 | 81.58 | 90.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.7919 | 0.8031 | 0.8158 | 0.5251 | original | False |
| svm_linear_oversampling | linear | 73.33 | 71.22 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.7136 | 0.7117 | 0.7122 | 0.7122 | oversampling | False |
| svm_rbf_oversampling | rbf | 100.00 | 99.88 | 100.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.9988 | 0.9988 | 0.9988 | 0.9988 | oversampling | False |
| svm_poly_oversampling | poly | 100.00 | 94.09 | 90.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.9467 | 0.9407 | 0.9409 | 0.9409 | oversampling | False |
| svm_linear_undersampling | linear | 77.17 | 68.15 | 50.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.6815 | 0.6814 | 0.6815 | 0.6815 | undersampling | False |
| svm_rbf_undersampling | rbf | 100.00 | 50.40 | 40.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.7510 | 0.3422 | 0.5040 | 0.5040 | undersampling | False |
| svm_poly_undersampling | poly | 100.00 | 60.89 | 50.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.6092 | 0.6086 | 0.6089 | 0.6089 | undersampling | False |
