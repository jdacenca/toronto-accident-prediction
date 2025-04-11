| Model | Kernel | Train Acc.% | Test Acc.% | Unseen Acc.% | Parameters | Precision | F1-Score | Recall | ROC Score | Class Imbalance | SMOTE |
|-------|--------|-------------|------------|--------------|------------|-----------|----------|--------|-----------|----------------|-------|
| svm_linear_smote | linear | 91.79 | 90.23 | 80.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9024 | 0.9023 | 0.9023 | 0.8959 | original | True |
| svm_rbf_smote | rbf | 95.68 | 91.03 | 90.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9105 | 0.9104 | 0.9103 | 0.9047 | original | True |
| svm_poly_smote | poly | 99.34 | 92.63 | 90.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.9261 | 0.9260 | 0.9263 | 0.9170 | original | True |
| svm_linear_original | linear | 92.21 | 89.72 | 90.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.8967 | 0.8965 | 0.8972 | 0.8840 | original | False |
| svm_rbf_original | rbf | 95.24 | 91.03 | 90.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9106 | 0.9093 | 0.9103 | 0.8940 | original | False |
| svm_poly_original | poly | 99.29 | 92.85 | 90.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.9284 | 0.9281 | 0.9285 | 0.9179 | original | False |
| svm_linear_oversampling | linear | 92.01 | 89.24 | 80.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.8924 | 0.8924 | 0.8924 | 0.8924 | oversampling | False |
| svm_rbf_oversampling | rbf | 96.24 | 93.17 | 90.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9320 | 0.9317 | 0.9317 | 0.9317 | oversampling | False |
| svm_poly_oversampling | poly | 99.46 | 94.68 | 90.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.9472 | 0.9467 | 0.9468 | 0.9468 | oversampling | False |
| svm_linear_undersampling | linear | 91.85 | 89.83 | 70.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.8984 | 0.8983 | 0.8983 | 0.8983 | undersampling | False |
| svm_rbf_undersampling | rbf | 95.01 | 90.72 | 70.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9072 | 0.9072 | 0.9072 | 0.9072 | undersampling | False |
| svm_poly_undersampling | poly | 99.21 | 91.21 | 60.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.9127 | 0.9121 | 0.9121 | 0.9122 | undersampling | False |
