| Model | Kernel | Train Acc.% | Test Acc.% | Unseen Acc.% | Parameters | Precision | F1-Score | Recall | ROC Score | Class Imbalance | SMOTE |
|-------|--------|-------------|------------|--------------|------------|-----------|----------|--------|-----------|----------------|-------|
| svm_linear_smote | linear | 99.97 | 98.18 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9816 | 0.9816 | 0.9818 | 0.9519 | original | True |
| svm_rbf_smote | rbf | 100.00 | 87.34 | 100.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.7627 | 0.8143 | 0.8734 | 0.5000 | original | True |
| svm_poly_smote| poly | 100.00 | 98.48 | 100.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.9847 | 0.9846 | 0.9848 | 0.9503 | original | True |
| svm_linear_original| linear | 99.92 | 98.18 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9816 | 0.9816 | 0.9818 | 0.9519 | original | False |
| svm_rbf_original | rbf | 100.00 | 87.34 | 100.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.7627 | 0.8143 | 0.8734 | 0.5000 | original | False |
| svm_poly_original | poly | 100.00 | 98.28 | 100.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.9827 | 0.9824 | 0.9828 | 0.9423 | original | False |
| svm_linear_oversampling | linear | 99.99 | 99.54 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9954 | 0.9954 | 0.9954 | 0.9954 | oversampling | False |
| svm_rbf_oversampling| rbf | 100.00 | 99.65 | 100.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.9965 | 0.9965 | 0.9965 | 0.9965 | oversampling | False |
| svm_poly_oversampling | poly | 100.00 | 99.94 | 100.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.9994 | 0.9994 | 0.9994 | 0.9994 | oversampling | False |
| svm_linear_undersampling | linear | 100.00 | 97.58 | 90.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9759 | 0.9758 | 0.9758 | 0.9758 | undersampling | False |
| svm_rbf_undersampling| rbf | 100.00 | 49.60 | 50.00 | {'svm__C': 1, 'svm__gamma': 3.0, 'svm__kernel': 'rbf'} | 0.2490 | 0.3315 | 0.4960 | 0.4960 | undersampling | False |
| svm_poly_undersampling | poly | 100.00 | 96.37 | 100.00 | {'svm__C': 10, 'svm__degree': 3, 'svm__gamma': 1.0, 'svm__kernel': 'poly'} | 0.9640 | 0.9637 | 0.9637 | 0.9637 | undersampling | False |
