| Model | Kernel | Train Acc.% | Test Acc.% | Unseen Acc.% | Parameters | Precision | F1-Score | Recall | ROC Score | Class Imbalance | SMOTE |
|-------|--------|-------------|------------|--------------|------------|-----------|----------|--------|-----------|----------------|-------|
| svm_linear_smote | linear | 97.90 | 97.83 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9785 | 0.9783 | 0.9783 | 0.9787 | original | True |
| svm_rbf_smote | rbf | 99.24 | 98.41 | 100.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9841 | 0.9841 | 0.9841 | 0.9838 | original | True |
| svm_poly_smote | poly | 99.96 | 98.33 | 100.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.9834 | 0.9834 | 0.9833 | 0.9832 | original | True |
| svm_linear_original | linear | 98.01 | 97.83 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9783 | 0.9783 | 0.9783 | 0.9768 | original | False |
| svm_rbf_original | rbf | 99.04 | 98.41 | 100.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9841 | 0.9841 | 0.9841 | 0.9830 | original | False |
| svm_poly_original | poly | 99.95 | 98.55 | 100.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.9855 | 0.9855 | 0.9855 | 0.9849 | original | False |
| svm_linear_oversampling | linear | 98.03 | 97.51 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9752 | 0.9751 | 0.9751 | 0.9751 | oversampling | False |
| svm_rbf_oversampling | rbf | 99.32 | 98.61 | 100.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9862 | 0.9861 | 0.9861 | 0.9861 | oversampling | False |
| svm_poly_oversampling | poly | 100.00 | 99.02 | 100.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.3, 'svm__kernel': 'poly'} | 0.9902 | 0.9902 | 0.9902 | 0.9902 | oversampling | False |
| svm_linear_undersampling | linear | 97.80 | 97.00 | 100.00 | {'svm__C': 1, 'svm__kernel': 'linear'} | 0.9703 | 0.9700 | 0.9700 | 0.9700 | undersampling | False |
| svm_rbf_undersampling | rbf | 98.89 | 98.26 | 100.00 | {'svm__C': 1, 'svm__gamma': 0.03, 'svm__kernel': 'rbf'} | 0.9826 | 0.9826 | 0.9826 | 0.9826 | undersampling | False |
| svm_poly_undersampling | poly | 99.95 | 98.26 | 100.00 | {'svm__C': 0.1, 'svm__degree': 3, 'svm__gamma': 0.1, 'svm__kernel': 'poly'} | 0.9826 | 0.9826 | 0.9826 | 0.9826 | undersampling | False |
