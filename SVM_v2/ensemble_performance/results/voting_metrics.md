| Classifier | Class Imbalance | Train Accuracy | Test Accuracy | Unseen Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------|-----------------|----------------|---------------|-----------------|-----------|--------|----------|---------|
| LogisticRegression |  oversampling | 0.9162 | 0.8958 | 0.8000 | 0.8958 | 0.8958 | 0.8958 | 0.9630 |
| RandomForestClassifier |  oversampling | 1.0000 | 0.9410 | 0.9000 | 0.9413 | 0.9410 | 0.9410 | 0.9882 |
| SVC |  oversampling | 0.9954 | 0.9479 | 0.9000 | 0.9482 | 0.9479 | 0.9479 | 0.9837 |
| DecisionTreeClassifier |  oversampling | 1.0000 | 0.9126 | 0.7000 | 0.9142 | 0.9126 | 0.9125 | 0.9126 |
| MLPClassifier |  oversampling | 0.9768 | 0.9236 | 0.9000 | 0.9255 | 0.9236 | 0.9235 | 0.9624 |
| hardVotingClassifier |  oversampling | 0.9971 | 0.9416 | 0.8000 | 0.9422 | 0.9416 | 0.9415 | 0.9416 |
| softVotingClassifier |  oversampling | 0.9971 | 0.9404 | 0.8000 | 0.9413 | 0.9404 | 0.9404 | 0.9889 |
