| Classifier | Class Imbalance | Train Accuracy | Test Accuracy | Unseen Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------|-----------------|----------------|---------------|-----------------|-----------|--------|----------|---------|
| LogisticRegression |  oversampling | 0.9162 | 0.8958 | 0.8000 | 0.8958 | 0.8958 | 0.8958 | 0.9630 |
| RandomForestClassifier |  oversampling | 1.0000 | 0.9410 | 0.9000 | 0.9413 | 0.9410 | 0.9410 | 0.9882 |
| SVC |  oversampling | 0.9954 | 0.9479 | 0.9000 | 0.9482 | 0.9479 | 0.9479 | 0.9837 |
| DecisionTreeClassifier |  oversampling | 1.0000 | 0.9126 | 0.7000 | 0.9142 | 0.9126 | 0.9125 | 0.9126 |
| MLPClassifier |  oversampling | 0.9768 | 0.9236 | 0.9000 | 0.9255 | 0.9236 | 0.9235 | 0.9624 |
| hardVotingClassifier |  oversampling | 0.9971 | 0.9416 | 0.8000 | 0.9422 | 0.9416 | 0.9415 | 0.9416 |
| softVotingClassifier |  oversampling | 0.9971 | 0.9404 | 0.8000 | 0.9413 | 0.9404 | 0.9404 | 0.9889 |
| LogisticRegression |  undersampling | 0.9151 | 0.8973 | 0.8000 | 0.8973 | 0.8973 | 0.8973 | 0.9578 |
| RandomForestClassifier |  undersampling | 1.0000 | 0.9062 | 0.7000 | 0.9062 | 0.9062 | 0.9062 | 0.9728 |
| SVC |  undersampling | 0.9931 | 0.9131 | 0.6000 | 0.9138 | 0.9131 | 0.9131 | 0.9651 |
| DecisionTreeClassifier |  undersampling | 1.0000 | 0.8697 | 0.9000 | 0.8698 | 0.8697 | 0.8697 | 0.8697 |
| MLPClassifier |  undersampling | 0.9736 | 0.8825 | 0.7000 | 0.8826 | 0.8825 | 0.8825 | 0.9522 |
| hardVotingClassifier |  undersampling | 0.9951 | 0.9181 | 0.7000 | 0.9183 | 0.9181 | 0.9181 | 0.9181 |
| softVotingClassifier |  undersampling | 0.9968 | 0.9102 | 0.7000 | 0.9102 | 0.9102 | 0.9102 | 0.9737 |
