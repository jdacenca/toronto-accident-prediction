| Classifier | Voting Type | Class Imbalance | Train Accuracy | Test Accuracy | Unseen Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------|-------------|-----------------|----------------|---------------|-----------------|-----------|--------|----------|---------|
| LogisticRegression(max_iter=1400) | Hard | oversampling | 0.9942 | 0.9832 | 1.0000 | 0.9832 | 0.9832 | 0.9832 | 0.9832 |
| RandomForestClassifier(class_weight='balanced', n_estimators=1000, n_jobs=-1,
                       random_state=37) | Hard | oversampling | 1.0000 | 0.9890 | 1.0000 | 0.9890 | 0.9890 | 0.9890 | 0.9890 |
| SVC(C=1, kernel='linear') | Hard | oversampling | 0.9997 | 0.9907 | 1.0000 | 0.9908 | 0.9907 | 0.9907 | 0.9907 |
| DecisionTreeClassifier(criterion='entropy', max_depth=42) | Hard | oversampling | 1.0000 | 0.9861 | 1.0000 | 0.9861 | 0.9861 | 0.9861 | 0.9861 |
| MLPClassifier(activation='tanh', alpha=0.01, hidden_layer_sizes=(15, 10, 1),
              learning_rate='invscaling', max_iter=1000) | Hard | oversampling | 0.9999 | 0.9878 | 1.0000 | 0.9878 | 0.9878 | 0.9878 | 0.9878 |
| VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1400)),
                             ('rf',
                              RandomForestClassifier(class_weight='balanced',
                                                     n_estimators=1000,
                                                     n_jobs=-1,
                                                     random_state=37)),
                             ('svm', SVC(C=1, kernel='linear')),
                             ('dt',
                              DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=42)),
                             ('nn',
                              MLPClassifier(activation='tanh', alpha=0.01,
                                            hidden_layer_sizes=(15, 10, 1),
                                            learning_rate='invscaling',
                                            max_iter=1000))]) | Hard | oversampling | 1.0000 | 0.9907 | 1.0000 | 0.9908 | 0.9907 | 0.9907 | 0.9907 |
| LogisticRegression(max_iter=1400) | Soft | oversampling | 0.9942 | 0.9832 | 1.0000 | 0.9832 | 0.9832 | 0.9832 | 0.9832 |
| RandomForestClassifier(class_weight='balanced', n_estimators=1000, n_jobs=-1,
                       random_state=37) | Soft | oversampling | 1.0000 | 0.9890 | 1.0000 | 0.9890 | 0.9890 | 0.9890 | 0.9890 |
| SVC(C=1, kernel='linear', probability=True) | Soft | oversampling | 0.9997 | 0.9907 | 1.0000 | 0.9908 | 0.9907 | 0.9907 | 0.9907 |
| DecisionTreeClassifier(criterion='entropy', max_depth=42) | Soft | oversampling | 1.0000 | 0.9867 | 1.0000 | 0.9867 | 0.9867 | 0.9867 | 0.9867 |
| MLPClassifier(activation='tanh', alpha=0.01, hidden_layer_sizes=(15, 10, 1),
              learning_rate='invscaling', max_iter=1000) | Soft | oversampling | 0.9999 | 0.9878 | 1.0000 | 0.9879 | 0.9878 | 0.9878 | 0.9878 |
| VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1400)),
                             ('rf',
                              RandomForestClassifier(class_weight='balanced',
                                                     n_estimators=1000,
                                                     n_jobs=-1,
                                                     random_state=37)),
                             ('svm',
                              SVC(C=1, kernel='linear', probability=True)),
                             ('dt',
                              DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=42)),
                             ('nn',
                              MLPClassifier(activation='tanh', alpha=0.01,
                                            hidden_layer_sizes=(15, 10, 1),
                                            learning_rate='invscaling',
                                            max_iter=1000))],
                 voting='soft') | Soft | oversampling | 1.0000 | 0.9913 | 1.0000 | 0.9913 | 0.9913 | 0.9913 | 0.9913 |
| LogisticRegression(max_iter=1400) | Hard | undersampling | 0.9935 | 0.9787 | 1.0000 | 0.9790 | 0.9787 | 0.9787 | 0.9787 |
| RandomForestClassifier(class_weight='balanced', n_estimators=1000, n_jobs=-1,
                       random_state=37) | Hard | undersampling | 1.0000 | 0.9845 | 1.0000 | 0.9845 | 0.9845 | 0.9845 | 0.9845 |
| SVC(C=1, kernel='linear') | Hard | undersampling | 0.9998 | 0.9729 | 1.0000 | 0.9729 | 0.9729 | 0.9729 | 0.9729 |
| DecisionTreeClassifier(criterion='entropy', max_depth=42) | Hard | undersampling | 1.0000 | 0.9719 | 1.0000 | 0.9719 | 0.9719 | 0.9719 | 0.9719 |
| MLPClassifier(activation='tanh', alpha=0.01, hidden_layer_sizes=(15, 10, 1),
              learning_rate='invscaling', max_iter=1000) | Hard | undersampling | 1.0000 | 0.9748 | 0.9000 | 0.9753 | 0.9748 | 0.9748 | 0.9748 |
| VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1400)),
                             ('rf',
                              RandomForestClassifier(class_weight='balanced',
                                                     n_estimators=1000,
                                                     n_jobs=-1,
                                                     random_state=37)),
                             ('svm', SVC(C=1, kernel='linear')),
                             ('dt',
                              DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=42)),
                             ('nn',
                              MLPClassifier(activation='tanh', alpha=0.01,
                                            hidden_layer_sizes=(15, 10, 1),
                                            learning_rate='invscaling',
                                            max_iter=1000))]) | Hard | undersampling | 1.0000 | 0.9806 | 1.0000 | 0.9808 | 0.9806 | 0.9806 | 0.9806 |
| LogisticRegression(max_iter=1400) | Soft | undersampling | 0.9935 | 0.9787 | 1.0000 | 0.9790 | 0.9787 | 0.9787 | 0.9787 |
| RandomForestClassifier(class_weight='balanced', n_estimators=1000, n_jobs=-1,
                       random_state=37) | Soft | undersampling | 1.0000 | 0.9845 | 1.0000 | 0.9845 | 0.9845 | 0.9845 | 0.9845 |
| SVC(C=1, kernel='linear', probability=True) | Soft | undersampling | 0.9998 | 0.9729 | 1.0000 | 0.9729 | 0.9729 | 0.9729 | 0.9729 |
| DecisionTreeClassifier(criterion='entropy', max_depth=42) | Soft | undersampling | 1.0000 | 0.9700 | 1.0000 | 0.9700 | 0.9700 | 0.9700 | 0.9700 |
| MLPClassifier(activation='tanh', alpha=0.01, hidden_layer_sizes=(15, 10, 1),
              learning_rate='invscaling', max_iter=1000) | Soft | undersampling | 0.9993 | 0.9739 | 1.0000 | 0.9739 | 0.9739 | 0.9739 | 0.9739 |
| VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1400)),
                             ('rf',
                              RandomForestClassifier(class_weight='balanced',
                                                     n_estimators=1000,
                                                     n_jobs=-1,
                                                     random_state=37)),
                             ('svm',
                              SVC(C=1, kernel='linear', probability=True)),
                             ('dt',
                              DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=42)),
                             ('nn',
                              MLPClassifier(activation='tanh', alpha=0.01,
                                            hidden_layer_sizes=(15, 10, 1),
                                            learning_rate='invscaling',
                                            max_iter=1000))],
                 voting='soft') | Soft | undersampling | 1.0000 | 0.9806 | 1.0000 | 0.9808 | 0.9806 | 0.9806 | 0.9806 |
