# Decision Tree Tuning Results 

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| basic | 90.86 | 93.83 | 0.7820 | 0.7790 | 0.7805 | original |
| gini | 90.86 | 93.83 | 0.7820 | 0.7790 | 0.7805 | original |
| entropy | 92.26 | 93.88 | 0.7706 | 0.8052 | 0.7875 | original |
| weighted | 91.86 | 93.78 | 0.7780 | 0.7809 | 0.7794 | original |
| basic oversampling | 81.16 | 81.41 | 0.4235 | 0.8858 | 0.5730 | original |
| gini oversampling | 81.16 | 81.41 | 0.4235 | 0.8858 | 0.5730 | original |
| entropy oversampling | 80.36 | 81.22 | 0.4186 | 0.8577 | 0.5627 | original |
| weighted oversampling | 80.99 | 81.41 | 0.4235 | 0.8858 | 0.5730 | original |
| basic undersampling | 81.16 | 81.41 | 0.4235 | 0.8858 | 0.5730 | original |
| gini undersampling | 81.16 | 81.41 | 0.4235 | 0.8858 | 0.5730 | original |
| entropy undersampling | 80.36 | 81.22 | 0.4186 | 0.8577 | 0.5627 | original |
| weighted undersampling | 80.99 | 81.41 | 0.4235 | 0.8858 | 0.5730 | original |
| basic SMOTE | 92.37 | 88.87 | 0.5909 | 0.6816 | 0.6330 | original |
| gini SMOTE | 92.37 | 88.87 | 0.5909 | 0.6816 | 0.6330 | original |
| entropy SMOTE | 93.13 | 89.56 | 0.6186 | 0.6742 | 0.6452 | original |
| weighted SMOTE | 92.31 | 88.87 | 0.5909 | 0.6816 | 0.6330 | original |
