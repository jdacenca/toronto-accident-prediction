# Decision Tree Tuning Results 

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| basic | 91.66 | 94.77 | 0.8051 | 0.8293 | 0.8170 | original |
| gini | 91.66 | 94.77 | 0.8051 | 0.8293 | 0.8170 | original |
| entropy | 91.22 | 93.42 | 0.7591 | 0.7805 | 0.7697 | original |
| basic smote | 93.29 | 91.26 | 0.6629 | 0.7711 | 0.7129 | smote |
| gini smote | 93.29 | 91.26 | 0.6629 | 0.7711 | 0.7129 | smote |
| entropy smote | 92.50 | 90.68 | 0.6490 | 0.7355 | 0.6895 | smote |
| basic random_over | 97.32 | 94.22 | 0.7844 | 0.8124 | 0.7982 | random_over |
| gini random_over | 97.32 | 94.22 | 0.7844 | 0.8124 | 0.7982 | random_over |
| entropy random_over | 97.21 | 95.27 | 0.8278 | 0.8386 | 0.8332 | random_over |
| basic random_under | 73.84 | 72.74 | 0.3161 | 0.8049 | 0.4540 | random_under |
| gini random_under | 73.84 | 72.74 | 0.3161 | 0.8049 | 0.4540 | random_under |
| entropy random_under | 71.89 | 72.53 | 0.3188 | 0.8368 | 0.4617 | random_under |
| basic smote_tomek | 93.30 | 91.63 | 0.6714 | 0.7936 | 0.7274 | smote_tomek |
| gini smote_tomek | 93.30 | 91.63 | 0.6714 | 0.7936 | 0.7274 | smote_tomek |
| entropy smote_tomek | 92.38 | 90.60 | 0.6463 | 0.7336 | 0.6872 | smote_tomek |
| basic smote_enn | 91.93 | 84.44 | 0.4680 | 0.7692 | 0.5820 | smote_enn |
| gini smote_enn | 91.93 | 84.44 | 0.4680 | 0.7692 | 0.5820 | smote_enn |
| entropy smote_enn | 92.44 | 83.89 | 0.4551 | 0.7317 | 0.5612 | smote_enn |
