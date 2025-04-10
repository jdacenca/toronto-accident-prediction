# Decision Tree Tuning Results 

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Sampling |
|-------|------------|-----------|-----------|---------|-----------|----------|
| basic | 90.37 | 93.45 | 0.7698 | 0.7640 | 0.7669 | original |
| gini | 90.37 | 93.45 | 0.7698 | 0.7640 | 0.7669 | original |
| entropy | 90.56 | 93.58 | 0.7730 | 0.7715 | 0.7723 | original |
| basic smote | 91.44 | 93.16 | 0.7229 | 0.8352 | 0.7750 | smote |
| gini smote | 91.44 | 93.16 | 0.7229 | 0.8352 | 0.7750 | smote |
| entropy smote | 91.72 | 93.26 | 0.7204 | 0.8539 | 0.7815 | smote |
| basic random_over | 97.29 | 94.29 | 0.7989 | 0.7959 | 0.7974 | random_over |
| gini random_over | 97.29 | 94.29 | 0.7989 | 0.7959 | 0.7974 | random_over |
| entropy random_over | 97.39 | 94.48 | 0.8107 | 0.7940 | 0.8023 | random_over |
| basic random_under | 75.06 | 74.48 | 0.3356 | 0.8258 | 0.4773 | random_under |
| gini random_under | 75.06 | 74.48 | 0.3356 | 0.8258 | 0.4773 | random_under |
| entropy random_under | 76.16 | 76.36 | 0.3580 | 0.8521 | 0.5042 | random_under |
| basic smote_tomek | 91.48 | 93.48 | 0.7364 | 0.8371 | 0.7835 | smote_tomek |
| gini smote_tomek | 91.48 | 93.48 | 0.7364 | 0.8371 | 0.7835 | smote_tomek |
| entropy smote_tomek | 91.82 | 93.50 | 0.7300 | 0.8558 | 0.7879 | smote_tomek |
| basic smote_enn | 91.22 | 84.15 | 0.4652 | 0.8258 | 0.5951 | smote_enn |
| gini smote_enn | 91.22 | 84.15 | 0.4652 | 0.8258 | 0.5951 | smote_enn |
| entropy smote_enn | 91.30 | 84.94 | 0.4811 | 0.8596 | 0.6169 | smote_enn |
