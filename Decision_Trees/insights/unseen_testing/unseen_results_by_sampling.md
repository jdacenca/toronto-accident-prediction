# Unseen Data Performance Comparison by Sampling Strategy


## None Strategy

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| basic | 80.00 | 0.0000 | 0.0000 | 0.0000 |
| gini | 80.00 | 0.0000 | 0.0000 | 0.0000 |
| entropy | 80.00 | 0.0000 | 0.0000 | 0.0000 |
| weighted | 100.00 | 1.0000 | 1.0000 | 1.0000 |

## Oversampling Strategy

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| basic oversampling | 100.00 | 1.0000 | 1.0000 | 1.0000 |
| gini oversampling | 100.00 | 1.0000 | 1.0000 | 1.0000 |
| entropy oversampling | 75.00 | 1.0000 | 0.5000 | 0.6667 |
| weighted oversampling | 100.00 | 1.0000 | 1.0000 | 1.0000 |

## Undersampling Strategy

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| basic undersampling | 25.00 | 0.0000 | 0.0000 | 0.0000 |
| gini undersampling | 25.00 | 0.0000 | 0.0000 | 0.0000 |
| entropy undersampling | 0.00 | 0.0000 | 0.0000 | 0.0000 |
| weighted undersampling | 25.00 | 0.0000 | 0.0000 | 0.0000 |

## Smote Strategy

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| basic SMOTE | 43.75 | 0.0000 | 0.0000 | 0.0000 |
| gini SMOTE | 43.75 | 0.0000 | 0.0000 | 0.0000 |
| entropy SMOTE | 43.75 | 0.0000 | 0.0000 | 0.0000 |
| weighted SMOTE | 43.75 | 0.0000 | 0.0000 | 0.0000 |
