# Week 4 Evaluation Report

## 1. Problem

Customer churn is a binary classification problem. The system predicts whether a customer is likely to leave based on service and customer attributes.

## 2. Data

The dataset contains 2,500 synthetic customer records. It contains numerical and categorical features and a controlled amount of missing data.

The synthetic nature of the data is an explicit limitation.

## 3. Preprocessing

### Numerical

- Median imputation
- Standard scaling

### Categorical

- Most-frequent imputation
- One-hot encoding
- Unknown-category handling

The preprocessing is part of the training pipeline. This reduces the risk of applying inconsistent transformations at inference time.

## 4. Model selection

Three models are compared:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

The comparison uses 5-fold stratified cross-validation and weighted F1.

Random Forest is then tuned with GridSearchCV and evaluated once on the held-out test set.

The actual scores are stored in:

- `results/model_comparison.csv`
- `results/metrics.json`
- `results/classification_report.txt`
- `results/confusion_matrix.csv`

## 5. Evaluation metrics

### Accuracy

Fraction of all predictions that are correct.

### Precision

Among customers predicted to churn, the proportion that actually churn.

### Recall

Among customers who actually churn, the proportion detected by the model.

### F1

Harmonic mean of precision and recall.

### ROC-AUC

Measures ranking ability across classification thresholds.

## 6. Why multiple metrics?

A single accuracy value can hide poor performance on one class. Precision and recall expose different error types, while F1 balances them. ROC-AUC gives a threshold-independent ranking view.

## 7. Deployment

The fitted preprocessing + model pipeline is serialized using Joblib.

FastAPI loads the serialized pipeline when the application starts and exposes a `/predict` endpoint.

## 8. Testing

The test suite checks:

- API health
- successful prediction
- prediction probability range
- input validation

Run:

```bash
pytest -q
```

## 9. Limitations

The biggest limitation is synthetic data. A production model would require:

- real historical data,
- temporal validation,
- calibration analysis,
- subgroup/bias analysis,
- drift monitoring,
- retraining strategy,
- authentication and authorization,
- HTTPS,
- logging and observability.
