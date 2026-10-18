# Week 4 Capstone Build Summary

## Model results

| model                |   cv_f1_weighted_mean |   cv_f1_weighted_std |
|:---------------------|----------------------:|---------------------:|
| LogisticRegression   |              0.731673 |            0.0176348 |
| HistGradientBoosting |              0.693351 |            0.0177614 |
| RandomForest         |              0.690552 |            0.0147039 |

### Held-out test set

- Accuracy: 0.7040
- Weighted precision: 0.7037
- Weighted recall: 0.7040
- Weighted F1: 0.7037
- ROC-AUC: 0.7333

## Verification

Pytest result:

```text
[32m.[0m[32m.[0m[32m.[0m[32m                                                                      [100%][0m
[32m[32m[1m3 passed[0m[32m in 1.33s[0m[0m
```

The trained model is saved at `models/churn_pipeline.joblib`.
