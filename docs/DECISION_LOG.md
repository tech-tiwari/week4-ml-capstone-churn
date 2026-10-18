# Decision Log

## Dataset

A synthetic dataset was selected because the goal is to demonstrate the complete deployment lifecycle without depending on an external download.

## Preprocessing

Numerical missing values use median imputation because it is robust to outliers and can be fitted only on the training data through the pipeline.

Categorical missing values use the most frequent category to retain rows while keeping the transformation simple and reproducible.

One-hot encoding is used because tree and linear models cannot directly consume arbitrary text categories.

Scaling is applied to numerical features because Logistic Regression is sensitive to feature scale. The same preprocessing pipeline is retained during inference so training and prediction transformations cannot diverge.

## Model comparison

Logistic Regression provides an interpretable baseline. Random Forest captures nonlinear interactions and mixed feature effects. HistGradientBoosting provides another nonlinear baseline.

Weighted F1 is used for comparison because it considers both precision and recall while accounting for class support.

## Leakage prevention

All learned preprocessing steps are inside the scikit-learn Pipeline. Therefore imputation, scaling and category learning are fitted only on training folds during cross-validation.

## Deployment

FastAPI was selected because it provides type validation, automatic OpenAPI documentation and a straightforward ASGI deployment path.

## Serialization

Joblib serializes the entire fitted pipeline, not only the estimator. This ensures the API applies the same preprocessing used during training.
