# Model Card

## Model

Random Forest classifier wrapped in a scikit-learn preprocessing pipeline.

## Intended use

Educational demonstration of an end-to-end ML deployment workflow.

## Input

Customer tenure, charges, contract, payment method, internet service and support/security attributes.

## Output

- `Yes` or `No` churn prediction
- estimated probability for the `Yes` class

## Training

- Stratified 80/20 train-test split
- 5-fold stratified cross-validation
- median imputation for numerical features
- most-frequent imputation for categorical features
- standard scaling for numerical variables
- one-hot encoding for categorical variables
- GridSearchCV for Random Forest hyperparameters

## Limitations

1. The data is synthetic.
2. It does not represent a real telecom population.
3. Probabilities should not be treated as calibrated business probabilities without calibration analysis.
4. The model should not be used for real customer decisions without validation, governance and monitoring.
