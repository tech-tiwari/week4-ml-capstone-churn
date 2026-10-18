# Week 4 ML Capstone — Customer Churn Prediction API

## Overview

This project is an end-to-end Machine Learning application built for the YuvaIntern AI Pioneers Week 4 task.

The application predicts whether a telecom customer is likely to churn (`Yes` / `No`) from customer and service attributes.

### Pipeline

`Raw data → validation → preprocessing → model comparison → hyperparameter tuning → evaluation → Joblib serialization → FastAPI prediction API`

## Objectives covered

- Data loading and validation
- Exploratory data analysis
- Missing-value handling
- Categorical encoding
- Numerical scaling
- Train/test split
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Evaluation with accuracy, precision, recall, F1 and ROC-AUC
- Model serialization with Joblib
- FastAPI prediction endpoint
- Input validation with Pydantic
- Automated API tests
- Documentation and presentation

## Dataset

The project uses a generated telecom-style dataset for a reproducible capstone demonstration. The generator deliberately creates realistic relationships between customer attributes and churn, including a controlled amount of missing data.

This avoids requiring an external dataset download and makes the project reproducible from scratch.

The dataset is **synthetic**, so model performance must not be interpreted as real-world business performance.

## Project structure

```text
week-4-ml-capstone/
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── API_GUIDE.md
│   ├── MODEL_CARD.md
│   └── DECISION_LOG.md
├── models/
├── presentation/
├── reports/
├── results/
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 1. Generate data

```bash
python src/generate_data.py
```

This creates:

```text
data/raw/customer_churn.csv
```

## 2. Train and evaluate

```bash
python src/train.py
```

The script:

1. validates the data,
2. separates target and features,
3. creates a preprocessing pipeline,
4. compares Logistic Regression, Random Forest and HistGradientBoosting,
5. tunes the selected model,
6. evaluates it on a held-out test set,
7. saves the final pipeline to `models/churn_pipeline.joblib`,
8. writes evaluation outputs to `results/`.

## 3. Start the API

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides an interactive Swagger UI.

## 4. Example request

POST `/predict`

```json
{
  "tenure_months": 5,
  "monthly_charges": 92.5,
  "total_charges": 480.0,
  "contract": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "online_security": "No",
  "senior_citizen": 0,
  "partner": "No",
  "dependents": "No"
}
```

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.78,
  "model_version": "1.0"
}
```

The exact probability depends on the trained model.

## 5. Run tests

```bash
pytest -q
```

## 6. Single prediction from Python

```bash
python src/predict.py
```

## Deployment

The API is production-structured and can be deployed to any platform supporting Python ASGI applications.

Start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For a real production deployment, use environment-based configuration, HTTPS, authentication, logging, monitoring and a managed model registry.

## Important limitation

Because the dataset is synthetic, this project demonstrates the **engineering and ML workflow**, not a validated production churn model.

A real deployment should replace the synthetic dataset with an approved real dataset, perform data governance checks, retrain on representative data and monitor drift.

## Week 4 deliverables

- End-to-end ML pipeline
- Serialized model
- FastAPI prediction API
- API documentation
- Automated tests
- Model card
- Decision log
- Evaluation report
- Presentation
