# API Guide

## Run

```bash
uvicorn api.main:app --reload
```

Swagger UI:

`http://127.0.0.1:8000/docs`

## Endpoints

### GET `/`

Returns basic API information.

### GET `/health`

Checks whether the API is running and whether the model is loaded.

### POST `/predict`

Accepts customer features and returns a churn prediction plus probability.

Example body:

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

## cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"tenure_months\":5,\"monthly_charges\":92.5,\"total_charges\":480.0,\"contract\":\"Month-to-month\",\"payment_method\":\"Electronic check\",\"internet_service\":\"Fiber optic\",\"tech_support\":\"No\",\"online_security\":\"No\",\"senior_citizen\":0,\"partner\":\"No\",\"dependents\":\"No\"}"
```

The returned probability is the model's estimated probability of the `Yes` class.
