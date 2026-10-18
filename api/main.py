from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/churn_pipeline.joblib")

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Prediction API for the Week 4 ML capstone project.",
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0, le=1000)
    total_charges: float = Field(..., ge=0, le=100000)
    contract: str
    payment_method: str
    internet_service: str
    tech_support: str
    online_security: str
    senior_citizen: int = Field(..., ge=0, le=1)
    partner: str
    dependents: str


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Run `python src/train.py` first.",
        )

    data = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(data)[0]
    probability = float(
        model.predict_proba(data)[:, list(model.classes_).index("Yes")][0]
    )

    return {
        "prediction": str(prediction),
        "churn_probability": round(probability, 4),
        "model_version": "1.0",
    }
