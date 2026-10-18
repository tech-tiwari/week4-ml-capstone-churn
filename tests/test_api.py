from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

payload = {
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
    "dependents": "No",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction():
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Yes", "No"}
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_validation():
    bad_payload = dict(payload)
    bad_payload["senior_citizen"] = 2
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
