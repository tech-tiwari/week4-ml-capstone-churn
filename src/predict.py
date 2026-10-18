import joblib
import pandas as pd

MODEL_PATH = "models/churn_pipeline.joblib"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame(
    [
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
            "dependents": "No",
        }
    ]
)

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[:, list(model.classes_).index("Yes")][0]

print("Prediction:", prediction)
print("Churn probability:", round(float(probability), 4))
