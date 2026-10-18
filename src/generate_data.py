from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N = 2500

rng = np.random.default_rng(RANDOM_STATE)

tenure = rng.integers(1, 73, N)
monthly = np.clip(rng.normal(72, 25, N), 20, 130)
contract = rng.choice(
    ["Month-to-month", "One year", "Two year"],
    N,
    p=[0.55, 0.25, 0.20],
)
payment = rng.choice(
    ["Electronic check", "Bank transfer", "Credit card", "Mailed check"],
    N,
    p=[0.35, 0.25, 0.25, 0.15],
)
internet = rng.choice(
    ["DSL", "Fiber optic", "No"],
    N,
    p=[0.35, 0.45, 0.20],
)
tech_support = rng.choice(["Yes", "No"], N, p=[0.30, 0.70])
security = rng.choice(["Yes", "No"], N, p=[0.35, 0.65])
senior = rng.binomial(1, 0.18, N)
partner = rng.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], N, p=[0.30, 0.70])

total = tenure * monthly + rng.normal(0, 80, N)
total = np.clip(total, 20, None)

# A transparent synthetic churn mechanism.
logit = (
    1.7
    - 0.045 * tenure
    + 0.018 * (monthly - 70)
    + 0.45 * (contract == "Month-to-month")
    - 0.35 * (contract == "Two year")
    + 0.35 * (payment == "Electronic check")
    + 0.30 * (internet == "Fiber optic")
    - 0.45 * (tech_support == "Yes")
    - 0.35 * (security == "Yes")
    + 0.30 * senior
    - 0.20 * (partner == "Yes")
    - 0.25 * (dependents == "Yes")
)

prob = 1 / (1 + np.exp(-logit))
churn = np.where(rng.random(N) < prob, "Yes", "No")

df = pd.DataFrame(
    {
        "tenure_months": tenure,
        "monthly_charges": monthly.round(2),
        "total_charges": total.round(2),
        "contract": contract,
        "payment_method": payment,
        "internet_service": internet,
        "tech_support": tech_support,
        "online_security": security,
        "senior_citizen": senior,
        "partner": partner,
        "dependents": dependents,
        "churn": churn,
    }
)

# Controlled missingness to demonstrate preprocessing.
for col, frac in {
    "monthly_charges": 0.025,
    "total_charges": 0.02,
    "tech_support": 0.015,
}.items():
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

out = Path("data/raw/customer_churn.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"Saved {len(df):,} rows to {out}")
print(df.head())
