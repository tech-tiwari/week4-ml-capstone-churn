from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("data/raw/customer_churn.csv")
MODEL_PATH = Path("models/churn_pipeline.joblib")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

required = {
    "tenure_months", "monthly_charges", "total_charges", "contract",
    "payment_method", "internet_service", "tech_support",
    "online_security", "senior_citizen", "partner", "dependents", "churn"
}
missing_columns = required - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

df = df.drop_duplicates().reset_index(drop=True)
X = df.drop(columns="churn")
y = df["churn"]

numeric_features = [
    "tenure_months", "monthly_charges", "total_charges", "senior_citizen"
]
categorical_features = [
    "contract", "payment_method", "internet_service",
    "tech_support", "online_security", "partner", "dependents"
]

numeric_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
comparison = []

for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    scores = cross_val_score(
        pipe, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    comparison.append(
        {
            "model": name,
            "cv_f1_weighted_mean": scores.mean(),
            "cv_f1_weighted_std": scores.std(),
        }
    )

comparison_df = pd.DataFrame(comparison).sort_values(
    "cv_f1_weighted_mean", ascending=False
)
comparison_df.to_csv(RESULTS / "model_comparison.csv", index=False)

# Tune the strongest practical baseline: Random Forest.
rf_pipe = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
        ),
    ]
)

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 12],
    "model__min_samples_leaf": [1, 3],
}

search = GridSearchCV(
    rf_pipe,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=cv,
    n_jobs=-1,
)
search.fit(X_train, y_train)

final_model = search.best_estimator_
pred = final_model.predict(X_test)
proba = final_model.predict_proba(X_test)[:, list(final_model.classes_).index("Yes")]

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
    "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
    "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    "roc_auc": roc_auc_score((y_test == "Yes").astype(int), proba),
}

with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

report = classification_report(y_test, pred, zero_division=0)
(RESULTS / "classification_report.txt").write_text(report, encoding="utf-8")

cm = confusion_matrix(y_test, pred, labels=["No", "Yes"])
pd.DataFrame(
    cm,
    index=["actual_no", "actual_yes"],
    columns=["predicted_no", "predicted_yes"],
).to_csv(RESULTS / "confusion_matrix.csv")

joblib.dump(final_model, MODEL_PATH)

metadata = {
    "model_version": "1.0",
    "target": "churn",
    "best_params": search.best_params_,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "features": list(X.columns),
}
with open(RESULTS / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Model comparison:")
print(comparison_df.to_string(index=False))
print("\nBest parameters:")
print(search.best_params_)
print("\nTest metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
print(f"\nSaved model to {MODEL_PATH}")
