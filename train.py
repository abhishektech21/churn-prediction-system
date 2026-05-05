"""
train.py — Train a Random Forest model on a synthetic customer churn dataset.

This script:
  1. Generates a small, realistic churn dataset (no external download needed).
  2. Preprocesses the data (handles missing values, encodes categoricals).
  3. Trains a Random Forest classifier.
  4. Evaluates on a held-out test set and prints metrics.
  5. Saves the trained model as model.pkl using joblib.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── 1. Generate a synthetic customer churn dataset ──────────────────────────

np.random.seed(42)
N = 1000  # number of customers

data = pd.DataFrame({
    "tenure":           np.random.randint(1, 72, N),          # months with company
    "monthly_charges":  np.round(np.random.uniform(20, 120, N), 2),
    "total_charges":    np.round(np.random.uniform(100, 8000, N), 2),
    "contract":         np.random.choice(["Month-to-month", "One year", "Two year"], N),
    "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], N),
    "gender":           np.random.choice(["Male", "Female"], N),
})

# Create a realistic churn label (higher churn for month-to-month, high charges, low tenure)
churn_probability = (
    0.3
    + 0.25 * (data["contract"] == "Month-to-month").astype(float)
    - 0.15 * (data["contract"] == "Two year").astype(float)
    + 0.002 * data["monthly_charges"]
    - 0.004 * data["tenure"]
    + 0.10 * (data["internet_service"] == "Fiber optic").astype(float)
)
churn_probability = churn_probability.clip(0.05, 0.95)
data["churn"] = np.random.binomial(1, churn_probability)

# Inject a few missing values for realism
missing_idx = np.random.choice(N, size=15, replace=False)
data.loc[missing_idx, "total_charges"] = np.nan

print(f"Dataset shape : {data.shape}")
print(f"Churn distribution:\n{data['churn'].value_counts()}\n")

# ── 2. Preprocess ───────────────────────────────────────────────────────────

# Fill missing values with the column median
data["total_charges"] = data["total_charges"].fillna(data["total_charges"].median())

# One-hot encode categorical columns
data_encoded = pd.get_dummies(data, columns=["contract", "internet_service", "gender"], drop_first=True)

# Separate features and target
X = data_encoded.drop("churn", axis=1)
y = data_encoded["churn"]

print(f"Features used ({len(X.columns)}): {list(X.columns)}\n")

# ── 3. Train / Test split ──────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train a Random Forest ───────────────────────────────────────────────

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 5. Evaluate ────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 6. Save the model ──────────────────────────────────────────────────────

joblib.dump(model, "model.pkl")
print("[OK] Model saved to model.pkl")
