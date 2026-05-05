"""
app.py — FastAPI REST API for Customer Churn Prediction.

Exposes a single POST /predict endpoint that accepts customer features
and returns a churn prediction (0 = No Churn, 1 = Churn).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# ── Load the trained model once at startup ──────────────────────────────────

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    raise RuntimeError(
        "model.pkl not found! Run 'python train.py' first to train and save the model."
    )

# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn based on their account features.",
    version="1.0.0",
)


# ── Request / Response schemas ──────────────────────────────────────────────

class CustomerInput(BaseModel):
    """JSON body expected by the /predict endpoint."""
    tenure: int                # months with the company (e.g. 12)
    monthly_charges: float     # monthly bill amount (e.g. 70.5)
    total_charges: float       # total amount billed (e.g. 840.0)
    contract: str              # "Month-to-month", "One year", or "Two year"
    internet_service: str      # "DSL", "Fiber optic", or "No"
    gender: str                # "Male" or "Female"

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 70.5,
                "total_charges": 840.0,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "gender": "Male",
            }
        }


class PredictionResponse(BaseModel):
    prediction: int            # 0 = No Churn, 1 = Churn
    label: str                 # human-readable label


# ── Helper: encode input to match training features ────────────────────────

def encode_input(data: CustomerInput) -> np.ndarray:
    """
    Convert the JSON input into the same one-hot-encoded feature vector
    that the model was trained on.

    Training columns (after pd.get_dummies with drop_first=True):
        tenure, monthly_charges, total_charges,
        contract_One year, contract_Two year,
        internet_service_Fiber optic, internet_service_No,
        gender_Male
    """
    features = [
        data.tenure,
        data.monthly_charges,
        data.total_charges,
        1 if data.contract == "One year" else 0,
        1 if data.contract == "Two year" else 0,
        1 if data.internet_service == "Fiber optic" else 0,
        1 if data.internet_service == "No" else 0,
        1 if data.gender == "Male" else 0,
    ]
    return np.array(features).reshape(1, -1)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. POST to /predict."}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """Accept customer features and return a churn prediction."""
    try:
        feature_vector = encode_input(customer)
        prediction = int(model.predict(feature_vector)[0])
        label = "Churn" if prediction == 1 else "No Churn"
        return PredictionResponse(prediction=prediction, label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
