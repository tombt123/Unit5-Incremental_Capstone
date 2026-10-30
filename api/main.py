"""Task 4: serve the churn model with FastAPI.

Run:   uvicorn api.main:app --reload        Docs: http://127.0.0.1:8000/docs
Secure: send the header  X-API-Key: <value of the API_KEY environment variable>
        (defaults to 'dev-secret-key' for local testing only).
"""
import csv
import json
import os
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src import config

API_KEY = os.getenv("API_KEY", "dev-secret-key")

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_model = None


def get_model():
    """Load the saved model pipeline once and reuse it."""
    global _model
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
        _model = joblib.load(config.MODEL_PATH)
    return _model


def check_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class Customer(BaseModel):
    """One bank customer. Out-of-range values are rejected automatically (HTTP 422)."""
    CreditScore: int = Field(..., ge=300, le=850)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(..., ge=0)

    model_config = {"json_schema_extra": {"examples": [{
        "CreditScore": 619, "Geography": "Germany", "Gender": "Female", "Age": 52, "Tenure": 2,
        "Balance": 125000.0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 0,
        "EstimatedSalary": 101348.88}]}}


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]


def risk_level(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


def log_predictions(frame: pd.DataFrame, probabilities) -> None:
    """Append every request to a CSV so monitoring can compare live data with the training data."""
    config.PREDICTION_LOG.parent.mkdir(exist_ok=True)
    new_file = not config.PREDICTION_LOG.exists()
    stamp = datetime.now(timezone.utc).isoformat()
    with open(config.PREDICTION_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(config.RAW_FEATURES + ["churn_probability", "timestamp"])
        for (_, row), p in zip(frame.iterrows(), probabilities):
            writer.writerow([row[c] for c in config.RAW_FEATURES] + [round(float(p), 4), stamp])


def score(customers: list[Customer]) -> list[Prediction]:
    frame = pd.DataFrame([c.model_dump() for c in customers])[config.RAW_FEATURES]
    probabilities = get_model().predict_proba(frame)[:, 1]
    log_predictions(frame, probabilities)
    return [Prediction(churn_probability=round(float(p), 4), will_churn=bool(p >= 0.5), risk_level=risk_level(p))
            for p in probabilities]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.get("/model-info", dependencies=[Depends(check_key)])
def model_info():
    """Which model is serving, and how it scored when it was trained."""
    if not config.METRICS_PATH.exists():
        raise HTTPException(503, "No model metadata found. Run: python -m src.train")
    info = json.loads(config.METRICS_PATH.read_text())
    return {k: info[k] for k in ("best_model", "registered_version", "trigger", "metrics")}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(check_key)])
def predict(customer: Customer):
    return score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction], dependencies=[Depends(check_key)])
def predict_batch(customers: list[Customer]):
    if not 1 <= len(customers) <= 1000:
        raise HTTPException(422, "Send between 1 and 1000 customers per request")
    return score(customers)
