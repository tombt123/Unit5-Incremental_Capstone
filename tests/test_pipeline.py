"""Automated tests: run with  python -m pytest -q tests
(The model-dependent tests need `python -m src.prepare_data` and `python -m src.train` to have run first;
`python pipeline.py` does this for you.)"""
import json
import os

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src import config
from src.drift_sim import drift_world
from src.features import add_features, load_raw
from src.prepare_data import clean

client = TestClient(app)
HEADERS = {"X-API-Key": os.getenv("API_KEY", "dev-secret-key")}
CUSTOMER = {"CreditScore": 600, "Geography": "Germany", "Gender": "Female", "Age": 50, "Tenure": 3,
            "Balance": 120000.0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 0,
            "EstimatedSalary": 90000.0}


# ---------- data preparation ----------
def test_load_raw_fixes_id_columns():
    df = load_raw()
    assert {"RowNumber", "CustomerId"} <= set(df.columns)
    assert df["CustomerId"].is_unique and len(df) == 10000


def test_clean_drops_identifiers_and_has_no_missing_values():
    df = clean(load_raw())
    assert list(df.columns) == config.RAW_FEATURES + [config.TARGET]
    assert df.isna().sum().sum() == 0


def test_add_features():
    row = pd.DataFrame([{"Balance": 0.0, "EstimatedSalary": 50000.0, "Tenure": 5, "Age": 50}])
    out = add_features(row).iloc[0]
    assert out["HasBalance"] == 0 and out["BalanceSalaryRatio"] == 0 and out["TenureByAge"] == 0.1


def test_train_test_split_is_stratified():
    train, test = pd.read_csv(config.TRAIN_PATH), pd.read_csv(config.TEST_PATH)
    assert len(train) + len(test) == 10000
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.01


# ---------- model ----------
def test_model_meets_quality_gate_and_is_registered():
    info = json.loads(config.METRICS_PATH.read_text())
    assert info["metrics"]["f1"] >= config.MIN_F1_TO_DEPLOY
    assert info["best_model"] in info["all_models"]
    assert info["registered_version"] >= 1
    best_f1 = max(m["f1"] for m in info["all_models"].values())
    assert info["metrics"]["f1"] == best_f1  # the champion really is the best by F1


# ---------- drift simulation ----------
def test_drift_world_severity_zero_changes_nothing():
    df = pd.read_csv(config.TEST_PATH)
    pd.testing.assert_frame_equal(drift_world(df, 0.0, np.random.default_rng(1)), df)


def test_drift_world_severity_one_shifts_age_and_flips_labels():
    df = pd.read_csv(config.TEST_PATH)
    drifted = drift_world(df, 1.0, np.random.default_rng(1))
    assert (drifted["Age"] == df["Age"] + 12).all()
    assert drifted[config.TARGET].mean() > df[config.TARGET].mean()


# ---------- API ----------
def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_requires_api_key():
    assert client.post("/predict", json=CUSTOMER).status_code == 401
    assert client.post("/predict", json=CUSTOMER, headers={"X-API-Key": "wrong"}).status_code == 401


def test_predict_returns_valid_result():
    r = client.post("/predict", json=CUSTOMER, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["will_churn"] == (body["churn_probability"] >= 0.5)


def test_risky_customer_scores_higher_than_safe_customer():
    safe = {**CUSTOMER, "Geography": "France", "Age": 28, "IsActiveMember": 1, "NumOfProducts": 2}
    risky = client.post("/predict", json=CUSTOMER, headers=HEADERS).json()["churn_probability"]
    calm = client.post("/predict", json=safe, headers=HEADERS).json()["churn_probability"]
    assert risky > calm


def test_predict_rejects_bad_input():
    for bad in ({**CUSTOMER, "Age": 5}, {**CUSTOMER, "Geography": "Mars"}, {**CUSTOMER, "HasCrCard": 3}):
        assert client.post("/predict", json=bad, headers=HEADERS).status_code == 422


def test_batch_and_model_info():
    r = client.post("/predict/batch", json=[CUSTOMER, CUSTOMER], headers=HEADERS)
    assert r.status_code == 200 and len(r.json()) == 2
    info = client.get("/model-info", headers=HEADERS).json()
    assert info["metrics"]["f1"] >= config.MIN_F1_TO_DEPLOY
