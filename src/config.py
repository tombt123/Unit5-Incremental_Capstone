"""Central configuration: paths, feature lists and settings used by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- data ----
RAW_DATA = ROOT / "data" / "raw" / "Churn_Modeling.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
RECENT_BATCH_PATH = PROCESSED_DIR / "recent_batch.csv"      # labelled "new" data (simulated)
RECENT_HOLDOUT_PATH = PROCESSED_DIR / "recent_holdout.csv"  # separate labelled data to judge a retrained model

# ---- outputs ----
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
MONITOR_DIR = ROOT / "monitoring" / "reports"
PREDICTION_LOG = ROOT / "monitoring" / "prediction_log.csv"

# ---- MLflow ----
MLFLOW_DB = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACTS = ROOT / "mlruns"
EXPERIMENT_NAME = "churn-prediction"
REGISTERED_MODEL_NAME = "churn-classifier"
CHAMPION_ALIAS = "champion"

# ---- modelling ----
TARGET = "Exited"
RANDOM_STATE = 42

# Columns the model receives (and the API accepts). Identifiers (RowNumber, CustomerId, Surname) are dropped.
RAW_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
# Engineered inside the model pipeline (see src/features.py)
NUMERIC_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                    "EstimatedSalary", "BalanceSalaryRatio", "TenureByAge"]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember", "HasBalance"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]

MIN_F1_TO_DEPLOY = 0.55          # quality gate: a model scoring below this F1 is not deployed

# ---- monitoring / retraining ----
N_PERIODS = 6                    # simulated weeks of monitoring
ACCURACY_DROP_TOLERANCE = 0.05   # retrain if accuracy falls more than 5 points below the baseline
DRIFT_SHARE_ALERT = 0.20         # also flag when >= 20% of columns drift
