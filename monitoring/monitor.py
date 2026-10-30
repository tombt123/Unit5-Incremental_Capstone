"""Task 4 (monitoring): track model accuracy and data drift over time with Evidently AI.

Run:  python -m monitoring.monitor           # simulation: 6 'weeks' of increasingly drifted customer data
      python -m monitoring.monitor --live    # drift check on real requests logged by the API
Add --fail-on-alert to exit with code 1 when retraining is needed (useful in CI / schedulers).

Outputs (monitoring/reports/): interactive Evidently HTML reports, performance_log.csv,
accuracy_over_time.png and status.json (read by src/retrain.py).
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src import config
from src.drift_sim import drift_world

NUMERICAL = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
BATCH_SIZE = 800


def make_dataset(df: pd.DataFrame, classification: bool = False) -> Dataset:
    kwargs = {"numerical_columns": NUMERICAL, "categorical_columns": CATEGORICAL}
    if classification:
        kwargs["classification"] = [BinaryClassification(target=config.TARGET, prediction_labels="prediction",
                                                         prediction_probas=None, pos_label=1)]
    return Dataset.from_pandas(df, data_definition=DataDefinition(**kwargs))


def drift_check(reference: pd.DataFrame, current: pd.DataFrame):
    """Evidently data-drift report: returns (share of drifted columns, count, report snapshot)."""
    snapshot = Report([DataDriftPreset()]).run(make_dataset(current[config.RAW_FEATURES]),
                                              make_dataset(reference[config.RAW_FEATURES]))
    share = count = None
    for metric in snapshot.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            share, count = float(metric["value"]["share"]), int(metric["value"]["count"])
    return share, count, snapshot


def classification_scores(model, batch: pd.DataFrame) -> dict:
    pred = model.predict(batch[config.RAW_FEATURES])
    y = batch[config.TARGET]
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}


def draw_chart(log: pd.DataFrame, baseline: float, threshold: float) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(log["period"], log["accuracy"], "o-", color="#2a6f97", label="accuracy")
    ax1.plot(log["period"], log["f1"], "s-", color="#6a994e", label="F1")
    ax1.axhline(baseline, color="grey", ls=":", label=f"baseline accuracy ({baseline:.3f})")
    ax1.axhline(threshold, color="#c1121f", ls="--", label=f"retrain threshold ({threshold:.3f})")
    ax1.set(title="Model performance over time", xlabel="week", ylabel="score", ylim=(0, 1))
    ax1.legend(loc="lower left")
    colours = ["#c1121f" if s >= config.DRIFT_SHARE_ALERT else "#2a6f97" for s in log["drift_share"]]
    ax2.bar(log["period"], log["drift_share"] * 100, color=colours)
    ax2.axhline(config.DRIFT_SHARE_ALERT * 100, color="#c1121f", ls="--", label="alert level")
    ax2.set(title="Share of input columns with drift", xlabel="week", ylabel="% of columns")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(config.MONITOR_DIR / "accuracy_over_time.png", dpi=130)
    plt.close(fig)


def run_simulation() -> dict:
    model = joblib.load(config.MODEL_PATH)
    baseline = float(json.loads(config.METRICS_PATH.read_text())["metrics"]["accuracy"])
    threshold = baseline - config.ACCURACY_DROP_TOLERANCE
    reference = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    pool_a, pool_b = test.iloc[: len(test) // 2], test.iloc[len(test) // 2:]  # b is kept for judging retraining
    rng = np.random.default_rng(config.RANDOM_STATE)

    rows, snapshots, batches = [], {}, {}
    for week in range(config.N_PERIODS):
        severity = week / (config.N_PERIODS - 1)
        batch = drift_world(pool_a.sample(BATCH_SIZE, random_state=100 + week), severity, rng)
        scores = classification_scores(model, batch)
        share, count, snapshot = drift_check(reference, batch)
        rows.append({"period": week, "drift_severity": round(severity, 2), **{k: round(v, 4) for k, v in scores.items()},
                     "drift_share": round(share, 3), "drifted_columns": count,
                     "accuracy_alert": scores["accuracy"] < threshold, "drift_alert": share >= config.DRIFT_SHARE_ALERT})
        snapshots[week], batches[week] = snapshot, batch

    log = pd.DataFrame(rows)
    config.MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(config.MONITOR_DIR / "performance_log.csv", index=False)
    draw_chart(log, baseline, threshold)

    last = config.N_PERIODS - 1
    snapshots[0].save_html(str(config.MONITOR_DIR / "data_drift_baseline.html"))
    snapshots[last].save_html(str(config.MONITOR_DIR / "data_drift_latest.html"))
    try:  # accuracy / precision / recall report: latest week vs. the first (undrifted) week
        for week, name in ((0, "ref"), (last, "cur")):
            batches[week] = batches[week].assign(prediction=model.predict(batches[week][config.RAW_FEATURES]))
        perf = Report([ClassificationPreset()]).run(make_dataset(batches[last], True), make_dataset(batches[0], True))
        perf.save_html(str(config.MONITOR_DIR / "model_performance_latest.html"))
    except Exception as exc:  # the drift report is the key artefact; don't fail the run over this one
        print(f"(Skipped model performance report: {exc.__class__.__name__}: {exc})")

    # Labelled "new" data, saved so src/retrain.py can learn from it and be judged on a separate holdout.
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    drift_world(pool_a.sample(BATCH_SIZE, random_state=200), 1.0, rng).to_csv(config.RECENT_BATCH_PATH, index=False)
    drift_world(pool_b.sample(BATCH_SIZE, random_state=300), 1.0, rng).to_csv(config.RECENT_HOLDOUT_PATH, index=False)

    latest = log.iloc[-1]
    status = {"mode": "simulation", "checked_at": datetime.now(timezone.utc).isoformat(),
              "baseline_accuracy": round(baseline, 4), "accuracy_threshold": round(threshold, 4),
              "latest_accuracy": float(latest["accuracy"]), "latest_drift_share": float(latest["drift_share"]),
              "retrain_needed": bool(latest["accuracy"] < threshold)}
    (config.MONITOR_DIR / "status.json").write_text(json.dumps(status, indent=2))

    print(log[["period", "drift_severity", "accuracy", "f1", "drift_share", "drifted_columns",
               "accuracy_alert", "drift_alert"]].to_string(index=False))
    print(f"\nBaseline accuracy {baseline:.3f}; retrain threshold {threshold:.3f}; latest accuracy {latest['accuracy']:.3f}")
    if status["retrain_needed"]:
        print("ALERT: accuracy has dropped below the threshold - retraining needed (python -m src.retrain).")
    else:
        print("OK: accuracy is within tolerance.")
    return status


def run_live() -> dict:
    """Drift check on real API traffic (no outcomes are known yet, so accuracy cannot be measured)."""
    if not config.PREDICTION_LOG.exists() or len(pd.read_csv(config.PREDICTION_LOG)) < 30:
        print("Fewer than 30 logged API requests - send more requests to /predict first.")
        return {"mode": "live", "retrain_needed": False}
    current = pd.read_csv(config.PREDICTION_LOG)
    share, count, snapshot = drift_check(pd.read_csv(config.TRAIN_PATH), current)
    config.MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(config.MONITOR_DIR / "data_drift_live.html"))
    print(f"Live traffic: {count} of {len(config.RAW_FEATURES)} columns drifted ({share:.0%}).")
    if share >= config.DRIFT_SHARE_ALERT:
        print("ALERT: input data has drifted - check outcomes and consider retraining.")
    return {"mode": "live", "drift_share": share, "retrain_needed": share >= config.DRIFT_SHARE_ALERT}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="check real API traffic instead of simulating")
    parser.add_argument("--fail-on-alert", action="store_true", help="exit with code 1 if retraining is needed")
    args = parser.parse_args()
    result = run_live() if args.live else run_simulation()
    sys.exit(1 if args.fail_on_alert and result["retrain_needed"] else 0)
