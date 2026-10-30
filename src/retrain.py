"""Task 4: retrain the model when monitoring shows accuracy has dropped.

Run once:       python -m src.retrain              # retrains only if monitoring/reports/status.json says so
                python -m src.retrain --force      # retrain regardless
Run on a loop:  python -m src.retrain --loop 3600  # every hour: monitor, then retrain if needed
In CI, .github/workflows/retrain.yml runs the same steps on a weekly schedule.

Retraining uses the original training data plus the most recent labelled data (weighted x3 by repetition)
and is judged on a separate recent holdout. Every retrain is logged to MLflow with the tag trigger=retrain
and registered as a new version of the model, which becomes the 'champion'.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src import train as trainer

STATUS_PATH = config.MONITOR_DIR / "status.json"
RECENT_WEIGHT = 3


def retrain(force: bool = False) -> bool:
    status = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {"retrain_needed": False}
    if not (status["retrain_needed"] or force):
        print("No retraining needed: accuracy is within tolerance.")
        return False

    old_info = json.loads(config.METRICS_PATH.read_text())
    old_model = joblib.load(config.MODEL_PATH)
    holdout = pd.read_csv(config.RECENT_HOLDOUT_PATH)
    old_pred = old_model.predict(holdout[config.RAW_FEATURES])
    old_acc, old_f1 = accuracy_score(holdout[config.TARGET], old_pred), f1_score(holdout[config.TARGET], old_pred)
    print(f"Current model ({old_info['best_model']}, v{old_info['registered_version']}) on recent data: "
          f"accuracy={old_acc:.3f}, F1={old_f1:.3f}")

    combined = pd.concat([pd.read_csv(config.TRAIN_PATH)] + [pd.read_csv(config.RECENT_BATCH_PATH)] * RECENT_WEIGHT)
    combined_path = config.PROCESSED_DIR / "train_retrain.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Retraining on {len(combined)} rows ({RECENT_WEIGHT}x weight on the most recent data)...")

    new_info = trainer.main(train_path=combined_path, eval_path=config.RECENT_HOLDOUT_PATH, trigger="retrain")
    record = {
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "reason": "forced" if force and not status["retrain_needed"] else "accuracy below threshold",
        "previous_model": old_info["best_model"], "previous_version": old_info["registered_version"],
        "previous_accuracy_on_recent_data": round(old_acc, 4), "previous_f1_on_recent_data": round(old_f1, 4),
        "new_model": new_info["best_model"], "new_version": new_info["registered_version"],
        "new_accuracy_on_recent_data": new_info["metrics"]["accuracy"], "new_f1_on_recent_data": new_info["metrics"]["f1"],
    }
    config.MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    (config.MONITOR_DIR / "retrain_log.json").write_text(json.dumps(record, indent=2))
    status.update(retrain_needed=False, last_retrain=record["retrained_at"])
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    print(f"\nRetrained: accuracy on recent data {old_acc:.3f} -> {record['new_accuracy_on_recent_data']:.3f}, "
          f"F1 {old_f1:.3f} -> {record['new_f1_on_recent_data']:.3f}. "
          f"New champion: v{record['new_version']} ({record['new_model']}).")
    return True


def monitor_then_retrain() -> None:
    subprocess.run([sys.executable, "-m", "monitoring.monitor"], cwd=config.ROOT, check=True)
    retrain()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--loop", type=int, metavar="SECONDS", help="repeat monitor + retrain every SECONDS")
    args = parser.parse_args()
    if args.loop:
        while True:
            print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] scheduled check")
            monitor_then_retrain()
            time.sleep(args.loop)
    else:
        retrain(args.force)
