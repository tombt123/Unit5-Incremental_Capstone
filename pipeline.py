"""One-command automation of the whole MLOps lifecycle.

Run:  python pipeline.py              # data -> EDA -> train + MLflow -> tests -> quality gate -> monitor -> retrain
      python pipeline.py --watch      # also re-run automatically whenever the data or any src/*.py file changes

The same stages run in GitHub Actions (.github/workflows/ci.yml on every push; retrain.yml weekly).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time

from src import config

WATCHED = [config.RAW_DATA, *sorted((config.ROOT / "src").glob("*.py"))]


def run(cmd: list[str], stage: str) -> None:
    print(f"\n=== {stage} ===", flush=True)
    if subprocess.run(cmd, cwd=config.ROOT).returncode != 0:
        sys.exit(f"Stage '{stage}' failed - stopping pipeline.")


def quality_gate() -> None:
    """Only continue to deployment if the champion model reaches the minimum F1."""
    f1 = json.loads(config.METRICS_PATH.read_text())["metrics"]["f1"]
    print(f"\n=== Quality gate ===\nChampion F1 {f1:.3f} (minimum {config.MIN_F1_TO_DEPLOY})")
    if f1 < config.MIN_F1_TO_DEPLOY:
        sys.exit("Model rejected: not good enough to deploy.")
    print("Model approved for deployment.")


def run_pipeline() -> None:
    py = sys.executable
    run([py, "-m", "src.prepare_data"], "1. Prepare data")
    run([py, "-m", "src.eda"], "2. Explore data (tables + charts)")
    run([py, "-m", "src.train", "--trigger", "pipeline"], "3. Train models, track in MLflow, register best")
    run([py, "-m", "pytest", "-q", "tests"], "4. Automated tests")
    quality_gate()
    run([py, "-m", "monitoring.monitor"], "5. Monitoring simulation (Evidently)")
    run([py, "-m", "src.retrain"], "6. Retrain if accuracy dropped")
    print("\nPipeline finished. Serve the model with:  uvicorn api.main:app")


def fingerprint() -> str:
    h = hashlib.md5()
    for p in WATCHED:
        h.update(p.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="re-run when data or code changes")
    args = parser.parse_args()
    run_pipeline()
    if args.watch:
        last = fingerprint()
        print("\nWatching for changes (Ctrl+C to stop)...")
        while True:
            time.sleep(5)
            if (now := fingerprint()) != last:
                print("\nChange detected - re-running pipeline.")
                last = now
                run_pipeline()
