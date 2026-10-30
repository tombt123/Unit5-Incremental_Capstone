"""Tasks 2 and 3: train several churn models, track them with MLflow and register the best one.

Run:  python -m src.train
Every model gets its own MLflow run (parameters, metrics, model artifact). The model with the best F1 score
is registered in the MLflow Model Registry and given the alias 'champion'. A local copy (models/model.joblib)
is saved for the API.
View the runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src import config
from src.features import build_pipeline


def candidate_models() -> dict:
    """The models to compare. Class weights help with the 80/20 class imbalance."""
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced", random_state=rs, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                                        random_state=rs),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08,
                                                                 class_weight="balanced", random_state=rs),
    }


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, model.predict_proba(X)[:, 1]),
    }


def simple_params(model) -> dict:
    """Parameters that MLflow can store (numbers, text, booleans)."""
    return {k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))}


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_DB)
    # Pin artifacts to this project's mlruns/ folder so the project works on any machine.
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=config.MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def chart(results: pd.DataFrame, suffix: str, title: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    ax = results[["accuracy", "precision", "recall", "f1"]].plot.bar(figsize=(9, 5), rot=15, width=.8)
    ax.set(title=title, ylabel="score", ylim=(0, 1))
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(config.FIG_DIR / f"model_comparison{suffix}.png", dpi=130); plt.close()


def main(train_path=config.TRAIN_PATH, eval_path=config.TEST_PATH, trigger="manual") -> dict:
    train, evaluation = pd.read_csv(train_path), pd.read_csv(eval_path)
    X_train, y_train = train[config.RAW_FEATURES], train[config.TARGET]
    X_eval, y_eval = evaluation[config.RAW_FEATURES], evaluation[config.TARGET]

    setup_mlflow()
    runs, fitted = {}, {}
    for name, clf in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            model = build_pipeline(clf).fit(X_train, y_train)
            metrics = {k: round(float(v), 4) for k, v in evaluate(model, X_eval, y_eval).items()}
            mlflow.set_tags({"model_family": name, "trigger": trigger})
            mlflow.log_params({**simple_params(clf), "train_rows": len(train), "eval_rows": len(evaluation)})
            mlflow.log_metrics(metrics)
            # cloudpickle: we trained and store these models ourselves (MLflow's default skops format
            # refuses tree models as "untrusted").
            mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3),
                                     serialization_format="cloudpickle")
            runs[name] = {"run_id": run.info.run_id, **metrics}
            fitted[name] = model
            print(f"{name:24s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    results = pd.DataFrame(runs).T.drop(columns="run_id").astype(float)
    config.REPORT_DIR.mkdir(exist_ok=True)
    # Retraining writes its own files so the original comparison is not overwritten.
    suffix = "_retrain" if trigger == "retrain" else ""
    results.round(4).to_csv(config.REPORT_DIR / f"model_comparison{suffix}.csv")
    chart(results, suffix, "Model comparison after retraining (recent holdout)" if suffix
          else "Model comparison (test set)")

    best = results["f1"].idxmax()  # Task 2: best model = highest F1
    print(f"\nBest model by F1: {best} (F1 = {results.loc[best, 'f1']:.4f})")

    # Task 3: register the best model and mark it as the champion.
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{runs[best]['run_id']}/model", config.REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, version.version)
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, version.version, "trigger", trigger)
    print(f"Registered '{config.REGISTERED_MODEL_NAME}' version {version.version} (alias '{config.CHAMPION_ALIAS}')")

    # Save artifacts for deployment.
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[best], config.MODEL_PATH)
    summary = {
        "best_model": best,
        "registered_version": int(version.version),
        "mlflow_run_id": runs[best]["run_id"],
        "trigger": trigger,
        "eval_rows": len(evaluation),
        "metrics": {k: float(results.loc[best, k]) for k in results.columns},
        "all_models": results.round(4).to_dict(orient="index"),
    }
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", default="manual", help="why training ran (manual, ci, retrain)")
    main(trigger=parser.parse_args().trigger)
