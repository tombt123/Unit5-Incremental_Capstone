# Bank Customer Churn: an end-to-end MLOps project

Predicts which of 10,000 bank customers will leave (`Exited = 1`), and covers the whole lifecycle:
data exploration, multi-model training, MLflow experiment tracking and model registry, a FastAPI service in a
Docker container, drift/accuracy monitoring with Evidently AI, and automatic retraining.

**Tools:** Python, pandas, scikit-learn, MLflow, FastAPI, Docker, Evidently AI, GitHub Actions.

## Quick start (for graders)

```bash
pip install -r requirements.txt
python pipeline.py            # runs everything below in about 2 minutes
```

`pipeline.py` runs: prepare data, EDA, train 4 models (MLflow), 13 automated tests, a quality gate,
the monitoring simulation, and retraining. Then:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiments + model registry at http://127.0.0.1:5000
API_KEY=choose-a-secret uvicorn api.main:app          # API docs at http://127.0.0.1:8000/docs
```

The dataset is included (`data/raw/Churn_Modeling.csv`), so no download is needed.
Python 3.12 to 3.14 works; `pandas` and `evidently` are pinned because newer pandas breaks Evidently 0.7.

## Project layout

```
data/raw/Churn_Modeling.csv    the dataset
src/config.py                  paths, feature lists, thresholds (one place to change settings)
src/features.py                data loading + feature engineering + model pipeline definition
src/prepare_data.py            cleaning and stratified train/test split
src/eda.py                     Task 1: exploration, grouping, pivot tables, charts
src/train.py                   Tasks 2-3: trains 4 models, logs to MLflow, registers the best
src/drift_sim.py               simulated customer-behaviour drift (for the monitoring demo)
src/retrain.py                 Task 4: retrains when accuracy drops (one-shot or on a loop)
monitoring/monitor.py          Task 4: Evidently drift + accuracy monitoring
api/main.py                    Task 4: FastAPI service
tests/test_pipeline.py         13 automated tests
pipeline.py                    one-command automation of all stages
Dockerfile                     container for the API
.github/workflows/ci.yml       CI/CD: tests + reports on every push, image build on main
.github/workflows/retrain.yml  weekly scheduled monitoring and retraining
reports/                       EDA insights, tables, charts, model comparison
monitoring/reports/            Evidently HTML reports, accuracy-over-time chart, retrain log
```

## Task 1: Working with the churn data (`python -m src.eda`)

* 10,000 rows; **no missing values, no duplicates, no impossible values**. 36% of customers have a zero balance.
* **Data quirk:** the first column is headed `CustomerId` but is a row number; the real id is in `CustomerId.1`.
  `load_raw()` renames them, and both ids plus `Surname` are dropped before modelling.
* **Class imbalance:** 20.4% churn. Always predicting "stays" would score 79.6% accuracy, so models are judged on F1.
* Groupby and pivot tables (`reports/tables/`): churn by Geography, Gender, NumOfProducts, IsActiveMember and age group,
  plus Geography x Gender pivots. Charts are in `reports/figures/`; the written summary is `reports/eda_insights.md`.

Key patterns: Germany churns most (32.4% vs about 16% in France and Spain); women churn more than men (25.1% vs 16.5%);
churn rises with age (56% at 51 to 60); inactive members churn at 26.9% vs 14.3%; customers with 3 or 4 products
churn at 83% and 100%, while 2 products is the safest (7.6%).

## Task 2: Predictive models (`python -m src.train`)

Four models share one scikit-learn pipeline (feature engineering, then scaling and encoding, then classifier), evaluated on a
stratified 20% test set (2,000 customers). Engineered features: `HasBalance`, `BalanceSalaryRatio`, `TenureByAge`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.713 | 0.388 | 0.713 | 0.502 | 0.775 |
| **Random forest (best by F1)** | 0.818 | 0.539 | 0.730 | **0.620** | 0.863 |
| Gradient boosting | 0.870 | 0.790 | 0.489 | 0.604 | 0.868 |
| Hist. gradient boosting | 0.812 | 0.527 | 0.735 | 0.614 | 0.861 |

Gradient boosting has the highest accuracy but misses half of the churners (recall 0.49), so F1 picks the random forest.
Which to prefer depends on the cost of a missed churner vs a wasted retention offer. The best model is saved to
`models/model.joblib`; its details are in `models/metrics.json`.

## Task 3: Experiment tracking with MLflow

Each model is one MLflow run (parameters, accuracy/precision/recall/F1/ROC-AUC, the model artifact, and tags for the model
family and what triggered training). The best run is registered as **`churn-classifier`** and given the alias **`champion`**.

To compare visually: run `mlflow ui --backend-store-uri sqlite:///mlflow.db`, open the `churn-prediction` experiment,
tick the runs and click **Compare** (parallel coordinates and scatter plots), then open **Models** for the registry.

## Task 4: Deployment and monitoring

**API** (`uvicorn api.main:app`, docs at `/docs`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `POST /predict` | one customer in, churn probability, yes/no and risk level out |
| `POST /predict/batch` | up to 1,000 customers at once |
| `GET /model-info` | which model is serving and how it scored |

Send the header `X-API-Key` (the value of the `API_KEY` environment variable; the default `dev-secret-key` is for local testing only).
Inputs are validated (for example, `Age` 18 to 100, `Geography` one of France/Germany/Spain), and bad input returns HTTP 422.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"CreditScore":619,"Geography":"Germany","Gender":"Female","Age":52,"Tenure":2,"Balance":125000,
       "NumOfProducts":1,"HasCrCard":1,"IsActiveMember":0,"EstimatedSalary":101348.88}'
# {"churn_probability":0.86,"will_churn":true,"risk_level":"high"}
```

**Docker**

```bash
docker build -t churn-api .
docker run -p 8000:8000 -e API_KEY=choose-a-secret churn-api
```

If no trained model is present the build trains one, so the image always works.

**Monitoring** (`python -m monitoring.monitor`) simulates six weeks of customer data that gradually drifts
(`src/drift_sim.py`): customers get older and richer (input drift), and younger non-German customers start churning, as if
after a competitor campaign (concept drift). For each week it records accuracy, precision, recall and F1, and runs an
Evidently data-drift report against the training data.

| Week | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 0.820 | 0.730 | 0.623 | 0.553 | 0.519 | 0.518 |
| Columns with drift | 0 | 1 | 2 | 2 | 2 | 2 |

The retrain alert fires when accuracy falls more than 5 points below the baseline (0.818, so below 0.768), which happens in week 1.
Outputs in `monitoring/reports/`: `data_drift_baseline.html`, `data_drift_latest.html`, `model_performance_latest.html`
(interactive Evidently reports), `accuracy_over_time.png`, `performance_log.csv`, `status.json`.
For real traffic, `python -m monitoring.monitor --live` checks the requests logged by the API for input drift
(outcomes are not known immediately, so accuracy cannot be measured live).

**Retraining** (`python -m src.retrain`) reads `status.json` and, if retraining is needed, retrains all four models on the
original data plus the newest labelled data (weighted 3x), evaluates them on a separate recent holdout, registers a new
model version, and makes it the champion. On the simulated data it raised accuracy from 0.529 to 0.724 and F1 from 0.616 to 0.799
(see `monitoring/reports/retrain_log.json`; registry versions 1 then 2).

**Scheduling:** `.github/workflows/retrain.yml` runs monitoring and retraining every Monday at 06:00 UTC (and on demand).
Locally, `python -m src.retrain --loop 3600` repeats the check every hour.

## CI/CD

* `ci.yml`: on every push or pull request: prepare data, EDA, train, run the 13 tests, run the monitoring simulation,
  upload the model and reports, and on `main` build the Docker image.
* `retrain.yml`: weekly monitoring and conditional retraining.
* `python pipeline.py --watch` does the same locally and re-runs when the data or any `src/*.py` file changes.

## Assumptions and limitations (please read)

* **The drift is simulated.** The dataset has no time dimension, so "new data" is the test set with artificial changes. The
  accuracy drop and the recovery after retraining depend on how I designed that simulation; they are not evidence about real
  customers. In production, monitoring would use real labelled outcomes as they arrive.
* The champion is chosen by F1 on the test set, as the task specifies. This slightly favours the winner; cross-validation would
  be a stronger selection method.
* After retraining, the new model is promoted automatically with no extra gate, and its metrics are measured on the recent
  (drifted) holdout, which has a higher churn rate, so they are not comparable with the original table. A real system should
  compare old and new models on the same data before promoting.
* The default 0.5 decision threshold is not tuned to business costs.
* Gender and Geography are used as features. In a real bank this needs a fairness and legal review.
* The Docker image could not be built in the environment where this project was developed (no Docker daemon). Its steps were checked
  by running them manually in a clean folder, but `docker build` itself is untested here; the CI workflow builds it on `main`.
* Re-running `python pipeline.py` resets the whole demo (it retrains from scratch). The MLflow database is not stored in git.

## Reproducibility

Fixed random seeds, stratified split, a single `config.py`, pinned `pandas` and `evidently` versions, feature engineering inside the saved
model pipeline (so the API cannot differ from training), all experiments logged in MLflow, and automated tests plus a quality gate (F1 of at least 0.55).
