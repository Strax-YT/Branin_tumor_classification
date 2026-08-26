# NeuroSight

Explainable, safety-aware brain MRI tumor classification. A lightweight CNN classifies MRI scans
into four classes, a FastAPI backend wraps it with Grad-CAM explainability, an AI safety decision
engine, and image quality checks, and a Streamlit frontend ties it together — with persistent
prediction history, tests, CI, Docker, and MLflow model tracking around it.

**For educational and research use only — not a diagnostic tool.**

## What it does

- Classifies a brain MRI scan as **Glioma**, **Meningioma**, **Pituitary tumor**, or **No Tumor**
- Explains every prediction with a **Grad-CAM** attention overlay showing which region of the
  scan drove the decision
- Scores every prediction with a **Safety Decision Engine** (confidence + entropy + top-2
  probability gap → a 0-100 reliability score and a High/Medium/Low tier), flagging low-reliability
  results for human review instead of presenting every output with equal confidence
- Runs an **MRI Image Quality Assessment** (resolution, sharpness, contrast, brightness) before
  analysis, so a poor scan comes with a warning rather than a silently unreliable result
- Keeps a **persistent prediction history** (SQLite) and exposes aggregate metrics via the API
- Tracks the deployed model's provenance (parameters, checksum, accuracy) and registers it in
  **MLflow** so deployments are reproducible and auditable

## Model

| | |
|---|---|
| Architecture | `LiteCNN` — 4-block custom CNN (Conv+BatchNorm+ReLU+MaxPool+Dropout), no pretrained backbone |
| Parameters | 422,788 (~1.6 MB) |
| Input | 128×128 RGB, ImageNet normalization |
| Test accuracy | **91.62%** (1,098 held-out test images), independently verified — see `results/logs/` |
| Training data | 27,705 train / 400 validation / 1,098 test images across the four classes |

Per-class precision: Glioma 95% · Meningioma 80% · No Tumor 99% · Pituitary 94%. Meningioma is
the known weak point (see `results/logs/finetune_comparison.txt` for a documented attempt to fix
it that didn't generalize, and was correctly not shipped).

The model itself is intentionally frozen — everything built around it (safety layer, quality
checks, API, history, tests, MLOps tooling) is what makes this project more than a single
classifier.

## Architecture

```
Streamlit UI  --REST-->  FastAPI (api/)  --uses-->  LiteCNN + Grad-CAM + Safety Engine + Quality
   (app/)                     |                              (app/*.py, models/model2.py)
                               |
                               +--> SQLite (neurosight.db)  -- prediction history
```

The frontend is a thin HTTP client — it never imports torch. All inference, explainability, and
scoring logic lives in the FastAPI service and is reused as-is (not duplicated) from `app/`'s
pure Python modules.

## Tech stack

Python 3.12 · PyTorch (CPU) + torchvision · FastAPI + Uvicorn · Streamlit · Plotly · OpenCV +
Pillow · SQLite · pytest · MLflow · Docker

## Project structure

```
app/                  Streamlit frontend + the reusable model/safety/quality modules
  simple_direct_app.py    the UI
  direct_model_loader.py  loads LiteCNN, runs inference + Grad-CAM
  safety_engine.py        confidence/entropy/gap -> reliability score & tier
  image_quality.py        blur/brightness/contrast/resolution scoring
  model_registry.py       checkpoint metadata + SHA-256 integrity check
api/                  FastAPI backend
  main.py                 endpoints
  service.py               orchestrates the app/ modules for the API
  db.py                    SQLite prediction history
  schemas.py               pydantic request/response models
models/model2.py      LiteCNN architecture + the original training pipeline
artifacts/model/      the deployed checkpoint + metadata.json (name, version, SHA-256, accuracy)
src/                  evaluation, calibration, and fine-tune experiment scripts
tests/                pytest suite (22 tests) + real MRI fixtures
mlops/log_model.py    logs the model + experiments into MLflow
.github/workflows/    CI (runs the test suite on push/PR)
docker-compose.yml, api/Dockerfile, app/Dockerfile
```

## Getting started

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install -r api/requirements.txt
pip install -r app/requirements.txt
pip install -r requirements-dev.txt   # only needed for tests/mlflow
```

**Run the backend** (terminal 1):
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
API docs: http://127.0.0.1:8000/docs

**Run the frontend** (terminal 2):
```powershell
streamlit run app/simple_direct_app.py
```
Opens at http://localhost:8501. It talks to the backend at `http://127.0.0.1:8000/api/v1` by
default — override with the `NEUROSIGHT_API_URL` environment variable if needed.

`neurosight.db` (prediction history) is created automatically on first run.

## API reference

| Endpoint | Description |
|---|---|
| `POST /api/v1/analyze` | Full pipeline: quality check, prediction, Grad-CAM, reliability score |
| `POST /api/v1/validate-image` | Quality check only, no model inference |
| `GET /api/v1/history` | Paginated prediction history (`?limit=&offset=`) |
| `GET /api/v1/metrics` | Aggregate stats (avg confidence, avg inference time, class distribution) |
| `GET /api/v1/model-info` | Model metadata (version, parameters, checksum) |
| `GET /api/v1/health` | Liveness check |

## Testing

```powershell
pytest tests/ -v
```
22 tests against the real model and a live API test client (no mocking) — model correctness,
preprocessing, quality scoring, the safety engine's review-triggering behavior, Grad-CAM output,
and full API round-trips. Runs automatically on push/PR via GitHub Actions.

## MLflow

```powershell
python mlops/log_model.py   # logs the baseline model + a past fine-tune experiment
mlflow ui                   # browse at http://127.0.0.1:5000
```

## Docker

```powershell
docker compose up --build
```
Two services — `ai-service` (FastAPI + model) and `frontend` (Streamlit). SQLite persists on a
named volume. **Note:** the compose config has been validated (`docker compose config`), but a
full build has not been run end-to-end yet — if it fails, check that first.

## Known limitations

- Meningioma precision (80%) is a genuine, unresolved weak point — a calibration fix and a
  fine-tune were both tried and didn't hold up on the test set (see `results/logs/`)
- Dataset versioning uses image counts, not a DVC hash — DVC isn't currently active in this repo
- `docker compose up --build` hasn't been build-verified end-to-end
