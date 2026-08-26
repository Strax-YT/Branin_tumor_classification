"""
NeuroSight AI backend -- FastAPI service wrapping the unchanged LiteCNN model,
Grad-CAM explainability, safety decision engine, and image quality assessment.

Run with:  uvicorn api.main:app --host 127.0.0.1 --port 8000
Docs at:   http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api import db, service
from api.schemas import (
    AnalyzeResponse,
    HealthResponse,
    HistoryResponse,
    MetricsResponse,
    QualityAssessment,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.get_model()  # load once at startup, not per-request
    db.init_db()
    yield


app = FastAPI(
    title="NeuroSight AI Backend",
    description="Brain MRI classification, explainability, and safety layer around LiteCNN.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": service.get_model() is not None}


@app.get("/api/v1/model-info")
def model_info():
    metadata = service.load_model_metadata()
    if metadata is None:
        raise HTTPException(status_code=404, detail="metadata.json not found.")
    return metadata


@app.get("/api/v1/metrics", response_model=MetricsResponse)
def metrics():
    return db.get_metrics()


@app.get("/api/v1/history", response_model=HistoryResponse)
def history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return {"items": db.get_history(limit, offset), "limit": limit, "offset": offset}


@app.post("/api/v1/validate-image", response_model=QualityAssessment)
async def validate_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        quality = service.validate_and_assess_quality(file_bytes)
    except service.InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return quality


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        result = service.run_analysis(file_bytes)
    except service.InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    db.record_prediction({
        "predicted_class": result["prediction"],
        "confidence": result["confidence"],
        "entropy": result["uncertainty"]["entropy"],
        "top_two_gap": result["uncertainty"]["top_two_gap"],
        "reliability_score": result["uncertainty"]["reliability_score"],
        "reliability_tier": result["uncertainty"]["tier"],
        "review_recommended": result["review_recommended"],
        "quality_score": result["quality"]["score"],
        "quality_status": result["quality"]["status"],
        "inference_time_ms": result["inference_time_ms"],
        "model_version": result["model"]["version"],
    })

    return result
