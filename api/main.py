"""
NeuroSight AI backend -- FastAPI service wrapping the unchanged LiteCNN model,
Grad-CAM explainability, safety decision engine, and image quality assessment.

Run with:  uvicorn api.main:app --host 127.0.0.1 --port 8000
Docs at:   http://127.0.0.1:8000/docs
"""

from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from api import service
from api.schemas import (
    AnalyzeResponse,
    HealthResponse,
    MetricsResponse,
    QualityAssessment,
)

_metrics = {
    "total_predictions": 0,
    "total_inference_time_ms": 0.0,
    "class_distribution": Counter(),
    "review_recommended_count": 0,
    "poor_quality_count": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.get_model()  # load once at startup, not per-request
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
    total = _metrics["total_predictions"]
    avg_time = _metrics["total_inference_time_ms"] / total if total else 0.0
    return {
        "total_predictions": total,
        "average_inference_time_ms": round(avg_time, 1),
        "class_distribution": dict(_metrics["class_distribution"]),
        "review_recommended_count": _metrics["review_recommended_count"],
        "poor_quality_count": _metrics["poor_quality_count"],
    }


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

    _metrics["total_predictions"] += 1
    _metrics["total_inference_time_ms"] += result["inference_time_ms"]
    _metrics["class_distribution"][result["prediction"]] += 1
    if result["review_recommended"]:
        _metrics["review_recommended_count"] += 1
    if result["quality"]["status"] == "Poor":
        _metrics["poor_quality_count"] += 1

    return result
