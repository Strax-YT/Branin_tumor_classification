"""Pydantic response models for the NeuroSight AI backend."""

from typing import Optional

from pydantic import BaseModel


class QualityAssessment(BaseModel):
    width: int
    height: int
    resolution_ok: bool
    sharpness_score: float
    brightness: float
    brightness_score: float
    contrast_score: float
    overall_score: float
    status: str


class UncertaintyInfo(BaseModel):
    entropy: float
    top_two_gap: float
    reliability_score: float
    tier: str
    message: str
    runner_up_class: str
    runner_up_probability: float


class QualitySummary(BaseModel):
    score: float
    status: str


class ModelInfo(BaseModel):
    name: str
    version: str
    parameters: Optional[int] = None


class AnalyzeResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    uncertainty: UncertaintyInfo
    quality: QualitySummary
    review_recommended: bool
    attention_overlay_png_b64: Optional[str] = None
    model: ModelInfo
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_predictions: int
    average_inference_time_ms: float
    class_distribution: dict[str, int]
    review_recommended_count: int
    poor_quality_count: int
    note: str = "In-memory only, resets on restart. Persistent history is a later step."
