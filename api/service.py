"""
Inference orchestration for the FastAPI backend.

Everything here reuses the existing, framework-agnostic modules in app/
(direct_model_loader, safety_engine, image_quality, model_registry)
unchanged -- this file only wires them together and adds the HTTP-facing
concerns (bytes in, JSON-friendly dict out, base64-encoded overlay image).
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from direct_model_loader import load_direct_model, ExplainabilityWrapper  # noqa: E402
from safety_engine import assess_reliability, TIER_MESSAGES  # noqa: E402
from image_quality import assess_quality  # noqa: E402
from model_registry import load_model_metadata  # noqa: E402

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None
_explainer = None


def get_model():
    """Load the model once per process (mirrors @st.cache_resource in the old app)."""
    global _model, _explainer
    if _model is None:
        _model = load_direct_model()
        if _model is not None:
            _explainer = ExplainabilityWrapper(_model, CLASS_NAMES)
    return _model


class InvalidImageError(ValueError):
    pass


def _decode_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except Exception as e:
        raise InvalidImageError(f"Could not decode file as an image: {e}") from e
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def validate_and_assess_quality(file_bytes):
    """Used by POST /validate-image -- decode + quality check only, no model."""
    image = _decode_image(file_bytes)
    return assess_quality(image)


def _encode_overlay(overlay_array):
    overlay_image = Image.fromarray(overlay_array.astype(np.uint8))
    buf = io.BytesIO()
    overlay_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_analysis(file_bytes):
    """Used by POST /analyze -- the full pipeline. Returns a JSON-ready dict."""
    model = get_model()
    if model is None:
        raise RuntimeError("Model is not loaded.")

    start = time.time()
    image = _decode_image(file_bytes)
    quality = assess_quality(image)

    input_tensor = _transform(image).unsqueeze(0)
    result = _explainer.predict_with_explanations(input_tensor)
    probabilities = result["probabilities"]
    reliability = assess_reliability(probabilities, CLASS_NAMES)

    overlay_b64 = None
    if result["attention_map"] is not None:
        overlay = _explainer.create_attention_overlay(image, result["attention_map"])
        overlay_b64 = _encode_overlay(overlay)

    metadata = load_model_metadata() or {}
    inference_time_ms = (time.time() - start) * 1000

    return {
        "prediction": CLASS_NAMES[result["predicted_class"]],
        "confidence": float(result["confidence"]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
        "uncertainty": {
            "entropy": reliability["entropy"],
            "top_two_gap": reliability["top2_gap"],
            "reliability_score": reliability["reliability_score"],
            "tier": reliability["tier"],
            "message": TIER_MESSAGES[reliability["tier"]],
            "runner_up_class": reliability["runner_up_class"],
            "runner_up_probability": reliability["runner_up_probability"],
        },
        "quality": {"score": quality["overall_score"], "status": quality["status"]},
        "review_recommended": reliability["review_recommended"],
        "attention_overlay_png_b64": overlay_b64,
        "model": {
            "name": metadata.get("model_name", "LiteCNN"),
            "version": metadata.get("model_version", "unknown"),
            "parameters": metadata.get("parameters"),
        },
        "inference_time_ms": round(inference_time_ms, 1),
    }
