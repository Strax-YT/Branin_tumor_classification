"""
MRI Image Quality Assessment -- runs before the image reaches LiteCNN.

Advisory, not blocking: a poor score is surfaced as a warning so the
user can make an informed call, but never disables analysis outright.
A slightly soft or dim scan may still be perfectly analyzable.

The sharpness/brightness/contrast scales below are heuristic (common
blur-detection rules of thumb), not tuned against this project's own
dataset -- there's no labeled "good/bad scan" set to calibrate against.
"""

import cv2
import numpy as np

MIN_DIMENSION = 64


def assess_quality(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    width, height = pil_image.size

    resolution_ok = min(width, height) >= MIN_DIMENSION

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(100.0, sharpness / 5.0)

    brightness = float(gray.mean())
    brightness_score = max(0.0, 100.0 - min(100.0, abs(brightness - 128.0) / 1.28))

    contrast = float(gray.std())
    contrast_score = min(100.0, contrast / 0.6)

    resolution_score = 100.0 if resolution_ok else 40.0
    overall_score = (sharpness_score + brightness_score + contrast_score + resolution_score) / 4.0

    if overall_score >= 70:
        status = "Suitable"
    elif overall_score >= 45:
        status = "Marginal"
    else:
        status = "Poor"

    return {
        "width": width,
        "height": height,
        "resolution_ok": resolution_ok,
        "sharpness_score": round(sharpness_score, 1),
        "brightness": round(brightness, 1),
        "brightness_score": round(brightness_score, 1),
        "contrast_score": round(contrast_score, 1),
        "overall_score": round(overall_score, 1),
        "status": status,
    }
