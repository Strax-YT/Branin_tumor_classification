"""
AI Safety Decision Engine.

Turns the softmax probabilities the model already produces into a
Prediction Reliability Score and a tiered recommendation. This layer
never touches the model -- it only interprets its output.

The thresholds below (0.5/0.3/0.2 weighting, 80/60 tier cutoffs) are
reasonable defaults, not values fit to labeled "was this reliable"
data -- there's no such dataset. Treat them as a starting point to
tune, not a validated calibration.

Softmax/entropy/gap heuristics also can't reliably tell "this is a
different imaging modality entirely" apart from "this is a genuinely
ambiguous brain MRI" -- both just look like low confidence. The Low
tier message reflects that honestly instead of claiming true
out-of-distribution detection.
"""

import math


def assess_reliability(probabilities, class_names):
    """
    probabilities: iterable of per-class probabilities (sums to ~1)
    class_names: list of class labels, same order as probabilities
    """
    probs = [float(p) for p in probabilities]
    n_classes = len(probs)

    confidence = max(probs)
    entropy = -sum(p * math.log(p + 1e-8) for p in probs)
    max_entropy = math.log(n_classes)
    entropy_score = max(0.0, 1 - entropy / max_entropy) if max_entropy > 0 else 1.0

    sorted_probs = sorted(probs, reverse=True)
    top2_gap = sorted_probs[0] - sorted_probs[1] if n_classes > 1 else sorted_probs[0]

    reliability_score = 100 * (0.5 * confidence + 0.3 * entropy_score + 0.2 * top2_gap)

    if reliability_score >= 80:
        tier, status = "High", "ACCEPTED"
    elif reliability_score >= 60:
        tier, status = "Medium", "AMBIGUOUS"
    else:
        tier, status = "Low", "UNCERTAIN"

    predicted_idx = probs.index(confidence)
    runner_up_idx = sorted(range(n_classes), key=lambda i: probs[i])[-2] if n_classes > 1 else predicted_idx

    return {
        "reliability_score": round(reliability_score, 1),
        "tier": tier,
        "status": status,
        "confidence": confidence,
        "entropy": entropy,
        "top2_gap": top2_gap,
        "predicted_class": class_names[predicted_idx],
        "runner_up_class": class_names[runner_up_idx],
        "runner_up_probability": probs[runner_up_idx],
        "review_recommended": status != "ACCEPTED",
    }


TIER_MESSAGES = {
    "High": "AI result -- high reliability.",
    "Medium": "Review advised -- the model is genuinely torn between two classes.",
    "Low": "Do not rely on this result. It may be an ambiguous scan, poor image "
           "quality, or an unsupported image type -- this heuristic can't tell those apart.",
}
