from safety_engine import assess_reliability

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]


def test_confident_prediction_is_high_reliability():
    probabilities = [0.97, 0.01, 0.01, 0.01]
    result = assess_reliability(probabilities, CLASS_NAMES)

    assert result["tier"] == "High"
    assert result["status"] == "ACCEPTED"
    assert result["review_recommended"] is False
    assert result["predicted_class"] == "Glioma Tumor"


def test_ambiguous_prediction_triggers_review():
    probabilities = [0.26, 0.25, 0.25, 0.24]
    result = assess_reliability(probabilities, CLASS_NAMES)

    assert result["tier"] == "Low"
    assert result["review_recommended"] is True


def test_runner_up_class_is_second_highest_probability():
    probabilities = [0.5, 0.05, 0.35, 0.10]
    result = assess_reliability(probabilities, CLASS_NAMES)

    assert result["predicted_class"] == "Glioma Tumor"
    assert result["runner_up_class"] == "No Tumor"
    assert result["runner_up_probability"] == 0.35


def test_reliability_score_in_valid_range():
    result = assess_reliability([0.4, 0.3, 0.2, 0.1], CLASS_NAMES)
    assert 0 <= result["reliability_score"] <= 100
