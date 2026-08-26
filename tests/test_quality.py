from PIL import Image

from image_quality import assess_quality


def test_assess_quality_returns_expected_keys(sample_image_bytes):
    import io
    image = Image.open(io.BytesIO(sample_image_bytes("glioma")))
    result = assess_quality(image)

    expected_keys = {
        "width", "height", "resolution_ok", "sharpness_score",
        "brightness", "brightness_score", "contrast_score",
        "overall_score", "status",
    }
    assert expected_keys.issubset(result.keys())
    assert 0 <= result["overall_score"] <= 100
    assert result["status"] in {"Suitable", "Marginal", "Poor"}


def test_solid_black_image_scores_low_brightness():
    image = Image.new("RGB", (128, 128), color=(0, 0, 0))
    result = assess_quality(image)
    assert result["brightness"] < 20
    assert result["brightness_score"] < 50


def test_low_resolution_image_flagged():
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    result = assess_quality(image)
    assert result["resolution_ok"] is False
