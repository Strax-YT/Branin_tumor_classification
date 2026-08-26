def test_health_reports_model_loaded(api_client):
    resp = api_client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_matches_metadata(api_client):
    resp = api_client.get("/api/v1/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "LiteCNN"
    assert body["parameters"] == 422788


def test_analyze_returns_valid_prediction(api_client, sample_image_bytes):
    files = {"file": ("glioma.png", sample_image_bytes("glioma"), "image/png")}
    resp = api_client.post("/api/v1/analyze", files=files)
    assert resp.status_code == 200

    body = resp.json()
    assert body["prediction"] in {"Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert "quality" in body
    assert "uncertainty" in body


def test_analyze_rejects_corrupted_image(api_client, corrupted_image_bytes):
    files = {"file": ("garbage.png", corrupted_image_bytes, "image/png")}
    resp = api_client.post("/api/v1/analyze", files=files)
    assert resp.status_code == 400


def test_validate_image_returns_quality_report(api_client, sample_image_bytes):
    files = {"file": ("glioma.png", sample_image_bytes("glioma"), "image/png")}
    resp = api_client.post("/api/v1/validate-image", files=files)
    assert resp.status_code == 200

    body = resp.json()
    assert "overall_score" in body
    assert "status" in body


def test_history_and_metrics_reflect_analyze_call(api_client, sample_image_bytes):
    files = {"file": ("glioma.png", sample_image_bytes("glioma"), "image/png")}
    analyze_resp = api_client.post("/api/v1/analyze", files=files)
    assert analyze_resp.status_code == 200

    history_resp = api_client.get("/api/v1/history")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history["items"]) == 1
    assert history["items"][0]["predicted_class"] == analyze_resp.json()["prediction"]

    metrics_resp = api_client.get("/api/v1/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["total_predictions"] == 1
