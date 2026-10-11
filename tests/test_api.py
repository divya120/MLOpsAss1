"""Integration tests for the FastAPI model serving endpoints and Prometheus metrics."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


@pytest.fixture
def sample_patient_json():
    return {
        "age": 60.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 140.0,
        "chol": 260.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 120.0,
        "exang": 1.0,
        "oldpeak": 2.0,
        "slope": 2.0,
        "ca": 2.0,
        "thal": 7.0
    }


def test_root_endpoint():
    """Verify root endpoint provides API details and status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint():
    """Verify health endpoint reports model loaded and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


def test_metrics_endpoint():
    """Verify Prometheus scrape endpoint exposes custom metric names."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "heart_disease_api_requests_total" in content
    assert "heart_disease_predictions_total" in content


def test_predict_endpoint_valid(sample_patient_json):
    """Verify /predict returns valid prediction, confidence, and latency."""
    response = client.post("/predict", json=sample_patient_json)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in (0, 1)
    assert isinstance(data["label"], str)
    assert 0.0 <= data["probability_disease"] <= 1.0
    assert 0.5 <= data["confidence"] <= 1.0
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0


def test_predict_endpoint_validation_error():
    """Verify invalid physiological values return HTTP 422 with validation errors."""
    invalid_payload = {
        "age": -10.0,  # Invalid: negative age
        "sex": 5.0,    # Invalid: not binary
        "cp": 1.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_batch_predict_endpoint(sample_patient_json):
    """Verify batch inference scores all records and aggregates latency."""
    batch_payload = {
        "patients": [sample_patient_json, sample_patient_json]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["prediction"] in (0, 1)
