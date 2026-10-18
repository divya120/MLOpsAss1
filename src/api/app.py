"""Production FastAPI Serving Application for Cardiovascular Risk Prediction.

Features:
- Validated Pydantic v2 ingestion
- Real-time and batch scoring
- Sub-millisecond latency tracking
- Native Prometheus metrics (/metrics)
- Structured JSON access logging
- Production health and readiness probes (/health)
"""

import sys
import time
import logging
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import Response, HTMLResponse
from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from src.models.predict import load_model, predict_single, predict_batch, MODEL_PATH
from src.api.schemas import (
    PatientData,
    BatchPatientData,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)
from src.api.middleware import StructuredLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api")

# Prometheus Metrics Definitions
HTTP_REQUESTS_TOTAL = Counter(
    "heart_disease_api_requests_total",
    "Total count of HTTP requests received",
    ["method", "endpoint", "status_code"]
)
PREDICTIONS_TOTAL = Counter(
    "heart_disease_predictions_total",
    "Total predictions broken down by clinical risk classification",
    ["prediction_class", "diagnosis_label"]
)
INFERENCE_LATENCY_SECONDS = Histogram(
    "heart_disease_inference_latency_seconds",
    "Time spent executing model pipeline inference",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
DISEASE_PROBABILITY_SUMMARY = Summary(
    "heart_disease_risk_probability",
    "Estimated probability of positive disease diagnosis"
)
MODEL_STATUS_GAUGE = Gauge(
    "heart_disease_model_loaded",
    "Status indicator showing whether machine learning pipeline is loaded (1 = Loaded, 0 = Failed)"
)

# Application Lifecycle
START_TIME = time.time()
PREDICTIONS_COUNTER = 0

app = FastAPI(
    title="Heart Disease UCI Risk Prediction API",
    description=(
        "Production-grade MLOps model serving API with integrated feature transformation, "
        "inference logging, Prometheus monitoring, and health probes."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Attach custom structured logging middleware
app.add_middleware(StructuredLoggingMiddleware)


@app.on_event("startup")
def startup_event():
    """Load model pipeline at application startup and register status metric."""
    logger.info("Initializing API and loading machine learning pipeline...")
    try:
        load_model(MODEL_PATH)
        MODEL_STATUS_GAUGE.set(1)
        logger.info("Model pipeline successfully loaded into memory.")
    except Exception as e:
        MODEL_STATUS_GAUGE.set(0)
        logger.error(f"Failed to load model pipeline during startup: {e}")


@app.get("/app", response_class=HTMLResponse, tags=["Web Application"])
def web_app():
    """Interactive diagnostic web application for clinical risk assessment."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Cardiac Risk Assessment Web App</h1><p>Template not found.</p>")


@app.get("/", tags=["General"])
def root(request: Request):
    """Root endpoint providing service metadata or redirecting to web app if accessed via browser."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return web_app()

    return {
        "service": "Heart Disease UCI Risk Prediction API",
        "version": "1.0.0",
        "status": "online",
        "web_app": "/app",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Liveness and readiness probe for Kubernetes and Docker health checks."""
    try:
        pipeline = load_model(MODEL_PATH)
        model_loaded = pipeline is not None
        status_str = "healthy" if model_loaded else "degraded"
    except Exception:
        model_loaded = False
        status_str = "unhealthy"

    uptime = round(time.time() - START_TIME, 2)
    return HealthResponse(
        status=status_str,
        model_loaded=model_loaded,
        version="1.0.0",
        uptime_seconds=uptime,
        total_predictions_served=PREDICTIONS_COUNTER
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics scrape target."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientData):
    """Predict heart disease presence/risk for an individual patient.
    
    Accepts 13 clinical biomarkers, transforms them through the scikit-learn
    pipeline, and returns binary prediction with calibrated risk confidence.
    """
    global PREDICTIONS_COUNTER
    start_time = time.perf_counter()

    try:
        patient_dict = patient.model_dump()
        result = predict_single(patient_dict, MODEL_PATH)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        latency_sec = latency_ms / 1000.0

        # Prometheus metrics recording
        PREDICTIONS_COUNTER += 1
        PREDICTIONS_TOTAL.labels(
            prediction_class=str(result["prediction"]),
            diagnosis_label=result["label"]
        ).inc()
        INFERENCE_LATENCY_SECONDS.observe(latency_sec)
        DISEASE_PROBABILITY_SUMMARY.observe(result["probability_disease"])
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="200").inc()

        return PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            probability_disease=result["probability_disease"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            latency_ms=latency_ms,
            model_version="1.0.0"
        )
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="500").inc()
        logger.error(f"Inference error during /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(batch: BatchPatientData):
    """Predict cardiovascular risk for a batch of multiple patient records."""
    global PREDICTIONS_COUNTER
    start_time = time.perf_counter()

    try:
        records = [p.model_dump() for p in batch.patients]
        results = predict_batch(records, MODEL_PATH)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        avg_latency_ms = round(total_latency_ms / len(records), 2)

        formatted_predictions = []
        for r in results:
            PREDICTIONS_COUNTER += 1
            PREDICTIONS_TOTAL.labels(
                prediction_class=str(r["prediction"]),
                diagnosis_label=r["label"]
            ).inc()
            DISEASE_PROBABILITY_SUMMARY.observe(r["probability_disease"])

            formatted_predictions.append(
                PredictionResponse(
                    prediction=r["prediction"],
                    label=r["label"],
                    probability_disease=r["probability_disease"],
                    confidence=r["confidence"],
                    probabilities=r["probabilities"],
                    latency_ms=avg_latency_ms,
                    model_version="1.0.0"
                )
            )

        INFERENCE_LATENCY_SECONDS.observe(total_latency_ms / 1000.0)
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status_code="200").inc()

        return BatchPredictionResponse(
            predictions=formatted_predictions,
            total_records=len(records),
            total_latency_ms=total_latency_ms
        )
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status_code="500").inc()
        logger.error(f"Error during /predict/batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch execution failure: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
