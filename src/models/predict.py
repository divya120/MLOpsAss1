"""Model inference service module for Heart Disease Risk Prediction.

Loads the fitted pipeline (preprocessing + estimator) and serves predictions
for individual dictionaries, batch records, or pandas DataFrames.
"""

import os
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Union, Dict, List, Any
import joblib
import pandas as pd
import numpy as np

from src.features.build_features import ALL_FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("predict")

MODEL_PATH = Path("models/model_pipeline.joblib")
_MODEL = None


def load_model(model_path: Union[str, Path] = MODEL_PATH):
    """Load and cache the trained pipeline."""
    global _MODEL
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fitted model artifact not found at {path.resolve()}. "
            f"Ensure training has been run via 'python src/models/train.py'."
        )
    if _MODEL is None:
        logger.info(f"Loading production pipeline from {path}")
        _MODEL = joblib.load(path)
    return _MODEL


def predict_single(data: Dict[str, Any], model_path: Union[str, Path] = MODEL_PATH) -> Dict[str, Any]:
    """Execute prediction on a single patient record.
    
    Args:
        data: Dictionary of 13 clinical features.
        model_path: Path to serialized joblib model pipeline.
        
    Returns:
        Structured prediction dictionary with class, label, probability, confidence.
    """
    pipeline = load_model(model_path)
    df = pd.DataFrame([data])

    # Ensure all expected feature columns are present
    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[ALL_FEATURE_COLUMNS]

    # Predict
    pred = int(pipeline.predict(df)[0])
    probas = pipeline.predict_proba(df)[0]
    prob_disease = float(probas[1])
    confidence = float(np.max(probas))

    label = "Heart Disease Risk Detected" if pred == 1 else "No Heart Disease Detected"

    return {
        "prediction": pred,
        "label": label,
        "probability_disease": round(prob_disease, 4),
        "confidence": round(confidence, 4),
        "probabilities": {
            "absence": round(float(probas[0]), 4),
            "presence": round(float(probas[1]), 4)
        }
    }


def predict_batch(records: List[Dict[str, Any]], model_path: Union[str, Path] = MODEL_PATH) -> List[Dict[str, Any]]:
    """Execute batch predictions on a list of patient records."""
    pipeline = load_model(model_path)
    df = pd.DataFrame(records)

    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[ALL_FEATURE_COLUMNS]

    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)

    results = []
    for pred, prob in zip(preds, probas):
        p_class = int(pred)
        prob_disease = float(prob[1])
        results.append({
            "prediction": p_class,
            "label": "Heart Disease Risk Detected" if p_class == 1 else "No Heart Disease Detected",
            "probability_disease": round(prob_disease, 4),
            "confidence": round(float(np.max(prob)), 4),
            "probabilities": {
                "absence": round(float(prob[0]), 4),
                "presence": round(float(prob[1]), 4)
            }
        })
    return results


if __name__ == "__main__":
    sample_input = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 3.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 0.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 0.0,
        "ca": 0.0,
        "thal": 1.0
    }
    print(json.dumps(predict_single(sample_input), indent=2))
