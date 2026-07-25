"""SageMaker inference handler for the longitudinal readiness model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_FILENAME = "model_file.joblib"
CONTENT_TYPE = "application/json"
CLASS_NAMES = {0: "REDUCE", 1: "MAINTAIN", 2: "PROGRESS"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _required_features(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return []


def model_fn(model_dir: str) -> Any:
    model_path = Path(model_dir) / MODEL_FILENAME
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Model loading failure: {exc}") from exc


def input_fn(request_body: str | bytes, request_content_type: str) -> pd.DataFrame:
    if request_content_type != CONTENT_TYPE:
        raise ValueError(f"Unsupported content type: {request_content_type}. Use {CONTENT_TYPE}.")
    try:
        if isinstance(request_body, bytes):
            request_body = request_body.decode("utf-8")
        payload = json.loads(request_body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(payload, dict) and "instances" in payload:
        records = payload["instances"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError("Input must be a JSON object or an object with an 'instances' list.")

    if not records:
        raise ValueError("Empty input: provide at least one feature record.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Each instance must be a JSON feature object.")
    return pd.DataFrame.from_records(records)


def predict_fn(input_data: pd.DataFrame, model: Any) -> dict[str, Any]:
    required = _required_features(model)
    missing = [feature for feature in required if feature not in input_data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    try:
        features = input_data[required] if required else input_data
        labels = model.predict(features)
        probabilities = model.predict_proba(features)
    except Exception as exc:
        raise RuntimeError(f"Prediction failure: {exc}") from exc

    predictions: list[dict[str, Any]] = []
    for label, probability_row in zip(labels, probabilities):
        label_int = int(label)
        probability_map = {
            CLASS_NAMES[index]: float(probability_row[index])
            for index in range(len(CLASS_NAMES))
        }
        predictions.append(
            {
                "readiness_label": label_int,
                "readiness": CLASS_NAMES[label_int],
                "confidence": float(max(probability_map.values())),
                "probabilities": probability_map,
            }
        )
    if len(predictions) == 1:
        return predictions[0]
    return {"predictions": predictions}


def output_fn(prediction: dict[str, Any], response_content_type: str) -> str:
    if response_content_type != CONTENT_TYPE:
        raise ValueError(f"Unsupported response content type: {response_content_type}. Use {CONTENT_TYPE}.")
    return json.dumps(_json_safe(prediction))
