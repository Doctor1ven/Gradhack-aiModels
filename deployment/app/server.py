"""Reusable SageMaker inference server for the packaged model archives."""

from __future__ import annotations

import importlib.util
import os
import traceback
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, Response, request


MODEL_DIR = Path(os.getenv("SAGEMAKER_MODEL_DIR", "/opt/ml/model"))
INFERENCE_MODULE_PATH = MODEL_DIR / "code" / "inference.py"
DEFAULT_CONTENT_TYPE = "application/json"

app = Flask(__name__)
_load_lock = Lock()
_inference_module: Any | None = None
_model: Any | None = None
_load_error: str | None = None


def _load() -> tuple[Any, Any]:
    global _inference_module, _model, _load_error
    if _inference_module is not None and _model is not None:
        return _inference_module, _model

    with _load_lock:
        if _inference_module is not None and _model is not None:
            return _inference_module, _model
        try:
            if not INFERENCE_MODULE_PATH.exists():
                raise FileNotFoundError(f"Missing inference module: {INFERENCE_MODULE_PATH}")
            spec = importlib.util.spec_from_file_location("model_inference", INFERENCE_MODULE_PATH)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not import inference module: {INFERENCE_MODULE_PATH}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.model_fn(str(MODEL_DIR))
            _inference_module = module
            _model = model
            _load_error = None
            return module, model
        except Exception:
            _load_error = traceback.format_exc()
            raise


def _content_type() -> str:
    raw = request.headers.get("Content-Type", DEFAULT_CONTENT_TYPE)
    return DEFAULT_CONTENT_TYPE if raw.startswith(DEFAULT_CONTENT_TYPE) else raw


def _accept() -> str:
    raw = request.headers.get("Accept", DEFAULT_CONTENT_TYPE)
    if raw in {"*/*", ""} or raw.startswith(DEFAULT_CONTENT_TYPE):
        return DEFAULT_CONTENT_TYPE
    return raw


@app.get("/ping")
def ping() -> Response:
    try:
        _load()
        return Response(response="ok", status=200, mimetype="text/plain")
    except Exception:
        body = _load_error or traceback.format_exc()
        return Response(response=body, status=500, mimetype="text/plain")


@app.post("/invocations")
def invocations() -> Response:
    try:
        module, model = _load()
        body = request.get_data()
        input_data = module.input_fn(body, _content_type())
        prediction = module.predict_fn(input_data, model)
        output = module.output_fn(prediction, _accept())
        return Response(response=output, status=200, mimetype=DEFAULT_CONTENT_TYPE)
    except ValueError as exc:
        return Response(response=str(exc), status=400, mimetype="text/plain")
    except Exception:
        return Response(response=traceback.format_exc(), status=500, mimetype="text/plain")
