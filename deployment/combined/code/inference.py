"""SageMaker inference handler serving both models behind one endpoint.

Receives the raw app payload from the API Lambda (see the backend repo's
docs/SAGEMAKER_CONTRACT.md):

    {"memberId": "...", "member": {...}, "history": [timeseries items ...]}

Computes the engineered features here — with the same formulas used by
prepare_longitudinal_dataset.py and prepare_vo2_dataset.py — runs the
readiness classifier and the VO2 forecaster, and returns one combined
prediction in the implementation plan's section 10.1 shape.

Also accepts a pre-engineered feature record ({"instances": [...]}) so the
original per-model test rows still work.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


READINESS_MODEL_FILE = "readiness_model.joblib"
VO2_MODEL_FILE = "vo2_model.joblib"
CONTENT_TYPE = "application/json"
CLASS_NAMES = {0: "REDUCE", 1: "MAINTAIN", 2: "PROGRESS"}
MODEL_VERSION = "readiness-xgb-longitudinal-1.0+vo2-xgb-1.0"

# recoveryStage text -> plan stage number (1..5)
STAGE_NUMBERS = {
    "acute recovery": 1,
    "early restart": 2,
    "building consistency": 3,
    "progression": 4,
    "maintenance": 5,
}

# Workout Type text -> plan activity vocabulary (walk | run | swim | rest)
ACTIVITY_MAP = {
    "walking": "walk",
    "walk": "walk",
    "running": "run",
    "run": "run",
    "jogging": "run",
    "swimming": "swim",
    "swim": "swim",
}


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


def model_fn(model_dir: str) -> dict[str, Any]:
    models = {}
    for key, filename in (("readiness", READINESS_MODEL_FILE), ("vo2", VO2_MODEL_FILE)):
        path = Path(model_dir) / filename
        if not path.exists():
            raise RuntimeError(f"Model file not found: {path}")
        models[key] = joblib.load(path)
    return models


def input_fn(request_body: str | bytes, request_content_type: str) -> dict[str, Any]:
    if not request_content_type.startswith(CONTENT_TYPE):
        raise ValueError(f"Unsupported content type: {request_content_type}. Use {CONTENT_TYPE}.")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    try:
        payload = json.loads(request_body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object.")
    return payload


# ---------------------------------------------------------------------------
# Raw history -> per-session records
# ---------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _first(record: dict, *keys: str) -> Any:
    """First non-None value among alternative field spellings."""
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def _item_timestamp(item: dict) -> str:
    """Full timestamp for ordering (not just the date)."""
    for key in ("createdAt", "timestamp", "recordDate"):
        value = item.get(key)
        if value:
            return str(value)
    sk = str(item.get("sk", ""))
    return sk.split("#", 1)[1] if "#" in sk else ""


def _item_date(item: dict) -> str | None:
    stamp = _item_timestamp(item)
    return stamp[:10] if stamp else None


def build_sessions(history: list[dict]) -> list[dict]:
    """Merge ACTIVITY/READING/CHECKIN items into one record per date, oldest first.

    Handles both the seeded field names (workoutType, durationMin, rpe,
    restingHeartRate, vo2MaxEstimate) and the app-written ones (activityType,
    durationMinutes, perceivedExertion, restingHr, vo2max).
    """
    # Oldest first, so that when a member logs several check-ins or workouts on
    # the same day the NEWEST one wins the merge. The Lambda sends history
    # newest-first, which would otherwise let a stale morning check-in
    # overwrite the one just submitted.
    by_date: dict[str, dict] = {}
    for item in sorted(history or [], key=_item_timestamp):
        date = _item_date(item)
        if not date:
            continue
        session = by_date.setdefault(date, {"date": date})
        item_type = str(item.get("type", "")).upper() or str(item.get("sk", "")).split("#")[0]

        if item_type == "ACTIVITY":
            session["workout_type"] = _first(item, "workoutType", "activityType")
            status = _first(item, "workoutStatus")
            if status is None and "completed" in item:
                status = "Completed" if item.get("completed") else "Missed"
            session["workout_status"] = status
            session["duration"] = _num(_first(item, "durationMin", "durationMinutes"))
            session["distance"] = _num(_first(item, "distanceKm"))
            session["avg_hr"] = _num(_first(item, "avgHeartRate"))
            session["max_hr"] = _num(_first(item, "maxHeartRate"))
            session["calories"] = _num(_first(item, "caloriesBurned"))
            session["rpe"] = _num(_first(item, "rpe", "perceivedExertion"))
        elif item_type == "READING":
            session["resting_hr"] = _num(_first(item, "restingHeartRate", "restingHr"))
            session["hrv"] = _num(_first(item, "hrvMs"))
            session["sleep_hours"] = _num(_first(item, "sleepHours"))
            session["sleep_quality"] = _num(_first(item, "sleepQualityScore"))
            session["vo2"] = _num(_first(item, "vo2MaxEstimate", "vo2max"))
        elif item_type == "CHECKIN":
            session["pain"] = _num(item.get("pain"))
            session["fatigue"] = _num(item.get("fatigue"))

    return [by_date[date] for date in sorted(by_date)]


# ---------------------------------------------------------------------------
# Sessions -> engineered feature row (training formulas)
# ---------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None

def _trend(values: list) -> float:
    """Training's trend(): last - first over the valid values in the window."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    return float(clean[-1] - clean[0])

def _days_between(a: str | None, b: str | None) -> float:
    try:
        return float((datetime.fromisoformat(b) - datetime.fromisoformat(a)).days)
    except (TypeError, ValueError):
        return 0.0


def _last_known(sessions: list[dict], key: str) -> Any:
    for session in reversed(sessions):
        if session.get(key) is not None:
            return session[key]
    return None


def build_feature_row(member: dict, sessions: list[dict]) -> dict[str, Any]:
    context = member.get("recoveryContext") or {}
    current = sessions[-1] if sessions else {}
    prior = sessions[:-1]          # everything before the current session
    window = prior[-3:]            # training used shift(1).rolling(3)

    def prior_vals(key: str) -> list:
        return [s.get(key) for s in window]

    completed_flags = [
        1 if str(s.get("workout_status") or "").strip().casefold() == "completed" else 0
        for s in prior
    ]
    loads = [
        (s.get("duration") or 0) * (s.get("rpe") or 0) if s.get("duration") is not None else None
        for s in prior
    ]
    load_clean = [l for l in loads if l is not None]
    previous = prior[-1] if prior else {}

    current_load = None
    if current.get("duration") is not None and current.get("rpe") is not None:
        current_load = current["duration"] * current["rpe"]

    def change_from_previous(key: str) -> float | None:
        if current.get(key) is not None and previous.get(key) is not None:
            return current[key] - previous[key]
        return None

    row: dict[str, Any] = {
        # ---- member profile / health context (same names as training) ----
        "Gender": member.get("gender"),
        "Age": _num(member.get("age")),
        "Activity Baseline": member.get("activityBaseline"),
        "Recovery Goal": member.get("recoveryGoal"),
        "Recovery Context": _first(context, "recoveryContext"),
        "Event Type": context.get("eventType"),
        "Condition Category": context.get("conditionCategory"),
        "Diagnosis or Event": context.get("diagnosisOrEvent"),
        "Severity": context.get("severity"),
        "Recovery Stage": context.get("recoveryStage"),
        "Mobility Limitation": context.get("mobilityLimitation"),
        "Pain Score": current.get("pain", _num(context.get("painScore"))),
        "Clinician Cleared": context.get("clinicianCleared"),
        "Contraindication Flag": context.get("contraindicationFlag"),
        "VO2 Risk Band": context.get("vo2RiskBand"),
        "Medication Impact": context.get("medicationImpact"),
        # ---- current session ----
        "Workout Type": current.get("workout_type"),
        "Workout Status": current.get("workout_status"),
        "Duration Min": current.get("duration"),
        "Distance Km": current.get("distance"),
        "Avg Heart Rate": current.get("avg_hr"),
        "Max Heart Rate": current.get("max_hr"),
        "Resting Heart Rate": current.get("resting_hr"),
        "HRV ms": current.get("hrv"),
        "Sleep Hours": current.get("sleep_hours"),
        "Sleep Quality Score": current.get("sleep_quality"),
        "Calories Burned": current.get("calories"),
        # Carry the last known VO2 forward: the newest item can be a check-in
        # without a workout, and the member's own last value beats the
        # pipeline imputer's population median.
        "VO2 Max Estimate": _last_known(sessions, "vo2"),
        "RPE 1-10": current.get("rpe"),
        # ---- flags that only exist in the synthetic training data ----
        "RHR Imputed": False,
        "HRV Imputed": False,
        "Sleep Hours Imputed": False,
        "Sleep Quality Imputed": False,
        "Setback Flag": False,
        # ---- longitudinal history features ----
        "session_number": float(len(sessions)),
        "Synthetic Session Number": float(len(sessions)),
        "days_since_previous_session": _days_between(previous.get("date"), current.get("date")) if prior else 0.0,
        "previous_duration": previous.get("duration"),
        "previous_rpe": previous.get("rpe"),
        "previous_vo2": previous.get("vo2"),
        "previous_hrv": previous.get("hrv"),
        "previous_pain": previous.get("pain"),
        "Previous Session Duration": previous.get("duration"),
        "Previous Session RPE": previous.get("rpe"),
        "Previous Session VO2": previous.get("vo2"),
        "Previous Session HRV": previous.get("hrv"),
        "current_training_load": current_load,
        "vo2_change_from_previous": change_from_previous("vo2"),
        "duration_change_from_previous": change_from_previous("duration"),
        "rpe_change_from_previous": change_from_previous("rpe"),
        "hrv_change_from_previous": change_from_previous("hrv"),
        "completion_rate_before_current": float(np.mean(completed_flags)) if completed_flags else 0.0,
        "setback_count_before_current": 0.0,
        "setback_rate_before_current": 0.0,
        "cumulative_training_load_before_current": float(np.sum(load_clean)) if load_clean else 0.0,
        "cumulative_training_load": float(np.sum(load_clean)) if load_clean else 0.0,
        "cumulative_training_minutes": float(np.sum([s.get("duration") or 0 for s in prior])),
        "average_training_load_last_3": _mean([l for l in loads[-3:]]),
        "training_load_trend_last_3": _trend([l for l in loads[-3:]]),
    }

    rolling = {
        "duration": "duration",
        "rpe": "rpe",
        "vo2": "vo2",
        "hrv": "hrv",
        "sleep": "sleep_hours",
        "resting_hr": "resting_hr",
        "pain": "pain",
    }
    for prefix, key in rolling.items():
        row[f"average_{prefix}_last_3"] = _mean(prior_vals(key))
        row[f"{prefix}_trend_last_3"] = _trend(prior_vals(key))

    return row


# ---------------------------------------------------------------------------
# Model outputs -> plan section 10.1 contract
# ---------------------------------------------------------------------------

def _stage_number(stage_text: Any) -> int:
    return STAGE_NUMBERS.get(str(stage_text or "").strip().casefold(), 2)


def _plan_activity(label: str, member: dict, current: dict) -> tuple[str, int, str]:
    """Readiness class -> (activity, duration_minutes, intensity)."""
    last_activity = ACTIVITY_MAP.get(
        str(current.get("workout_type") or "").strip().casefold(), "walk"
    )
    preferred = ACTIVITY_MAP.get(
        str(member.get("activityPreference") or "").strip().casefold(), last_activity
    )
    last_duration = current.get("duration") or 20

    if label == "REDUCE":
        pain = current.get("pain")
        if (pain is not None and pain >= 7) or str(
            current.get("workout_status") or ""
        ).strip().casefold() not in ("completed", ""):
            return "rest", 0, "none"
        return "walk", int(max(10, round(last_duration * 0.6))), "very_low"
    if label == "PROGRESS":
        return preferred, int(min(45, round(last_duration + 5))), "moderate"
    return last_activity, int(min(30, round(last_duration))), "low"


def _top_factors(row: dict[str, Any]) -> list[dict[str, str]]:
    """Transparent signal panel derived from the readiness labelling rules."""
    checks = [
        ("sleep_hours", row.get("Sleep Hours"), lambda v: v >= 6.5),
        ("pain_score", row.get("Pain Score"), lambda v: v <= 3),
        ("perceived_exertion", row.get("RPE 1-10"), lambda v: v <= 7),
        ("hrv", row.get("HRV ms"), lambda v: v >= 35),
        ("vo2_trend", row.get("vo2_trend_last_3"), lambda v: v >= 0),
        ("resting_hr_trend", row.get("resting_hr_trend_last_3"), lambda v: v <= 0),
        ("session_completion_rate", row.get("completion_rate_before_current"), lambda v: v >= 0.8),
    ]
    factors = []
    for name, value, is_positive in checks:
        if value is None:
            continue
        factors.append({"feature": name, "direction": "positive" if is_positive(value) else "negative"})
    return factors[:5]


def predict_fn(payload: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    if "instances" in payload:                       # pre-engineered test rows
        row = dict(payload["instances"][0])
        member, sessions = {}, []
    elif "member" in payload or "history" in payload:
        member = payload.get("member") or {}
        sessions = build_sessions(payload.get("history") or [])
        row = build_feature_row(member, sessions)
    else:                                            # a bare feature record
        row, member, sessions = dict(payload), {}, []

    current = sessions[-1] if sessions else {}
    frame = pd.DataFrame([row])

    # Readiness classifier: reindex to its training columns; the pipeline's
    # imputers fill anything the app data cannot provide.
    readiness_model = models["readiness"]
    readiness_frame = frame.reindex(columns=_required_features(readiness_model))
    label_int = int(readiness_model.predict(readiness_frame)[0])
    probs = readiness_model.predict_proba(readiness_frame)[0]
    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    label = CLASS_NAMES[label_int]

    # VO2 forecaster
    vo2_model = models["vo2"]
    vo2_frame = frame.reindex(columns=_required_features(vo2_model))
    predicted_vo2 = float(vo2_model.predict(vo2_frame)[0])
    current_vo2 = row.get("VO2 Max Estimate")
    current_vo2 = float(current_vo2) if current_vo2 is not None else None
    predicted_change = predicted_vo2 - current_vo2 if current_vo2 is not None else None

    if predicted_change is None:
        vo2_trend_text = "unknown"
    elif predicted_change >= 0.3:
        vo2_trend_text = "improving"
    elif predicted_change <= -0.3:
        vo2_trend_text = "declining"
    else:
        vo2_trend_text = "stable"

    # Days until +1.0 VO2 at the predicted 4-week (28-day) rate, clamped 7..90
    if predicted_change is not None and predicted_change > 0.05:
        days_to_milestone = int(np.clip(round(28.0 / predicted_change), 7, 90))
    else:
        days_to_milestone = 90

    activity, duration_minutes, intensity = _plan_activity(label, member, current)

    prediction = {
        # ---- plan section 10.1 contract ----
        "recovery_score": round(100 * (0.25 * probabilities["REDUCE"]
                                       + 0.60 * probabilities["MAINTAIN"]
                                       + 0.95 * probabilities["PROGRESS"]), 1),
        "readiness_score": round(100 * (1.0 - probabilities["REDUCE"]), 1),
        "recovery_stage": _stage_number(row.get("Recovery Stage")),
        "recovery_trend": vo2_trend_text,
        "setback_probability": round(probabilities["REDUCE"], 4),
        "recommended_activity": activity,
        "duration_minutes": duration_minutes,
        "intensity": intensity,
        "predicted_days_to_milestone": days_to_milestone,
        "confidence": round(float(max(probabilities.values())), 4),
        "top_factors": _top_factors(row),
        "model_version": MODEL_VERSION,
        # ---- extra detail for the dashboard / demo ----
        "readiness": label,
        "readiness_label": label_int,
        "probabilities": probabilities,
        "current_vo2": current_vo2,
        "predicted_vo2_4_weeks": round(predicted_vo2, 2),
        "predicted_vo2_change": round(predicted_change, 2) if predicted_change is not None else None,
        "sessions_used": len(sessions),
    }
    return prediction


def output_fn(prediction: dict[str, Any], response_content_type: str) -> str:
    if not response_content_type.startswith(CONTENT_TYPE):
        raise ValueError(f"Unsupported response content type: {response_content_type}. Use {CONTENT_TYPE}.")
    return json.dumps(_json_safe(prediction))
