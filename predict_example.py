"""Run one example prediction with the saved Recovery Readiness pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("model_outputs/recovery_readiness_pipeline.joblib")
TEST_PATH = Path("prepared_data/test.csv")
TARGET_COLUMN = "readiness_label"
CLASS_NAMES = ["REDUCE", "MAINTAIN", "PROGRESS"]
SECTION_WIDTH = 49
LABEL_WIDTH = 30

# Each entry is: (CSV column name, display label, optional unit).
# Columns that are not present in the input are simply skipped.
INPUT_GROUPS = {
    "PERSON": [
        ("Age", "Age", None),
        ("Gender", "Gender", None),
        ("Activity Baseline", "Activity Baseline", None),
    ],
    "WORKOUT": [
        ("Workout Type", "Workout Type", None),
        ("Workout Status", "Workout Status", None),
        ("Duration Min", "Duration", "min"),
        ("Distance Km", "Distance", "km"),
        ("Calories Burned", "Calories", "kcal"),
        ("RPE 1-10", "RPE", None),
    ],
    "PHYSIOLOGY": [
        ("Avg Heart Rate", "Average HR", "bpm"),
        ("Max Heart Rate", "Maximum HR", "bpm"),
        ("Resting Heart Rate", "Resting HR", "bpm"),
        ("HRV ms", "HRV", "ms"),
        ("Sleep Hours", "Sleep", "hrs"),
        ("Sleep Quality Score", "Sleep Quality", None),
        ("VO2 Max Estimate", "VO2 Max", "ml/kg/min"),
        ("RHR Imputed", "Resting HR Imputed", None),
        ("HRV Imputed", "HRV Imputed", None),
        ("Sleep Hours Imputed", "Sleep Hours Imputed", None),
        ("Sleep Quality Imputed", "Sleep Quality Imputed", None),
    ],
    "RECOVERY": [
        ("Recovery Goal", "Recovery Goal", None),
        ("Recovery Context", "Recovery Context", None),
        ("Event Type", "Event Type", None),
        ("Condition Category", "Condition", None),
        ("Diagnosis or Event", "Diagnosis / Event", None),
        ("Severity", "Severity", None),
        ("Recovery Stage", "Recovery Stage", None),
        ("Mobility Limitation", "Mobility Limitation", None),
        ("Pain Score", "Pain Score", None),
        ("VO2 Risk Band", "VO2 Risk Band", None),
    ],
    "SAFETY": [
        ("Clinician Cleared", "Clinician Cleared", None),
        ("Contraindication Flag", "Contraindication Flag", None),
        ("Medication Impact", "Medication Impact", None),
    ],
    "DERIVED FEATURES": [
        ("Workout Load", "Workout Load", None),
        ("Days Since Event", "Days Since Event", "days"),
        ("Heart Rate Reserve Used", "Heart Rate Reserve", "bpm"),
        ("Average Speed KmH", "Speed", "km/h"),
        (
            "Any Physiological Value Imputed",
            "Any Value Imputed",
            None,
        ),
    ],
    "SESSION HISTORY": [
        ("Setback Flag", "Current Setback", None),
        ("Previous Session Duration", "Previous Duration", "min"),
        ("Previous Session RPE", "Previous RPE", None),
        ("Previous Session VO2", "Previous VO2 Max", "ml/kg/min"),
        ("Previous Session HRV", "Previous HRV", "ms"),
        ("Synthetic Session Number", "Synthetic Session", None),
        ("session_number", "Session Number", None),
        ("days_since_previous_session", "Days Since Previous", "days"),
        ("current_training_load", "Current Training Load", None),
        ("vo2_change_from_previous", "VO2 Change", "ml/kg/min"),
        ("duration_change_from_previous", "Duration Change", "min"),
        ("rpe_change_from_previous", "RPE Change", None),
        ("hrv_change_from_previous", "HRV Change", "ms"),
    ],
    "RECENT AVERAGES": [
        ("average_vo2_last_3", "Average VO2 (Last 3)", "ml/kg/min"),
        ("average_duration_last_3", "Avg Duration (Last 3)", "min"),
        ("average_rpe_last_3", "Average RPE (Last 3)", None),
        ("average_hrv_last_3", "Average HRV (Last 3)", "ms"),
        ("average_sleep_last_3", "Average Sleep (Last 3)", "hrs"),
        ("average_resting_hr_last_3", "Avg Resting HR (Last 3)", "bpm"),
        ("average_pain_last_3", "Average Pain (Last 3)", None),
        ("average_training_load_last_3", "Avg Load (Last 3)", None),
    ],
    "TRENDS & CUMULATIVE": [
        ("vo2_trend_last_3", "VO2 Trend", None),
        ("duration_trend_last_3", "Duration Trend", None),
        ("rpe_trend_last_3", "RPE Trend", None),
        ("hrv_trend_last_3", "HRV Trend", None),
        ("sleep_trend_last_3", "Sleep Trend", None),
        ("resting_hr_trend_last_3", "Resting HR Trend", None),
        ("pain_trend_last_3", "Pain Trend", None),
        ("completion_rate_before_current", "Previous Completion Rate", None),
        ("setback_rate_before_current", "Previous Setback Rate", None),
        ("cumulative_training_minutes", "Cumulative Training", "min"),
        ("cumulative_training_load", "Cumulative Load", None),
    ],
}


def format_value(value: object, unit: str | None = None) -> str:
    """Format one model input value for presentation."""
    if pd.isna(value):
        rendered = "Missing"
    elif isinstance(value, (bool, np.bool_)):
        rendered = "Yes" if value else "No"
    elif isinstance(value, float):
        rendered = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        rendered = str(value)

    return f"{rendered} {unit}" if unit and rendered != "Missing" else rendered


def print_heading(title: str) -> None:
    print("=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


def print_model_input(features: pd.DataFrame) -> None:
    """Print every supplied feature in audience-friendly logical groups."""
    row = features.iloc[0]
    displayed_columns: set[str] = set()

    print_heading("MODEL INPUT")
    for group_name, fields in INPUT_GROUPS.items():
        available_fields = [field for field in fields if field[0] in row.index]
        if not available_fields:
            continue

        print()
        print(group_name)
        print("-" * SECTION_WIDTH)
        for column, label, unit in available_fields:
            displayed_columns.add(column)
            print(f"{label + ':':<{LABEL_WIDTH}}{format_value(row[column], unit)}")

    # This makes the display future-proof: newly added model features remain visible
    # even before they have been assigned to a presentation group.
    remaining_columns = [
        column for column in row.index if column not in displayed_columns
    ]
    if remaining_columns:
        print()
        print("OTHER MODEL FEATURES")
        print("-" * SECTION_WIDTH)
        for column in remaining_columns:
            print(f"{column + ':':<{LABEL_WIDTH}}{format_value(row[column])}")

    print()
    print("Patient Data  ->  AI Model  ->  Prediction")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH.resolve()}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test dataset: {TEST_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Test dataset is missing '{TARGET_COLUMN}'.")

    example = test_df.iloc[[0]].copy()
    actual_label = int(example[TARGET_COLUMN].iloc[0])
    features = example.drop(columns=[TARGET_COLUMN])

    print_model_input(features)

    predicted_label = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    print()
    print_heading("MODEL OUTPUT")
    print()
    print(f"Predicted Label: {CLASS_NAMES[predicted_label]}")
    print()
    print("Confidence Scores")
    print("-" * SECTION_WIDTH)
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name:<12}{probability:.4f}")
    print()
    print(f"Actual Test Label: {CLASS_NAMES[actual_label]}")


if __name__ == "__main__":
    main()
