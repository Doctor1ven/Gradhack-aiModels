"""Run one next-session readiness prediction from the longitudinal pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("longitudinal_model_outputs/recovery_readiness_longitudinal_pipeline.joblib")
TEST_PATH = Path("longitudinal_data/test.csv")
TARGET_COLUMN = "future_readiness_label"
CLASS_NAMES = ["REDUCE", "MAINTAIN", "PROGRESS"]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH.resolve()}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {TEST_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET_COLUMN not in test.columns:
        raise ValueError(f"Test data is missing target column '{TARGET_COLUMN}'.")

    example = test.iloc[[0]].copy()
    actual_label = int(example[TARGET_COLUMN].iloc[0])
    features = example.drop(columns=[TARGET_COLUMN])

    predicted_label = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    print("Longitudinal example prediction:")
    print(f"Predicted next-session label: {predicted_label} ({CLASS_NAMES[predicted_label]})")
    print(f"Actual next-session label: {actual_label} ({CLASS_NAMES[actual_label]})")
    print("Probabilities:")
    print(f"REDUCE: {probabilities[0]:.4f}")
    print(f"MAINTAIN: {probabilities[1]:.4f}")
    print(f"PROGRESS: {probabilities[2]:.4f}")


if __name__ == "__main__":
    main()
