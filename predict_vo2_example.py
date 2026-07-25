"""Run one four-week VO2 Max forecast example."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("vo2_model_outputs/vo2_forecast_pipeline.joblib")
TEST_PATH = Path("vo2_data/test.csv")
TARGET_COLUMN = "future_vo2_4_weeks"
CURRENT_VO2_COLUMN = "VO2 Max Estimate"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH.resolve()}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {TEST_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET_COLUMN not in test.columns:
        raise ValueError(f"Test data is missing target column '{TARGET_COLUMN}'.")
    if CURRENT_VO2_COLUMN not in test.columns:
        raise ValueError(f"Test data is missing current VO2 column '{CURRENT_VO2_COLUMN}'.")

    example = test.iloc[[0]].copy()
    actual_future_vo2 = float(example[TARGET_COLUMN].iloc[0])
    current_vo2 = float(example[CURRENT_VO2_COLUMN].iloc[0])
    features = example.drop(columns=[TARGET_COLUMN])
    predicted_future_vo2 = float(model.predict(features)[0])
    predicted_change = predicted_future_vo2 - current_vo2
    actual_change = actual_future_vo2 - current_vo2
    absolute_error = abs(actual_future_vo2 - predicted_future_vo2)

    print("VO2 Forecast Example")
    print()
    print(f"Current VO2 Max: {current_vo2:.1f}")
    print(f"Predicted VO2 Max in 4 weeks: {predicted_future_vo2:.1f}")
    print(f"Actual VO2 Max in 4 weeks: {actual_future_vo2:.1f}")
    print(f"Predicted change: {predicted_change:+.1f}")
    print(f"Actual change: {actual_change:+.1f}")
    print(f"Absolute error: {absolute_error:.1f} ml/kg/min")


if __name__ == "__main__":
    main()
