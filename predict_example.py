"""Run one example prediction with the saved Recovery Readiness pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("model_outputs/recovery_readiness_pipeline.joblib")
TEST_PATH = Path("prepared_data/test.csv")
TARGET_COLUMN = "readiness_label"
CLASS_NAMES = ["REDUCE", "MAINTAIN", "PROGRESS"]


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

    predicted_label = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    print("Example prediction:")
    print(f"Predicted label: {predicted_label} ({CLASS_NAMES[predicted_label]})")
    print(f"Actual test label: {actual_label} ({CLASS_NAMES[actual_label]})")
    print("Probabilities:")
    print(f"REDUCE: {probabilities[0]:.4f}")
    print(f"MAINTAIN: {probabilities[1]:.4f}")
    print(f"PROGRESS: {probabilities[2]:.4f}")


if __name__ == "__main__":
    main()
