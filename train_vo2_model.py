"""Train four-week VO2 Max forecast regression models.

Metrics describe synthetic-data performance only. This model is not clinically
validated and should not be used for medical decision-making.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


DATA_DIR = Path("vo2_data")
OUTPUT_DIR = Path("vo2_model_outputs")
TARGET_COLUMN = "future_vo2_4_weeks"
CURRENT_VO2_COLUMN = "VO2 Max Estimate"
RANDOM_STATE = 42

LEAKAGE_COLUMNS = {
    "Entity Number",
    "First Name",
    "Surname",
    "Health Record ID",
    "Exercise Record ID",
    "Record Date",
    "Event Date",
    "future_record_date",
    "future_session_number",
    "vo2_change_4_weeks",
}


def load_split(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing VO2 split: {path.resolve()}")
    return pd.read_csv(path)


def validate_split(name: str, data: pd.DataFrame) -> None:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"{name} is missing target column '{TARGET_COLUMN}'.")
    leakage = sorted(
        column
        for column in data.columns
        if column != TARGET_COLUMN
        if column in LEAKAGE_COLUMNS
        or column.casefold().startswith(("future_", "next_"))
        or "record id" in column.casefold()
        or column.casefold() in {"record date", "event date"}
    )
    if leakage:
        raise ValueError(f"{name} contains leakage columns: {leakage}")
    if CURRENT_VO2_COLUMN not in data.columns:
        raise ValueError(f"{name} is missing current VO2 input feature '{CURRENT_VO2_COLUMN}'.")


def validate_matching_features(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> list[str]:
    feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    for name, data in {"validation": validation, "test": test}.items():
        columns = [column for column in data.columns if column != TARGET_COLUMN]
        if columns != feature_columns:
            missing = sorted(set(feature_columns) - set(columns))
            extra = sorted(set(columns) - set(feature_columns))
            raise ValueError(
                f"{name} features do not match train. Missing: {missing}; extra: {extra}"
            )
    return feature_columns


def split_xy(data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return data[feature_columns].copy(), data[TARGET_COLUMN].astype(float).copy()


def detect_column_types(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = list(data.select_dtypes(include=["object", "string", "category", "bool"]).columns)
    numeric = [column for column in data.columns if column not in categorical]
    return categorical, numeric


def make_preprocessor(categorical: list[str], numeric: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_models(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    return {
        "DummyMean": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical, numeric, scale_numeric=False)),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical, numeric, scale_numeric=True)),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical, numeric, scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical, numeric, scale_numeric=False)),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=450,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=4,
                        reg_alpha=0.05,
                        reg_lambda=1.5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, predictions)
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, predictions)),
        "MedianAbsoluteError": float(median_absolute_error(y_true, predictions)),
    }


def transformed_feature_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        values = np.zeros(len(feature_names))
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(20)
    )


def save_feature_importance(importance: pd.DataFrame, csv_path: Path, png_path: Path) -> None:
    importance.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_data = importance.sort_values("importance")
    ax.barh(plot_data["feature"], plot_data["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Transformed Features")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def save_actual_vs_predicted(y_true: pd.Series, predictions: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, predictions, alpha=0.45, s=18)
    min_value = min(float(y_true.min()), float(np.min(predictions)))
    max_value = max(float(y_true.max()), float(np.max(predictions)))
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
    ax.set_xlabel("Actual future VO2 Max")
    ax.set_ylabel("Predicted future VO2 Max")
    ax.set_title("Actual vs Predicted VO2 Max")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_residual_distribution(residuals: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=40, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_comparison(comparison: pd.DataFrame) -> None:
    print("\nValidation model comparison:")
    print(
        comparison[["model", "MAE", "RMSE", "R2", "MedianAbsoluteError"]]
        .sort_values(["MAE", "RMSE"], ascending=[True, True])
        .to_string(index=False)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_df = load_split("test.csv")
    for name, data in {"train.csv": train_df, "validation.csv": validation_df, "test.csv": test_df}.items():
        validate_split(name, data)

    feature_columns = validate_matching_features(train_df, validation_df, test_df)
    x_train, y_train = split_xy(train_df, feature_columns)
    x_validation, y_validation = split_xy(validation_df, feature_columns)
    x_test, y_test = split_xy(test_df, feature_columns)
    categorical, numeric = detect_column_types(x_train)

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Numeric features: {len(numeric)}")
    print(f"Categorical features: {len(categorical)}")

    models = make_models(categorical, numeric)
    fitted_models: dict[str, Pipeline] = {}
    validation_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        validation_metrics[name] = evaluate_predictions(y_validation, model.predict(x_validation))

    comparison = pd.DataFrame([{"model": name, **metrics} for name, metrics in validation_metrics.items()])
    print_comparison(comparison)
    selected_name = comparison.sort_values(["MAE", "RMSE", "R2"], ascending=[True, True, False]).iloc[0]["model"]
    selected_model = fitted_models[str(selected_name)]
    print(f"\nSelected model: {selected_name}")

    test_predictions = selected_model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_predictions)
    persistence_predictions = x_test[CURRENT_VO2_COLUMN].astype(float).to_numpy()
    persistence_mae = float(mean_absolute_error(y_test, persistence_predictions))
    improvement = (
        (persistence_mae - test_metrics["MAE"]) / persistence_mae * 100
        if persistence_mae != 0
        else 0.0
    )

    print("\nSelected model test metrics:")
    print(f"Test MAE: {test_metrics['MAE']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test R2: {test_metrics['R2']:.4f}")
    print(f"Test Median Absolute Error: {test_metrics['MedianAbsoluteError']:.4f}")
    print(f"Persistence-baseline MAE: {persistence_mae:.4f}")
    print(f"MAE improvement over persistence baseline: {improvement:.2f}%")

    predictions_df = pd.DataFrame(
        {
            "actual_future_vo2": y_test.to_numpy(),
            "predicted_future_vo2": test_predictions,
            "current_vo2": x_test[CURRENT_VO2_COLUMN].to_numpy(),
        }
    )
    predictions_df["actual_vo2_change"] = predictions_df["actual_future_vo2"] - predictions_df["current_vo2"]
    predictions_df["predicted_vo2_change"] = predictions_df["predicted_future_vo2"] - predictions_df["current_vo2"]
    predictions_df["absolute_error"] = (
        predictions_df["actual_future_vo2"] - predictions_df["predicted_future_vo2"]
    ).abs()
    predictions_df["residual"] = predictions_df["actual_future_vo2"] - predictions_df["predicted_future_vo2"]

    importance = transformed_feature_importance(selected_model)
    print("\nTop 20 transformed features:")
    print(importance.to_string(index=False))

    metrics = {
        "synthetic_data_notice": "Trained from synthetic longitudinal data; not clinically validated.",
        "selected_model": str(selected_name),
        "selection_criteria": ["lowest validation MAE", "validation RMSE", "validation R2"],
        "validation": validation_metrics,
        "test": test_metrics,
        "persistence_baseline": {
            "prediction": "current VO2 Max Estimate",
            "MAE": persistence_mae,
            "percentage_improvement_over_persistence_mae": float(improvement),
        },
    }

    joblib.dump(selected_model, OUTPUT_DIR / "vo2_forecast_pipeline.joblib")
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    save_feature_importance(
        importance,
        OUTPUT_DIR / "feature_importance.csv",
        OUTPUT_DIR / "feature_importance.png",
    )
    save_actual_vs_predicted(y_test, test_predictions, OUTPUT_DIR / "actual_vs_predicted.png")
    save_residual_distribution(predictions_df["residual"].to_numpy(), OUTPUT_DIR / "residual_distribution.png")

    print("\nGenerated model outputs:")
    for filename in [
        "vo2_forecast_pipeline.joblib",
        "metrics.json",
        "model_comparison.csv",
        "test_predictions.csv",
        "feature_importance.csv",
        "feature_importance.png",
        "actual_vs_predicted.png",
        "residual_distribution.png",
    ]:
        print((OUTPUT_DIR / filename).resolve())


if __name__ == "__main__":
    main()
