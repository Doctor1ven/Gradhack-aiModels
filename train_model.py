"""Train and evaluate Recovery Readiness classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DATA_DIR = Path("prepared_data")
OUTPUT_DIR = Path("model_outputs")
RANDOM_STATE = 42
TARGET_COLUMN = "readiness_label"
CLASS_NAMES = ["REDUCE", "MAINTAIN", "PROGRESS"]
CLASS_LABELS = [0, 1, 2]

LEAKAGE_COLUMNS = {
    "Entity Number",
    "First Name",
    "Surname",
    "Exercise Record ID",
    "Health Record ID",
    "Record Date",
    "Event Date",
    "readiness_text",
    "progress_score",
}


def load_split(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path.resolve()}")
    return pd.read_csv(path)


def validate_split(name: str, df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{name} is missing target column '{TARGET_COLUMN}'.")

    present_leakage = sorted(LEAKAGE_COLUMNS.intersection(df.columns))
    if present_leakage:
        raise ValueError(
            f"{name} contains leakage/identifier columns that must not be model "
            f"inputs: {present_leakage}"
        )

    unexpected_labels = sorted(set(df[TARGET_COLUMN].dropna()) - set(CLASS_LABELS))
    if unexpected_labels:
        raise ValueError(f"{name} contains unexpected readiness labels: {unexpected_labels}")


def validate_matching_features(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> list[str]:
    feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    for name, df in {"validation": validation, "test": test}.items():
        columns = [column for column in df.columns if column != TARGET_COLUMN]
        if columns != feature_columns:
            missing = sorted(set(feature_columns) - set(columns))
            extra = sorted(set(columns) - set(feature_columns))
            raise ValueError(
                f"{name} feature columns do not match train columns. "
                f"Missing: {missing}; extra: {extra}"
            )
    return feature_columns


def split_xy(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_columns].copy(), df[TARGET_COLUMN].astype(int).copy()


def detect_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns
    numeric_columns = [column for column in df.columns if column not in categorical_columns]
    return list(categorical_columns), numeric_columns


def make_preprocessor(
    categorical_columns: list[str], numeric_columns: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_baseline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=10000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_xgboost(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x)
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predictions,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "reduce_recall": float(recall[0]),
        "maintain_recall": float(recall[1]),
        "progress_recall": float(recall[2]),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(CLASS_NAMES))
        },
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        metrics["macro_f1"],
        metrics["reduce_recall"],
        metrics["progress_recall"],
    )


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Recovery Readiness Confusion Matrix")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_importance(model: Pipeline, path_csv: Path, path_png: Path) -> None:
    xgb_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    importance.to_csv(path_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_data = importance.sort_values("importance")
    ax.barh(plot_data["feature"], plot_data["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 XGBoost Transformed Features")
    fig.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)


def probability_frame(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        probabilities,
        columns=[
            "probability_reduce",
            "probability_maintain",
            "probability_progress",
        ],
    )


def print_validation_comparison(comparison: pd.DataFrame) -> None:
    print("\nValidation metrics:")
    print(
        comparison[
            [
                "model",
                "accuracy",
                "macro_f1",
                "reduce_recall",
                "maintain_recall",
                "progress_recall",
            ]
        ].to_string(index=False)
    )


def print_test_metrics(metrics: dict[str, Any], matrix: np.ndarray) -> None:
    print("\nSelected model test metrics:")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"REDUCE recall: {metrics['reduce_recall']:.4f}")
    print(f"MAINTAIN recall: {metrics['maintain_recall']:.4f}")
    print(f"PROGRESS recall: {metrics['progress_recall']:.4f}")

    print("\nPer-class precision, recall, and F1:")
    for label, values in metrics["per_class"].items():
        print(
            f"{label}: precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, f1={values['f1']:.4f}, "
            f"support={values['support']}"
        )

    print("\nConfusion matrix values:")
    print(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_df = load_split("test.csv")

    for name, df in {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }.items():
        validate_split(name, df)

    feature_columns = validate_matching_features(train_df, validation_df, test_df)
    x_train, y_train = split_xy(train_df, feature_columns)
    x_validation, y_validation = split_xy(validation_df, feature_columns)
    x_test, y_test = split_xy(test_df, feature_columns)

    categorical_columns, numeric_columns = detect_column_types(x_train)
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    baseline = make_baseline(make_preprocessor(categorical_columns, numeric_columns))
    xgboost = make_xgboost(make_preprocessor(categorical_columns, numeric_columns))

    baseline.fit(x_train, y_train)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    xgboost.fit(x_train, y_train, model__sample_weight=sample_weights)

    baseline_validation = evaluate_model(baseline, x_validation, y_validation)
    xgboost_validation = evaluate_model(xgboost, x_validation, y_validation)

    comparison = pd.DataFrame(
        [
            {"model": "LogisticRegression", **baseline_validation},
            {"model": "XGBoost", **xgboost_validation},
        ]
    ).drop(columns=["per_class"])
    print_validation_comparison(comparison)

    selected_name, selected_model, selected_validation = max(
        [
            ("LogisticRegression", baseline, baseline_validation),
            ("XGBoost", xgboost, xgboost_validation),
        ],
        key=lambda item: selection_key(item[2]),
    )
    print(f"\nSelected model: {selected_name}")

    test_predictions = selected_model.predict(x_test)
    test_probabilities = selected_model.predict_proba(x_test)
    test_metrics = evaluate_model(selected_model, x_test, y_test)
    matrix = confusion_matrix(y_test, test_predictions, labels=CLASS_LABELS)
    print_test_metrics(test_metrics, matrix)

    metrics = {
        "selected_model": selected_name,
        "selection_criteria": ["macro_f1", "reduce_recall", "progress_recall"],
        "validation": {
            "LogisticRegression": baseline_validation,
            "XGBoost": xgboost_validation,
        },
        "test": test_metrics,
        "confusion_matrix": matrix.tolist(),
    }

    report = classification_report(
        y_test,
        test_predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    predictions_df = pd.DataFrame(
        {
            "actual_label": y_test.to_numpy(),
            "predicted_label": test_predictions,
            "actual_text": [CLASS_NAMES[value] for value in y_test.to_numpy()],
            "predicted_text": [CLASS_NAMES[value] for value in test_predictions],
        }
    )
    predictions_df = pd.concat(
        [predictions_df, probability_frame(test_probabilities)],
        axis=1,
    )

    joblib.dump(selected_model, OUTPUT_DIR / "recovery_readiness_pipeline.joblib")
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    save_confusion_matrix(matrix, OUTPUT_DIR / "confusion_matrix.png")

    if selected_name == "XGBoost":
        save_feature_importance(
            selected_model,
            OUTPUT_DIR / "feature_importance.csv",
            OUTPUT_DIR / "feature_importance.png",
        )
    else:
        save_feature_importance(
            xgboost,
            OUTPUT_DIR / "feature_importance.csv",
            OUTPUT_DIR / "feature_importance.png",
        )

    print("\nGenerated model outputs:")
    for filename in [
        "recovery_readiness_pipeline.joblib",
        "metrics.json",
        "classification_report.csv",
        "test_predictions.csv",
        "confusion_matrix.png",
        "model_comparison.csv",
        "feature_importance.csv",
        "feature_importance.png",
    ]:
        print((OUTPUT_DIR / filename).resolve())


if __name__ == "__main__":
    main()
