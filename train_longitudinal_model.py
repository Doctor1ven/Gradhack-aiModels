"""Train longitudinal next-session Recovery Readiness models.

The models are trained on synthetic longitudinal data. Metrics describe
performance on this synthetic task only and are not clinically validated.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DATA_DIR = Path("longitudinal_data")
OUTPUT_DIR = Path("longitudinal_model_outputs")
TARGET_COLUMN = "future_readiness_label"
RANDOM_STATE = 42
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["REDUCE", "MAINTAIN", "PROGRESS"]

LEAKAGE_COLUMNS = {
    "Entity Number",
    "First Name",
    "Surname",
    "Exercise Record ID",
    "Health Record ID",
    "Record Date",
    "Event Date",
    "positive_score",
    "negative_score",
    "future_readiness_text",
    "Recovery Profile",
    "Is Synthetic",
}


def load_split(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing longitudinal split: {path.resolve()}")
    return pd.read_csv(path)


def validate_split(name: str, data: pd.DataFrame) -> None:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"{name} is missing target column '{TARGET_COLUMN}'.")
    leakage = sorted(
        column
        for column in data.columns
        if column in LEAKAGE_COLUMNS
        or column.startswith("next_")
        or column.endswith("_change")
    )
    if leakage:
        raise ValueError(f"{name} contains leakage columns: {leakage}")
    unexpected_labels = sorted(set(data[TARGET_COLUMN].dropna()) - set(CLASS_LABELS))
    if unexpected_labels:
        raise ValueError(f"{name} contains unexpected labels: {unexpected_labels}")


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
    return data[feature_columns].copy(), data[TARGET_COLUMN].astype(int).copy()


def detect_column_types(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = list(data.select_dtypes(include=["object", "string", "category"]).columns)
    numeric = [column for column in data.columns if column not in categorical]
    return categorical, numeric


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
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
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_models(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical, numeric)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical, numeric)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=14,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(categorical, numeric)),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,
                        eval_metric="mlogloss",
                        n_estimators=300,
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
        ),
    }


def evaluate_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=CLASS_LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
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


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metrics["macro_f1"],
        metrics["reduce_recall"],
        metrics["balanced_accuracy"],
        metrics["progress_recall"],
    )


def fit_model(name: str, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    if name in {"LogisticRegression", "XGBoost"}:
        model.fit(x_train, y_train, model__sample_weight=sample_weights)
    else:
        model.fit(x_train, y_train)
    return model


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Longitudinal Confusion Matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def transformed_feature_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(estimator.coef_), axis=0)
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
    print("\nValidation comparison:")
    print(
        comparison[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
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
    print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
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

    print("\nConfusion matrix:")
    print(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_df = load_split("test.csv")

    for name, data in {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }.items():
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
    validation_metrics: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        fitted_models[name] = fit_model(name, model, x_train, y_train)
        validation_metrics[name] = evaluate_model(fitted_models[name], x_validation, y_validation)

    comparison = pd.DataFrame(
        [{"model": name, **metrics} for name, metrics in validation_metrics.items()]
    ).drop(columns=["per_class"])
    print_validation_comparison(comparison)

    selected_name = max(validation_metrics, key=lambda name: selection_key(validation_metrics[name]))
    selected_model = fitted_models[selected_name]
    print(f"\nSelected model: {selected_name}")

    test_predictions = selected_model.predict(x_test)
    test_probabilities = selected_model.predict_proba(x_test)
    test_metrics = evaluate_model(selected_model, x_test, y_test)
    matrix = confusion_matrix(y_test, test_predictions, labels=CLASS_LABELS)
    print_test_metrics(test_metrics, matrix)

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            test_predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    predictions_df = pd.DataFrame(
        {
            "actual_label": y_test.to_numpy(),
            "predicted_label": test_predictions,
            "actual_text": [CLASS_NAMES[value] for value in y_test.to_numpy()],
            "predicted_text": [CLASS_NAMES[value] for value in test_predictions],
        }
    )
    predictions_df = pd.concat([predictions_df, probability_frame(test_probabilities)], axis=1)

    importance = transformed_feature_importance(selected_model)
    print("\nTop 20 transformed features:")
    print(importance.to_string(index=False))

    metrics = {
        "synthetic_data_notice": "Trained from synthetic longitudinal data; not clinically validated.",
        "selected_model": selected_name,
        "selection_criteria": ["macro_f1", "reduce_recall", "balanced_accuracy", "progress_recall"],
        "validation": validation_metrics,
        "test": test_metrics,
        "confusion_matrix": matrix.tolist(),
    }

    joblib.dump(selected_model, OUTPUT_DIR / "recovery_readiness_longitudinal_pipeline.joblib")
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    save_confusion_matrix(matrix, OUTPUT_DIR / "confusion_matrix.png")
    save_feature_importance(
        importance,
        OUTPUT_DIR / "feature_importance.csv",
        OUTPUT_DIR / "feature_importance.png",
    )

    print("\nGenerated model outputs:")
    for filename in [
        "recovery_readiness_longitudinal_pipeline.joblib",
        "metrics.json",
        "model_comparison.csv",
        "classification_report.csv",
        "test_predictions.csv",
        "confusion_matrix.png",
        "feature_importance.csv",
        "feature_importance.png",
    ]:
        print((OUTPUT_DIR / filename).resolve())


if __name__ == "__main__":
    main()
