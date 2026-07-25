"""Prepare four-week VO2 Max forecast regression datasets.

This workflow uses synthetic weekly sessions to predict a member's VO2 Max
Estimate four sessions into the future. The source data is synthetic and the
resulting model is not clinically validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_FILE = Path("synthetic_data/synthetic_weekly_sessions.csv")
OUTPUT_DIR = Path("vo2_data")
TARGET_COLUMN = "future_vo2_4_weeks"
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "Entity Number",
    "Record Date",
    "Synthetic Session Number",
    "VO2 Max Estimate",
    "Duration Min",
    "Distance Km",
    "Resting Heart Rate",
    "HRV ms",
    "Sleep Hours",
    "RPE 1-10",
]

NUMERIC_COLUMNS = [
    "Age",
    "Mobility Limitation",
    "Pain Score",
    "Duration Min",
    "Distance Km",
    "Avg Heart Rate",
    "Max Heart Rate",
    "Resting Heart Rate",
    "HRV ms",
    "Sleep Hours",
    "Sleep Quality Score",
    "Calories Burned",
    "VO2 Max Estimate",
    "RPE 1-10",
    "Previous Session Duration",
    "Previous Session RPE",
    "Previous Session VO2",
    "Previous Session HRV",
    "Synthetic Session Number",
]

IDENTIFIER_AND_AUDIT_ONLY_COLUMNS = {
    "Entity Number",
    "First Name",
    "Surname",
    "Health Record ID",
    "Exercise Record ID",
    "Event Date",
    "Record Date",
    "future_record_date",
    "future_session_number",
    TARGET_COLUMN,
    "vo2_change_4_weeks",
    "Recovery Profile",
    "Is Synthetic",
}

LEAKAGE_PATTERNS = [
    "future_",
    "next_",
    "record date",
    "event date",
]


def load_source_data() -> pd.DataFrame:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing synthetic sessions file: {SOURCE_FILE.resolve()}")
    data = pd.read_csv(SOURCE_FILE)
    data.columns = [str(column).strip() for column in data.columns]
    return data


def validate_required_columns(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def convert_types(data: pd.DataFrame) -> pd.DataFrame:
    data["Record Date"] = pd.to_datetime(data["Record Date"], errors="coerce")
    if data["Record Date"].isna().any():
        raise ValueError("Record Date contains unparsable values.")
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def validate_chronology(data: pd.DataFrame) -> None:
    bad_members: list[Any] = []
    for entity_number, group in data.groupby("Entity Number", sort=False):
        sorted_group = group.sort_values(["Record Date", "Synthetic Session Number"])
        if not group.index.equals(sorted_group.index):
            bad_members.append(entity_number)
        if (group["Record Date"].diff().dropna() < pd.Timedelta(0)).any():
            bad_members.append(entity_number)
    if bad_members:
        raise ValueError(f"Sessions are not chronological for members: {bad_members[:10]}")


def trend(values: pd.Series) -> float:
    valid = values.dropna()
    if len(valid) < 2:
        return 0.0
    return float(valid.iloc[-1] - valid.iloc[0])


def as_bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.casefold().isin({"true", "1", "yes", "y"})


def rolling_prior_mean(data: pd.DataFrame, source: pd.Series, window: int = 3) -> pd.Series:
    return (
        source.groupby(data["Entity Number"])
        .shift(1)
        .groupby(data["Entity Number"])
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_prior_trend(data: pd.DataFrame, source: pd.Series, window: int = 3) -> pd.Series:
    return (
        source.groupby(data["Entity Number"])
        .shift(1)
        .groupby(data["Entity Number"])
        .rolling(window=window, min_periods=2)
        .apply(trend, raw=False)
        .reset_index(level=0, drop=True)
        .fillna(0)
    )


def add_future_target(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("Entity Number", group_keys=False)
    data[TARGET_COLUMN] = grouped["VO2 Max Estimate"].shift(-4)
    data["future_record_date"] = grouped["Record Date"].shift(-4)
    data["future_session_number"] = grouped["Synthetic Session Number"].shift(-4)
    data["vo2_change_4_weeks"] = data[TARGET_COLUMN] - data["VO2 Max Estimate"]
    return data


def add_historical_features(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("Entity Number", group_keys=False)
    data["session_number"] = data["Synthetic Session Number"]
    data["days_since_previous_session"] = grouped["Record Date"].diff().dt.days
    data["current_training_load"] = data["Duration Min"] * data["RPE 1-10"]

    previous_map = {
        "vo2_change_from_previous": "VO2 Max Estimate",
        "duration_change_from_previous": "Duration Min",
        "rpe_change_from_previous": "RPE 1-10",
        "hrv_change_from_previous": "HRV ms",
    }
    for output_column, source_column in previous_map.items():
        data[output_column] = data[source_column] - grouped[source_column].shift(1)

    rolling_sources = {
        "vo2": data["VO2 Max Estimate"],
        "duration": data["Duration Min"],
        "rpe": data["RPE 1-10"],
        "hrv": data["HRV ms"],
        "sleep": data["Sleep Hours"],
        "resting_hr": data["Resting Heart Rate"],
        "pain": data["Pain Score"] if "Pain Score" in data.columns else pd.Series(np.nan, index=data.index),
        "training_load": data["current_training_load"],
    }
    for prefix, source in rolling_sources.items():
        data[f"average_{prefix}_last_3"] = rolling_prior_mean(data, source)

    trend_sources = {
        key: value
        for key, value in rolling_sources.items()
        if key != "training_load"
    }
    for prefix, source in trend_sources.items():
        data[f"{prefix}_trend_last_3"] = rolling_prior_trend(data, source)

    completed = (
        data["Workout Status"].astype(str).str.strip().str.casefold().eq("completed").astype(int)
        if "Workout Status" in data.columns
        else pd.Series(0, index=data.index)
    )
    setback = (
        as_bool_series(data["Setback Flag"]).astype(int)
        if "Setback Flag" in data.columns
        else pd.Series(0, index=data.index)
    )

    data["completion_rate_before_current"] = (
        grouped.apply(lambda g: completed.loc[g.index].shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    data["setback_rate_before_current"] = (
        grouped.apply(lambda g: setback.loc[g.index].shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    data["cumulative_training_minutes"] = (
        grouped["Duration Min"].apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    data["cumulative_training_load"] = (
        data["current_training_load"]
        .groupby(data["Entity Number"])
        .shift(1)
        .groupby(data["Entity Number"])
        .expanding()
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    return data


def leakage_columns(feature_columns: list[str]) -> list[str]:
    failures: list[str] = []
    for column in feature_columns:
        lowered = column.casefold()
        if any(pattern in lowered for pattern in LEAKAGE_PATTERNS):
            failures.append(column)
        if column in IDENTIFIER_AND_AUDIT_ONLY_COLUMNS:
            failures.append(column)
        if column == "vo2_change_4_weeks":
            failures.append(column)
        if "record id" in lowered or lowered.endswith(" id") or " record id" in lowered:
            failures.append(column)
    return sorted(set(failures))


def build_model_dataset(audit: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    drop_columns = [column for column in IDENTIFIER_AND_AUDIT_ONLY_COLUMNS if column in audit.columns]
    model = audit.drop(columns=drop_columns)
    empty_feature_columns = [
        column
        for column in model.columns
        if column != TARGET_COLUMN and model[column].isna().all()
    ]
    model = model.drop(columns=empty_feature_columns)
    feature_columns = [column for column in model.columns if column != TARGET_COLUMN]
    failures = leakage_columns(feature_columns)
    if "VO2 Max Estimate" not in feature_columns:
        failures.append("VO2 Max Estimate was accidentally removed from model features.")
    model[TARGET_COLUMN] = audit[TARGET_COLUMN].values
    feature_columns = [column for column in model.columns if column != TARGET_COLUMN]
    return model, feature_columns, sorted(set(failures))


def split_by_member(audit: pd.DataFrame, model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_frame = model.copy()
    split_frame["Entity Number"] = audit["Entity Number"].values
    members = pd.Series(audit["Entity Number"].dropna().unique())
    train_members, temp_members = train_test_split(
        members, test_size=0.30, random_state=RANDOM_STATE
    )
    validation_members, test_members = train_test_split(
        temp_members, test_size=0.50, random_state=RANDOM_STATE
    )
    train = split_frame[split_frame["Entity Number"].isin(train_members)].copy()
    validation = split_frame[split_frame["Entity Number"].isin(validation_members)].copy()
    test = split_frame[split_frame["Entity Number"].isin(test_members)].copy()
    for frame in (train, validation, test):
        frame.drop(columns=["Entity Number"], inplace=True)
    return train, validation, test


def stats(series: pd.Series, include_std: bool = True) -> dict[str, float]:
    output = {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
    }
    if include_std:
        output["std"] = float(series.std())
    return output


def save_outputs(
    audit: pd.DataFrame,
    model: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    summary: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    audit.to_csv(OUTPUT_DIR / "vo2_audit_dataset.csv", index=False)
    model.to_csv(OUTPUT_DIR / "model_dataset.csv", index=False)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    with (OUTPUT_DIR / "feature_list.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)
    with (OUTPUT_DIR / "preparation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Original row count: {summary['original_row_count']}")
    print(f"Unique member count: {summary['unique_member_count']}")
    print(f"Usable four-week prediction rows: {summary['usable_four_week_prediction_rows']}")
    print(f"Rows removed due to missing future VO2: {summary['rows_removed_missing_future_vo2']}")
    print("\nTarget statistics:")
    for key, value in summary["target_statistics"].items():
        print(f"{key}: {value:.4f}")
    print("\nVO2 change over four weeks:")
    for key, value in summary["vo2_change_4_weeks_statistics"].items():
        print(f"{key}: {value:.4f}")
    print("\nSplit sizes:")
    print(f"Train: {summary['train_rows']}")
    print(f"Validation: {summary['validation_rows']}")
    print(f"Test: {summary['test_rows']}")
    print(f"\nNumeric features: {summary['numeric_feature_count']}")
    print(f"Categorical features: {summary['categorical_feature_count']}")
    print("\nLeakage-check results:")
    if summary["leakage_columns_remaining"]:
        print(f"FAILED: {summary['leakage_columns_remaining']}")
    else:
        print("Passed: no future, identifier, name, raw date, or target-helper columns remain.")
    print("\nGenerated files:")
    for path in summary["generated_files"]:
        print(path)


def main() -> None:
    data = load_source_data()
    validate_required_columns(data)
    original_rows = len(data)
    original_members = int(data["Entity Number"].nunique())
    data = convert_types(data)
    data = data.sort_values(["Entity Number", "Record Date", "Synthetic Session Number"]).reset_index(drop=True)
    validate_chronology(data)
    data = add_future_target(data)
    data = add_historical_features(data)

    audit = data.dropna(subset=[TARGET_COLUMN]).copy()
    removed_missing_future = original_rows - len(audit)
    model, feature_columns, leakage_remaining = build_model_dataset(audit)
    train, validation, test = split_by_member(audit, model)

    categorical = list(model[feature_columns].select_dtypes(include=["object", "string", "category", "bool"]).columns)
    numeric = [column for column in feature_columns if column not in categorical]

    summary = {
        "synthetic_data_notice": "Trained from synthetic longitudinal data; not clinically validated.",
        "original_row_count": int(original_rows),
        "unique_member_count": original_members,
        "usable_four_week_prediction_rows": int(len(audit)),
        "rows_removed_missing_future_vo2": int(removed_missing_future),
        "target_statistics": stats(audit[TARGET_COLUMN]),
        "vo2_change_4_weeks_statistics": stats(audit["vo2_change_4_weeks"], include_std=False),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "feature_count": int(len(feature_columns)),
        "leakage_columns_remaining": leakage_remaining,
        "generated_files": [
            str((OUTPUT_DIR / "vo2_audit_dataset.csv").resolve()),
            str((OUTPUT_DIR / "model_dataset.csv").resolve()),
            str((OUTPUT_DIR / "train.csv").resolve()),
            str((OUTPUT_DIR / "validation.csv").resolve()),
            str((OUTPUT_DIR / "test.csv").resolve()),
            str((OUTPUT_DIR / "feature_list.json").resolve()),
            str((OUTPUT_DIR / "preparation_summary.json").resolve()),
        ],
    }

    save_outputs(audit, model, train, validation, test, feature_columns, summary)
    print_summary(summary)
    if leakage_remaining:
        raise SystemExit("Leakage check failed.")


if __name__ == "__main__":
    main()
