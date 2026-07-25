"""Prepare longitudinal next-session Recovery Readiness datasets.

This stage uses synthetic weekly sessions to build member timelines. The target
is future_readiness_label for the next weekly session, not same-row readiness.
The data is synthetic MVP data and is not clinically validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_FILE = Path("synthetic_data/synthetic_weekly_sessions.csv")
OUTPUT_DIR = Path("longitudinal_data")
RANDOM_STATE = 42
TARGET_COLUMN = "future_readiness_label"
TARGET_TEXT_COLUMN = "future_readiness_text"
PROGRESS_SCORE_THRESHOLD = 10
CLASS_TEXT = {0: "REDUCE", 1: "MAINTAIN", 2: "PROGRESS"}

REQUIRED_COLUMNS = [
    "Entity Number",
    "Record Date",
    "Synthetic Session Number",
    "Exercise Record ID",
    "Workout Status",
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
    "Pain Score",
    "Setback Flag",
]

NEXT_COLUMN_MAP = {
    "Workout Status": "next_workout_status",
    "Duration Min": "next_duration",
    "Distance Km": "next_distance",
    "Avg Heart Rate": "next_avg_heart_rate",
    "Max Heart Rate": "next_max_heart_rate",
    "Resting Heart Rate": "next_resting_heart_rate",
    "HRV ms": "next_hrv",
    "Sleep Hours": "next_sleep_hours",
    "Sleep Quality Score": "next_sleep_quality",
    "Calories Burned": "next_calories",
    "VO2 Max Estimate": "next_vo2",
    "RPE 1-10": "next_rpe",
    "Pain Score": "next_pain",
    "Setback Flag": "next_setback_flag",
}

CHANGE_COLUMNS = [
    "duration_change",
    "distance_change",
    "resting_hr_change",
    "hrv_change",
    "sleep_change",
    "vo2_change",
    "rpe_change",
    "pain_change",
]

IDENTIFIER_AND_AUDIT_ONLY_COLUMNS = {
    "Entity Number",
    "First Name",
    "Surname",
    "Exercise Record ID",
    "Health Record ID",
    "Record Date",
    "Event Date",
    "Recovery Profile",
    "Is Synthetic",
    "Synthetic Session Number",
    "positive_score",
    "negative_score",
    TARGET_TEXT_COLUMN,
}


def load_source_data() -> pd.DataFrame:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing synthetic sessions file: {SOURCE_FILE.resolve()}")
    data = pd.read_csv(SOURCE_FILE)
    data.columns = [str(column).strip() for column in data.columns]
    return data


def validate_source_data(data: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        failures.append(f"Missing required columns: {missing}")
        return failures

    if data["Exercise Record ID"].duplicated().any():
        failures.append("Duplicate Exercise Record ID values found.")
    if data["Entity Number"].isna().any():
        failures.append("Entity Number contains missing values.")
    if data["Record Date"].isna().any():
        failures.append("Record Date contains missing values.")

    ordered = data.copy()
    ordered["Record Date"] = pd.to_datetime(ordered["Record Date"], errors="coerce")
    if ordered["Record Date"].isna().any():
        failures.append("Record Date contains unparsable values.")
    ordered = ordered.sort_values(["Entity Number", "Record Date", "Synthetic Session Number"])
    bad_members = []
    for entity_number, group in ordered.groupby("Entity Number"):
        if (group["Record Date"].diff().dropna() < pd.Timedelta(0)).any():
            bad_members.append(entity_number)
    if bad_members:
        failures.append(f"Sessions are not chronological for members: {bad_members[:10]}")

    return failures


def to_numeric(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def add_next_session_columns(data: pd.DataFrame) -> pd.DataFrame:
    for source_column, next_column in NEXT_COLUMN_MAP.items():
        data[next_column] = data.groupby("Entity Number")[source_column].shift(-1)
    return data


def add_change_columns(data: pd.DataFrame) -> pd.DataFrame:
    data["duration_change"] = data["next_duration"] - data["Duration Min"]
    data["distance_change"] = data["next_distance"] - data["Distance Km"]
    data["resting_hr_change"] = data["next_resting_heart_rate"] - data["Resting Heart Rate"]
    data["hrv_change"] = data["next_hrv"] - data["HRV ms"]
    data["sleep_change"] = data["next_sleep_hours"] - data["Sleep Hours"]
    data["vo2_change"] = data["next_vo2"] - data["VO2 Max Estimate"]
    data["rpe_change"] = data["next_rpe"] - data["RPE 1-10"]
    data["pain_change"] = data["next_pain"] - data["Pain Score"]
    return data


def trend(values: pd.Series) -> float:
    valid = values.dropna()
    if len(valid) < 2:
        return 0.0
    return float(valid.iloc[-1] - valid.iloc[0])


def add_historical_features(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("Entity Number", group_keys=False)
    data["session_number"] = data["Synthetic Session Number"]
    data["days_since_previous_session"] = grouped["Record Date"].diff().dt.days.fillna(0)
    data["previous_duration"] = grouped["Duration Min"].shift(1)
    data["previous_rpe"] = grouped["RPE 1-10"].shift(1)
    data["previous_vo2"] = grouped["VO2 Max Estimate"].shift(1)
    data["previous_hrv"] = grouped["HRV ms"].shift(1)
    data["previous_pain"] = grouped["Pain Score"].shift(1)

    rolling_sources = {
        "duration": "Duration Min",
        "rpe": "RPE 1-10",
        "vo2": "VO2 Max Estimate",
        "hrv": "HRV ms",
        "sleep": "Sleep Hours",
        "resting_hr": "Resting Heart Rate",
        "pain": "Pain Score",
    }
    for prefix, source_column in rolling_sources.items():
        prior = grouped[source_column].shift(1)
        data[f"average_{prefix}_last_3"] = (
            prior.groupby(data["Entity Number"])
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        data[f"{prefix}_trend_last_3"] = (
            prior.groupby(data["Entity Number"])
            .rolling(window=3, min_periods=2)
            .apply(trend, raw=False)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    completed = data["Workout Status"].astype(str).str.casefold().eq("completed").astype(int)
    setback = data["Setback Flag"].map(as_bool).astype(int)
    training_load = data["Duration Min"] * data["RPE 1-10"]

    data["completion_rate_before_current"] = (
        grouped.apply(lambda g: completed.loc[g.index].shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    data["setback_count_before_current"] = (
        grouped.apply(lambda g: setback.loc[g.index].shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    data["cumulative_training_load_before_current"] = (
        grouped.apply(lambda g: training_load.loc[g.index].shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    data["average_training_load_last_3"] = (
        training_load.groupby(data["Entity Number"])
        .shift(1)
        .groupby(data["Entity Number"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return data


def score_future_row(row: pd.Series) -> tuple[int, int, int]:
    next_status = str(row["next_workout_status"]).strip().casefold()
    next_setback = as_bool(row["next_setback_flag"])
    next_pain = float(row["next_pain"])
    next_rpe = float(row["next_rpe"])
    next_sleep = float(row["next_sleep_hours"])
    duration_change = float(row["duration_change"])
    distance_change = float(row["distance_change"])
    resting_hr_change = float(row["resting_hr_change"])
    hrv_change = float(row["hrv_change"])
    vo2_change = float(row["vo2_change"])
    rpe_change = float(row["rpe_change"])
    pain_change = float(row["pain_change"])
    next_duration = float(row["next_duration"])
    current_duration = float(row["Duration Min"])

    sharp_unplanned_duration_drop = (
        current_duration >= 15
        and duration_change <= -0.35 * max(current_duration, 1)
        and next_status == "completed"
        and next_rpe >= 7
    )

    critical_conditions = [
        next_setback,
        next_status != "completed",
        next_pain >= 7,
        pain_change >= 2,
        next_rpe >= 9,
        hrv_change <= -8,
        resting_hr_change >= 8,
        next_sleep < 5,
        vo2_change <= -1.0,
        sharp_unplanned_duration_drop,
    ]

    negative_score = int(sum(critical_conditions))
    negative_score += int(rpe_change >= 2)
    negative_score += int(distance_change <= -1.0 and next_duration > 0)

    positive_conditions = [
        next_status == "completed",
        next_pain <= 3 or pain_change < 0,
        next_rpe <= 7,
        duration_change >= -3 and duration_change <= 12,
        distance_change >= -0.3,
        hrv_change >= -2,
        resting_hr_change <= 2,
        next_sleep >= 6.5,
        vo2_change >= -0.2,
        not next_setback,
    ]
    positive_score = int(sum(positive_conditions))

    if any(critical_conditions) or negative_score >= 3:
        label = 0
    elif positive_score >= PROGRESS_SCORE_THRESHOLD:
        label = 2
    else:
        label = 1
    return positive_score, negative_score, label


def add_future_target(data: pd.DataFrame) -> pd.DataFrame:
    scores = data.apply(score_future_row, axis=1, result_type="expand")
    data["positive_score"] = scores[0].astype(int)
    data["negative_score"] = scores[1].astype(int)
    data[TARGET_COLUMN] = scores[2].astype(int)
    data[TARGET_TEXT_COLUMN] = data[TARGET_COLUMN].map(CLASS_TEXT)
    return data


def build_model_dataset(audit: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    leakage_columns = set(IDENTIFIER_AND_AUDIT_ONLY_COLUMNS)
    leakage_columns.update(column for column in audit.columns if column.startswith("next_"))
    leakage_columns.update(CHANGE_COLUMNS)
    leakage_columns.update({"Previous Session Duration", "Previous Session RPE", "Previous Session VO2", "Previous Session HRV"})

    model = audit.drop(columns=[column for column in leakage_columns if column in audit.columns])
    feature_columns = [column for column in model.columns if column != TARGET_COLUMN]
    leakage_remaining = [
        column
        for column in feature_columns
        if column.startswith("next_")
        or column in CHANGE_COLUMNS
        or column in {"positive_score", "negative_score", TARGET_TEXT_COLUMN}
        or column in IDENTIFIER_AND_AUDIT_ONLY_COLUMNS
    ]
    return model, feature_columns, leakage_remaining


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


def label_distribution(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    counts = data[TARGET_TEXT_COLUMN].value_counts().reindex(CLASS_TEXT.values(), fill_value=0)
    total = len(data)
    return {
        label: {"count": int(count), "percentage": round(float(count / total * 100), 2)}
        for label, count in counts.items()
    }


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
    audit.to_csv(OUTPUT_DIR / "longitudinal_audit_dataset.csv", index=False)
    model.to_csv(OUTPUT_DIR / "model_dataset.csv", index=False)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    with (OUTPUT_DIR / "feature_list.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)
    with (OUTPUT_DIR / "preparation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Unique members: {summary['unique_member_count']}")
    print(f"Usable session pairs: {summary['usable_session_pair_count']}")
    print(f"Rows removed because no next session exists: {summary['rows_removed_no_next_session']}")
    print("\nTarget distribution:")
    for label, values in summary["target_distribution"].items():
        print(f"{label}: {values['count']} ({values['percentage']}%)")
    print("\nSplit sizes:")
    print(f"Train: {summary['train_rows']}")
    print(f"Validation: {summary['validation_rows']}")
    print(f"Test: {summary['test_rows']}")
    print("\nLeakage-check results:")
    if summary["leakage_columns_remaining"]:
        print(f"FAILED: {summary['leakage_columns_remaining']}")
    else:
        print("Passed: no next-session or target-helper columns remain in model features.")
    print("\nFeature count:", summary["feature_count"])
    print("\nGenerated files:")
    for path in summary["generated_files"]:
        print(path)


def main() -> None:
    data = load_source_data()
    validation_failures = validate_source_data(data)
    if validation_failures:
        raise SystemExit("Source validation failed: " + "; ".join(validation_failures))

    numeric_columns = [
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
        "Pain Score",
        "Synthetic Session Number",
    ]
    data = to_numeric(data, numeric_columns)
    data["Record Date"] = pd.to_datetime(data["Record Date"], errors="coerce")
    data = data.sort_values(["Entity Number", "Record Date", "Synthetic Session Number"]).reset_index(drop=True)

    original_rows = len(data)
    data = add_next_session_columns(data)
    data = add_change_columns(data)
    data = add_historical_features(data)
    audit = data.dropna(subset=["next_duration"]).copy()
    removed_no_next = original_rows - len(audit)
    audit = add_future_target(audit)

    model, feature_columns, leakage_remaining = build_model_dataset(audit)
    train, validation, test = split_by_member(audit, model)

    summary = {
        "synthetic_data_notice": "Trained from synthetic longitudinal data; not clinically validated.",
        "unique_member_count": int(audit["Entity Number"].nunique()),
        "usable_session_pair_count": int(len(audit)),
        "rows_removed_no_next_session": int(removed_no_next),
        "target_distribution": label_distribution(audit),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_columns)),
        "leakage_columns_remaining": leakage_remaining,
        "generated_files": [
            str((OUTPUT_DIR / "longitudinal_audit_dataset.csv").resolve()),
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
