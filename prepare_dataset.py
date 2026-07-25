"""Prepare Recovery Readiness datasets from the source Excel workbook.

The readiness target produced here is generated from transparent MVP rules.
It is not clinician-validated ground truth and should be reviewed before any
clinical or operational use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EXCEL_FILE = Path("Fully_sorted2.xlsx")
OUTPUT_DIR = Path("prepared_data")
RANDOM_STATE = 42
PROGRESS_SCORE_THRESHOLD = 8

REQUIRED_SHEETS = ["Personal Information", "Exercise Data", "Health Data"]

PERSONAL_COLUMNS = [
    "Entity Number",
    "First Name",
    "Surname",
    "Gender",
    "Age",
    "Activity Baseline",
    "Recovery Goal",
]

DATE_COLUMNS = ["Record Date", "Event Date"]

KNOWN_NUMERIC_COLUMNS = [
    "Age",
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
]

NON_FEATURE_COLUMNS = [
    "Entity Number",
    "First Name",
    "Surname",
    "Exercise Record ID",
    "Health Record ID",
    "Record Date",
    "Event Date",
    "readiness_text",
    "progress_score",
]


def normalize_name(name: object) -> str:
    """Normalize workbook names so capitalization and whitespace do not matter."""
    return " ".join(str(name).strip().casefold().split())


def require_workbook(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path.resolve()}. Place Fully_sorted2.xlsx in this folder."
        )


def resolve_sheet_name(sheet_names: Iterable[str], expected_name: str) -> str:
    lookup = {normalize_name(name): name for name in sheet_names}
    normalized = normalize_name(expected_name)
    if normalized not in lookup:
        available = ", ".join(sheet_names)
        raise ValueError(
            f"Missing required sheet '{expected_name}'. Available sheets: {available}"
        )
    return lookup[normalized]


def strip_column_whitespace(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    renamed = {column: str(column).strip() for column in df.columns}
    stripped = df.rename(columns=renamed)
    normalized_columns = [normalize_name(column) for column in stripped.columns]
    duplicates = pd.Series(normalized_columns)[
        pd.Series(normalized_columns).duplicated()
    ].tolist()
    if duplicates:
        raise ValueError(
            f"Sheet '{sheet_name}' has duplicate column names after trimming/case "
            f"normalization: {sorted(set(duplicates))}"
        )
    return stripped


def resolve_column(df: pd.DataFrame, expected_name: str, sheet_name: str) -> str:
    lookup = {normalize_name(column): column for column in df.columns}
    normalized = normalize_name(expected_name)
    if normalized not in lookup:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required column '{expected_name}'. "
            f"Available columns: {available}"
        )
    return lookup[normalized]


def canonicalize_columns(
    df: pd.DataFrame, expected_columns: Iterable[str], sheet_name: str
) -> pd.DataFrame:
    rename_map = {
        resolve_column(df, expected, sheet_name): expected for expected in expected_columns
    }
    return df.rename(columns=rename_map)


def load_required_sheets(path: Path) -> dict[str, pd.DataFrame]:
    require_workbook(path)
    workbook = pd.ExcelFile(path)
    sheets: dict[str, pd.DataFrame] = {}

    for expected_sheet in REQUIRED_SHEETS:
        actual_sheet = resolve_sheet_name(workbook.sheet_names, expected_sheet)
        df = pd.read_excel(path, sheet_name=actual_sheet)
        df = strip_column_whitespace(df, actual_sheet)
        df = canonicalize_columns(df, ["Entity Number"], actual_sheet)
        sheets[expected_sheet] = df
        print(f"{expected_sheet} shape: {df.shape}")

    return sheets


def validate_entity_key(df: pd.DataFrame, sheet_name: str) -> None:
    if "Entity Number" not in df.columns:
        raise ValueError(f"Sheet '{sheet_name}' is missing required column 'Entity Number'.")
    if df["Entity Number"].isna().any():
        count = int(df["Entity Number"].isna().sum())
        raise ValueError(f"Sheet '{sheet_name}' has {count} blank Entity Number values.")


def validate_many_to_one(df: pd.DataFrame, sheet_name: str) -> None:
    duplicated = df["Entity Number"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "Entity Number"].head(10).tolist()
        raise ValueError(
            f"Sheet '{sheet_name}' has duplicate Entity Number values, which would "
            f"create an unexpected many-to-many join. Example values: {examples}"
        )


def select_personal_columns(personal: pd.DataFrame) -> pd.DataFrame:
    personal = canonicalize_columns(personal, PERSONAL_COLUMNS, "Personal Information")
    return personal[PERSONAL_COLUMNS].copy()


def merge_dataset(
    personal: pd.DataFrame, exercise: pd.DataFrame, health: pd.DataFrame
) -> pd.DataFrame:
    for sheet_name, df in {
        "Personal Information": personal,
        "Exercise Data": exercise,
        "Health Data": health,
    }.items():
        validate_entity_key(df, sheet_name)

    validate_many_to_one(personal, "Personal Information")
    validate_many_to_one(health, "Health Data")

    master = exercise.merge(
        personal,
        on="Entity Number",
        how="left",
        validate="many_to_one",
    )
    master = master.merge(
        health,
        on="Entity Number",
        how="left",
        validate="many_to_one",
    )

    unmatched_personal = int(master["First Name"].isna().sum()) if "First Name" in master else 0
    health_id_column = "Health Record ID" if "Health Record ID" in master.columns else None
    unmatched_health = int(master[health_id_column].isna().sum()) if health_id_column else 0
    if unmatched_personal or unmatched_health:
        raise ValueError(
            "Exercise Data contains Entity Number values without matching member or "
            f"health rows. Missing personal matches: {unmatched_personal}; "
            f"missing health matches: {unmatched_health}."
        )

    print(f"Merged dataset shape: {master.shape}")
    return master


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def convert_numeric_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_columns = [column for column in KNOWN_NUMERIC_COLUMNS if column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df, numeric_columns


def fill_missing_values(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    for column in numeric_columns:
        if df[column].isna().any():
            median = df[column].median()
            fill_value = 0 if pd.isna(median) else median
            df[column] = df[column].fillna(fill_value)

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()

    return df


def print_missing_summary(df: pd.DataFrame, title: str) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\n{title}")
    if missing.empty:
        print("No missing values.")
    else:
        print(missing.to_string())


def truthy_flag(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    return normalized.isin({"1", "true", "yes", "y", "imputed"})


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"Record Date", "Event Date"}.issubset(df.columns):
        df["Days Since Event"] = (df["Record Date"] - df["Event Date"]).dt.days

    if {"Duration Min", "RPE 1-10"}.issubset(df.columns):
        df["Workout Load"] = df["Duration Min"] * df["RPE 1-10"]

    if {"Max Heart Rate", "Resting Heart Rate"}.issubset(df.columns):
        df["Heart Rate Reserve Used"] = df["Max Heart Rate"] - df["Resting Heart Rate"]

    if {"Distance Km", "Duration Min"}.issubset(df.columns):
        duration_hours = df["Duration Min"].replace(0, np.nan) / 60
        df["Average Speed KmH"] = (
            df["Distance Km"] / duration_hours
        ).replace([np.inf, -np.inf], np.nan)

    imputation_flag_columns = [
        column for column in df.columns if "imputed" in normalize_name(column)
    ]
    if imputation_flag_columns:
        df["Any Physiological Value Imputed"] = (
            pd.concat([truthy_flag(df[column]) for column in imputation_flag_columns], axis=1)
            .any(axis=1)
            .astype(int)
        )

    engineered_numeric = [
        "Days Since Event",
        "Workout Load",
        "Heart Rate Reserve Used",
        "Average Speed KmH",
        "Any Physiological Value Imputed",
    ]
    existing_engineered = [column for column in engineered_numeric if column in df.columns]
    return fill_missing_values(df, existing_engineered)


def text_value(row: pd.Series, column: str) -> str | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    value = str(row[column]).strip().casefold()
    return value if value != "unknown" else None


def numeric_value(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return None


def has_critical_reduce_signal(row: pd.Series) -> bool:
    """Return True when a safety-first REDUCE condition applies."""
    clinician_cleared = text_value(row, "Clinician Cleared")
    contraindication = text_value(row, "Contraindication Flag")
    recovery_stage = text_value(row, "Recovery Stage")

    pain = numeric_value(row, "Pain Score")
    sleep = numeric_value(row, "Sleep Hours")
    rpe = numeric_value(row, "RPE 1-10")

    reduce_conditions = [
        clinician_cleared in {"no", "false", "0"},
        contraindication in {"yes", "true", "1"},
        pain is not None and pain >= 7,
        recovery_stage is not None and "acute" in recovery_stage,
        sleep is not None and sleep < 5,
        rpe is not None and rpe >= 9,
    ]
    return any(reduce_conditions)


def calculate_progress_score(row: pd.Series) -> int:
    """Count applicable positive recovery signals for auditable MVP labels."""
    clinician_cleared = text_value(row, "Clinician Cleared")
    contraindication = text_value(row, "Contraindication Flag")
    recovery_stage = text_value(row, "Recovery Stage")
    workout_status = text_value(row, "Workout Status")
    vo2_risk = text_value(row, "VO2 Risk Band")
    mobility_limitation = text_value(row, "Mobility Limitation")

    pain = numeric_value(row, "Pain Score")
    sleep = numeric_value(row, "Sleep Hours")
    rpe = numeric_value(row, "RPE 1-10")
    hrv = numeric_value(row, "HRV ms")

    positive_conditions = [
        clinician_cleared in {"yes", "true", "1"},
        contraindication in {"no", "false", "0"},
        pain is not None and pain <= 3,
        recovery_stage in {"building consistency", "maintenance"},
        sleep is not None and sleep >= 6.5,
        rpe is not None and rpe <= 7,
        workout_status == "completed",
        vo2_risk in {"low", "moderate"},
        hrv is not None and hrv >= 35,
    ]

    if mobility_limitation is not None:
        positive_conditions.append(
            mobility_limitation in {"none", "no", "mild", "minor", "low"}
        )

    return int(sum(positive_conditions))


def create_readiness_label(row: pd.Series) -> int:
    """Return 0 REDUCE, 1 MAINTAIN, or 2 PROGRESS using MVP safety rules."""
    if has_critical_reduce_signal(row):
        return 0

    if calculate_progress_score(row) >= PROGRESS_SCORE_THRESHOLD:
        return 2

    return 1


def add_readiness_target(df: pd.DataFrame) -> pd.DataFrame:
    df["progress_score"] = df.apply(calculate_progress_score, axis=1)
    df["readiness_label"] = df.apply(create_readiness_label, axis=1)
    df["readiness_text"] = df["readiness_label"].map(
        {0: "REDUCE", 1: "MAINTAIN", 2: "PROGRESS"}
    )
    return df


def print_label_distribution(df: pd.DataFrame) -> None:
    counts = df["readiness_text"].value_counts().reindex(
        ["REDUCE", "MAINTAIN", "PROGRESS"], fill_value=0
    )
    percentages = (counts / len(df) * 100).round(2)
    print("\nReadiness label distribution:")
    for label in counts.index:
        print(f"{label}: {counts[label]} ({percentages[label]}%)")


def build_model_dataset(master: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [column for column in NON_FEATURE_COLUMNS if column in master.columns]
    return master.drop(columns=drop_columns).copy()


def split_by_entity(
    master: pd.DataFrame, model_dataset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_frame = model_dataset.copy()
    split_frame["Entity Number"] = master["Entity Number"].values

    members = pd.Series(master["Entity Number"].dropna().unique())
    if len(members) < 3:
        raise ValueError("At least three unique Entity Number values are required to split.")

    train_members, temp_members = train_test_split(
        members,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    validation_members, test_members = train_test_split(
        temp_members,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    train = split_frame[split_frame["Entity Number"].isin(train_members)].copy()
    validation = split_frame[split_frame["Entity Number"].isin(validation_members)].copy()
    test = split_frame[split_frame["Entity Number"].isin(test_members)].copy()

    for split in (train, validation, test):
        split.drop(columns=["Entity Number"], inplace=True)

    print("\nFinal split sizes:")
    print(f"Train: {len(train)} rows, {len(train_members)} members")
    print(f"Validation: {len(validation)} rows, {len(validation_members)} members")
    print(f"Test: {len(test)} rows, {len(test_members)} members")

    return train, validation, test


def save_outputs(
    master: pd.DataFrame,
    model_dataset: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    outputs = {
        "master_dataset_with_identifiers.csv": master,
        "model_dataset.csv": model_dataset,
        "train.csv": train,
        "validation.csv": validation,
        "test.csv": test,
    }

    paths: list[Path] = []
    for filename, dataframe in outputs.items():
        path = OUTPUT_DIR / filename
        dataframe.to_csv(path, index=False)
        paths.append(path)

    print("\nGenerated files:")
    for path in paths:
        print(path.resolve())

    return paths


def main() -> None:
    try:
        sheets = load_required_sheets(EXCEL_FILE)
        personal = select_personal_columns(sheets["Personal Information"])
        exercise = sheets["Exercise Data"].copy()
        health = sheets["Health Data"].copy()

        master = merge_dataset(personal, exercise, health)
        print_missing_summary(master, "Missing-value summary before cleaning:")

        master = convert_dates(master)
        master, numeric_columns = convert_numeric_fields(master)
        master = fill_missing_values(master, numeric_columns)
        master = engineer_features(master)
        master = add_readiness_target(master)

        print_missing_summary(master, "Missing-value summary after cleaning:")
        print_label_distribution(master)

        model_dataset = build_model_dataset(master)
        train, validation, test = split_by_entity(master, model_dataset)
        save_outputs(master, model_dataset, train, validation, test)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
