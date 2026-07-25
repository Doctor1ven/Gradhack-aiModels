"""Generate synthetic weekly Recovery Readiness exercise sessions.

This script creates MVP synthetic data from the existing workbook. The output is
not clinically validated and must be treated as synthetic simulation data only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXCEL_FILE = Path("Fully_sorted2.xlsx")
OUTPUT_DIR = Path("synthetic_data")
RANDOM_STATE = 42

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
EXERCISE_COLUMNS = [
    "Exercise Record ID",
    "Entity Number",
    "Record Date",
    "Workout Type",
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
    "RHR Imputed",
    "HRV Imputed",
    "Sleep Hours Imputed",
    "Sleep Quality Imputed",
]
SIMULATED_NUMERIC_COLUMNS = [
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
SYNTHETIC_COLUMNS = [
    "Synthetic Session Number",
    "Recovery Profile",
    "Is Synthetic",
    "Setback Flag",
    "Previous Session Duration",
    "Previous Session RPE",
    "Previous Session VO2",
    "Previous Session HRV",
]
PROFILE_NAMES = [
    "fast recovery",
    "normal recovery",
    "slow recovery",
    "recovery with setbacks",
    "poor adherence",
    "strong adherence",
]


def normalize_name(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def resolve_column(df: pd.DataFrame, expected: str, sheet_name: str) -> str:
    lookup = {normalize_name(column): column for column in df.columns}
    normalized = normalize_name(expected)
    if normalized not in lookup:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required column '{expected}'. "
            f"Available columns: {available}"
        )
    return lookup[normalized]


def canonicalize_columns(
    df: pd.DataFrame, expected_columns: list[str], sheet_name: str
) -> pd.DataFrame:
    rename_map = {
        resolve_column(df, expected, sheet_name): expected for expected in expected_columns
    }
    return df.rename(columns=rename_map)


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and lightly validate the workbook source sheets."""
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Workbook not found: {EXCEL_FILE.resolve()}")

    workbook = pd.ExcelFile(EXCEL_FILE)
    sheet_lookup = {normalize_name(name): name for name in workbook.sheet_names}
    missing_sheets = [
        sheet for sheet in REQUIRED_SHEETS if normalize_name(sheet) not in sheet_lookup
    ]
    if missing_sheets:
        raise ValueError(f"Missing required workbook sheets: {missing_sheets}")

    personal = pd.read_excel(
        EXCEL_FILE, sheet_name=sheet_lookup[normalize_name("Personal Information")]
    )
    exercise = pd.read_excel(
        EXCEL_FILE, sheet_name=sheet_lookup[normalize_name("Exercise Data")]
    )
    health = pd.read_excel(
        EXCEL_FILE, sheet_name=sheet_lookup[normalize_name("Health Data")]
    )

    personal.columns = [str(column).strip() for column in personal.columns]
    exercise.columns = [str(column).strip() for column in exercise.columns]
    health.columns = [str(column).strip() for column in health.columns]

    personal = canonicalize_columns(personal, PERSONAL_COLUMNS, "Personal Information")
    exercise = canonicalize_columns(exercise, ["Entity Number"], "Exercise Data")
    health = canonicalize_columns(health, ["Entity Number"], "Health Data")

    missing_exercise = [column for column in EXERCISE_COLUMNS if column not in exercise]
    if missing_exercise:
        raise ValueError(f"Exercise Data is missing required columns: {missing_exercise}")

    for name, df in {
        "Personal Information": personal,
        "Exercise Data": exercise,
        "Health Data": health,
    }.items():
        if df["Entity Number"].isna().any():
            raise ValueError(f"{name} contains missing Entity Number values.")

    return personal[PERSONAL_COLUMNS].copy(), exercise.copy(), health.copy()


def source_ranges(exercise: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for column in [
        "HRV ms",
        "VO2 Max Estimate",
        "Sleep Quality Score",
        "Resting Heart Rate",
        "Avg Heart Rate",
        "Max Heart Rate",
    ]:
        values = pd.to_numeric(exercise[column], errors="coerce")
        ranges[column] = (float(values.min()), float(values.max()))
    return ranges


def assign_recovery_profile(rng: np.random.Generator) -> str:
    """Assign one reproducible member recovery profile."""
    probabilities = [0.15, 0.30, 0.20, 0.15, 0.10, 0.10]
    return str(rng.choice(PROFILE_NAMES, p=probabilities))


def latest_exercise_by_member(exercise: pd.DataFrame) -> pd.DataFrame:
    data = exercise.copy()
    data["Record Date"] = pd.to_datetime(data["Record Date"], errors="coerce")
    return data.sort_values(["Entity Number", "Record Date"]).drop_duplicates(
        "Entity Number", keep="last"
    )


def numeric_or_default(value: Any, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(numeric) else float(numeric)


def text_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row.index or pd.isna(row[column]):
        return default
    return str(row[column]).strip()


def sensible_workout_type(personal_row: pd.Series, health_row: pd.Series) -> str:
    goal = normalize_name(text_value(personal_row, "Recovery Goal"))
    stage = normalize_name(text_value(health_row, "Recovery Stage"))
    context = normalize_name(text_value(health_row, "Recovery Context"))
    mobility = normalize_name(text_value(health_row, "Mobility Limitation"))

    unsafe_for_running = any(
        phrase in " ".join([stage, context, mobility])
        for phrase in ["acute", "surgery", "injury", "avoid impact", "short bouts"]
    )
    if "swim" in goal or "pool" in mobility:
        return "Swimming"
    if "running" in goal and not unsafe_for_running:
        return "Running"
    return "Walking"


def create_starting_state(
    personal_row: pd.Series,
    health_row: pd.Series,
    latest_exercise: pd.Series | None,
    ranges: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Create a realistic previous-session state for simulation week one."""
    if latest_exercise is not None:
        state = {column: latest_exercise.get(column) for column in EXERCISE_COLUMNS}
        state["Record Date"] = pd.to_datetime(state["Record Date"], errors="coerce")
    else:
        baseline = normalize_name(text_value(personal_row, "Activity Baseline"))
        pain = numeric_or_default(health_row.get("Pain Score"), 4)
        cleared = normalize_name(text_value(health_row, "Clinician Cleared"))
        contraindication = normalize_name(text_value(health_row, "Contraindication Flag"))
        acute = "acute" in normalize_name(text_value(health_row, "Recovery Stage"))

        if cleared == "no" or contraindication == "yes" or acute:
            duration = 10
            rpe = 5
        elif baseline in {"active", "moderate activity"}:
            duration = 28
            rpe = 6
        elif baseline == "inactive":
            duration = 14
            rpe = 6
        else:
            duration = 18
            rpe = 6

        hrv_min, hrv_max = ranges["HRV ms"]
        vo2_min, vo2_max = ranges["VO2 Max Estimate"]
        state = {
            "Exercise Record ID": None,
            "Entity Number": personal_row["Entity Number"],
            "Record Date": pd.Timestamp("2026-07-24"),
            "Workout Type": sensible_workout_type(personal_row, health_row),
            "Workout Status": "Completed",
            "Duration Min": duration,
            "Distance Km": max(0.2, duration / 15),
            "Avg Heart Rate": 118 + pain,
            "Max Heart Rate": 145 + pain * 2,
            "Resting Heart Rate": 75 + pain / 2,
            "HRV ms": np.clip(50 - pain * 1.5, hrv_min, hrv_max),
            "Sleep Hours": 6.5,
            "Sleep Quality Score": 88,
            "Calories Burned": duration * 6,
            "VO2 Max Estimate": np.clip(34 - pain * 0.4, vo2_min, vo2_max),
            "RPE 1-10": rpe,
            "RHR Imputed": False,
            "HRV Imputed": False,
            "Sleep Hours Imputed": False,
            "Sleep Quality Imputed": False,
        }

    state["Pain Score"] = numeric_or_default(health_row.get("Pain Score"), 4)
    return state


def profile_parameters(profile: str) -> dict[str, float]:
    return {
        "fast recovery": {"progress": 1.25, "noise": 0.9, "adherence": 0.94},
        "normal recovery": {"progress": 1.0, "noise": 1.0, "adherence": 0.90},
        "slow recovery": {"progress": 0.65, "noise": 1.1, "adherence": 0.88},
        "recovery with setbacks": {"progress": 0.80, "noise": 1.25, "adherence": 0.86},
        "poor adherence": {"progress": 0.45, "noise": 1.35, "adherence": 0.68},
        "strong adherence": {"progress": 1.15, "noise": 0.65, "adherence": 0.98},
    }[profile]


def choose_workout_status(profile: str, safety_limited: bool, rng: np.random.Generator) -> str:
    if safety_limited:
        return str(rng.choice(["Completed", "Skipped"], p=[0.55, 0.45]))
    adherence = profile_parameters(profile)["adherence"]
    if rng.random() <= adherence:
        return "Completed"
    return str(rng.choice(["Skipped", "Modified"], p=[0.65, 0.35]))


def choose_workout_type(
    personal_row: pd.Series,
    health_row: pd.Series,
    previous_type: str,
    session_number: int,
    safety_limited: bool,
) -> str:
    preferred = sensible_workout_type(personal_row, health_row)
    if safety_limited:
        return "Walking" if preferred != "Swimming" else "Swimming"
    if preferred == "Running" and session_number < 4:
        return "Walking"
    return preferred if preferred else previous_type


def apply_setback(
    next_state: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Apply a temporary setback to the current week."""
    next_state["Pain Score"] = min(10, next_state["Pain Score"] + rng.uniform(1.5, 3.0))
    next_state["RPE 1-10"] = min(10, next_state["RPE 1-10"] + rng.uniform(1.0, 2.0))
    next_state["Duration Min"] = max(0, next_state["Duration Min"] * rng.uniform(0.55, 0.8))
    next_state["Distance Km"] = max(0, next_state["Distance Km"] * rng.uniform(0.45, 0.75))
    next_state["HRV ms"] = max(ranges["HRV ms"][0], next_state["HRV ms"] - rng.uniform(3, 8))
    next_state["VO2 Max Estimate"] = max(
        ranges["VO2 Max Estimate"][0], next_state["VO2 Max Estimate"] - rng.uniform(0.2, 0.8)
    )
    return next_state


def simulate_next_week(
    previous: dict[str, Any],
    personal_row: pd.Series,
    health_row: pd.Series,
    profile: str,
    session_number: int,
    setback_weeks: set[int],
    ranges: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Simulate one weekly session from the prior session state."""
    params = profile_parameters(profile)
    stage = normalize_name(text_value(health_row, "Recovery Stage"))
    cleared = normalize_name(text_value(health_row, "Clinician Cleared"))
    contraindication = normalize_name(text_value(health_row, "Contraindication Flag"))
    acute = "acute" in stage
    safety_limited = cleared == "no" or contraindication == "yes" or acute

    previous_duration = numeric_or_default(previous.get("Duration Min"), 15)
    previous_rpe = numeric_or_default(previous.get("RPE 1-10"), 6)
    previous_vo2 = numeric_or_default(previous.get("VO2 Max Estimate"), 35)
    previous_hrv = numeric_or_default(previous.get("HRV ms"), 53)
    previous_pain = numeric_or_default(previous.get("Pain Score"), 4)
    previous_sleep = numeric_or_default(previous.get("Sleep Hours"), 6.8)

    high_rpe_penalty = 0.25 if previous_rpe >= 8 else 1.0
    no_progress = cleared == "no" or contraindication == "yes"
    progress = 0 if no_progress else params["progress"] * high_rpe_penalty
    noise = params["noise"]

    status = choose_workout_status(profile, safety_limited, rng)
    skipped = status == "Skipped"
    setback = session_number in setback_weeks

    duration_change = rng.normal(2.0 * progress, 2.0 * noise)
    if skipped:
        duration = 0
    elif safety_limited:
        duration = min(25, max(5, previous_duration + rng.normal(0, 2)))
    else:
        duration = previous_duration + duration_change

    pain = previous_pain - rng.uniform(0.05, 0.35) * progress + rng.normal(0, 0.35 * noise)
    sleep = previous_sleep + rng.normal(0.05 * progress, 0.25 * noise)
    if profile == "poor adherence":
        sleep -= rng.uniform(0, 0.25)

    hrv = previous_hrv + rng.normal(0.6 * progress, 1.4 * noise)
    vo2 = previous_vo2 + rng.normal(0.25 * progress, 0.35 * noise)
    resting_hr = numeric_or_default(previous.get("Resting Heart Rate"), 73)
    resting_hr += rng.normal(-0.25 * progress, 1.0 * noise)

    if sleep < 6:
        hrv -= rng.uniform(1.5, 4.0)
        resting_hr += rng.uniform(1.0, 3.0)
    if pain >= 6:
        duration *= rng.uniform(0.75, 0.95)

    rpe = previous_rpe + rng.normal(-0.15 * progress, 0.65 * noise)
    rpe += max(0, pain - 4) * 0.35
    if sleep < 6:
        rpe += 0.4

    workout_type = choose_workout_type(
        personal_row, health_row, str(previous.get("Workout Type", "Walking")), session_number, safety_limited
    )
    distance_factor = {"Walking": 0.075, "Swimming": 0.045, "Running": 0.12}[workout_type]
    distance = 0 if skipped else duration * distance_factor * rng.uniform(0.85, 1.15)

    avg_hr = resting_hr + 38 + rpe * 2.2 + rng.normal(0, 4)
    max_hr = avg_hr + 22 + rpe * 2.5 + rng.normal(0, 5)
    calories = 0 if skipped else duration * (4.0 + rpe * 0.45) * rng.uniform(0.85, 1.15)

    next_state = {
        **previous,
        "Record Date": pd.to_datetime(previous["Record Date"]) + pd.Timedelta(days=7),
        "Workout Type": workout_type,
        "Workout Status": status,
        "Duration Min": duration,
        "Distance Km": distance,
        "Avg Heart Rate": avg_hr,
        "Max Heart Rate": max_hr,
        "Resting Heart Rate": resting_hr,
        "HRV ms": hrv,
        "Sleep Hours": sleep,
        "Sleep Quality Score": numeric_or_default(previous.get("Sleep Quality Score"), 90)
        + rng.normal(0.5 * progress, 1.5 * noise),
        "Calories Burned": calories,
        "VO2 Max Estimate": vo2,
        "RPE 1-10": rpe,
        "Pain Score": pain,
        "Setback Flag": setback,
        "Previous Session Duration": previous_duration,
        "Previous Session RPE": previous_rpe,
        "Previous Session VO2": previous_vo2,
        "Previous Session HRV": previous_hrv,
    }

    if setback:
        next_state = apply_setback(next_state, ranges, rng)

    next_state["Pain Score"] = round(float(np.clip(next_state["Pain Score"], 0, 10)), 1)
    next_state["RPE 1-10"] = int(round(float(np.clip(next_state["RPE 1-10"], 1, 10))))
    next_state["Sleep Hours"] = round(float(np.clip(next_state["Sleep Hours"], 3, 10)), 1)
    next_state["Sleep Quality Score"] = int(
        round(float(np.clip(next_state["Sleep Quality Score"], *ranges["Sleep Quality Score"])))
    )
    next_state["HRV ms"] = int(round(float(np.clip(next_state["HRV ms"], *ranges["HRV ms"]))))
    next_state["VO2 Max Estimate"] = round(
        float(np.clip(next_state["VO2 Max Estimate"], *ranges["VO2 Max Estimate"])), 1
    )
    next_state["Resting Heart Rate"] = int(
        round(float(np.clip(next_state["Resting Heart Rate"], 45, 110)))
    )
    next_state["Duration Min"] = int(round(float(np.clip(next_state["Duration Min"], 0, 120))))
    next_state["Distance Km"] = round(float(max(0, next_state["Distance Km"])), 2)
    next_state["Avg Heart Rate"] = int(round(float(np.clip(next_state["Avg Heart Rate"], 80, 190))))
    next_state["Max Heart Rate"] = int(
        round(float(np.clip(max(next_state["Max Heart Rate"], next_state["Avg Heart Rate"] + 5), 90, 220)))
    )
    next_state["Calories Burned"] = int(round(float(np.clip(next_state["Calories Burned"], 0, 1200))))

    return next_state


def synthetic_record_id(entity_number: Any, session_number: int) -> str:
    clean_entity = str(entity_number).strip().replace(" ", "_")
    return f"SYN-{clean_entity}-{session_number:02d}"


def generate_weekly_sessions(
    personal: pd.DataFrame, exercise: pd.DataFrame, health: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    ranges = source_ranges(exercise)
    latest = latest_exercise_by_member(exercise).set_index("Entity Number")
    health_by_member = health.drop_duplicates("Entity Number").set_index("Entity Number")

    records: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []

    for _, personal_row in personal.sort_values("Entity Number").iterrows():
        entity_number = personal_row["Entity Number"]
        if entity_number not in health_by_member.index:
            raise ValueError(f"Entity Number {entity_number} is missing from Health Data.")

        health_row = health_by_member.loc[entity_number]
        profile = assign_recovery_profile(rng)
        session_count = int(rng.integers(8, 13))
        setback_weeks: set[int] = set()
        if profile == "recovery with setbacks":
            setback_count = int(rng.choice([1, 2], p=[0.8, 0.2]))
            setback_weeks = set(rng.choice(range(2, session_count + 1), size=setback_count, replace=False))

        latest_row = latest.loc[entity_number] if entity_number in latest.index else None
        previous = create_starting_state(personal_row, health_row, latest_row, ranges)

        profile_records.append(
            {
                "Entity Number": entity_number,
                "Recovery Profile": profile,
                "Synthetic Sessions": session_count,
                "Has Setback": bool(setback_weeks),
                "Setback Weeks": ",".join(map(str, sorted(setback_weeks))),
            }
        )

        for session_number in range(1, session_count + 1):
            current = simulate_next_week(
                previous,
                personal_row,
                health_row,
                profile,
                session_number,
                setback_weeks,
                ranges,
                rng,
            )
            current["Exercise Record ID"] = synthetic_record_id(entity_number, session_number)
            current["Entity Number"] = entity_number
            current["Synthetic Session Number"] = session_number
            current["Recovery Profile"] = profile
            current["Is Synthetic"] = True
            current["RHR Imputed"] = False
            current["HRV Imputed"] = False
            current["Sleep Hours Imputed"] = False
            current["Sleep Quality Imputed"] = False

            record = {**personal_row.to_dict(), **health_row.to_dict(), **current}
            records.append(record)
            previous = current

    synthetic = pd.DataFrame(records)
    profiles = pd.DataFrame(profile_records)
    return synthetic, profiles


def validate_generated_data(
    synthetic: pd.DataFrame, personal: pd.DataFrame, profiles: pd.DataFrame
) -> list[str]:
    failures: list[str] = []

    required_columns = EXERCISE_COLUMNS + PERSONAL_COLUMNS + SYNTHETIC_COLUMNS
    missing_columns = [column for column in required_columns if column not in synthetic.columns]
    if missing_columns:
        failures.append(f"Missing required generated columns: {missing_columns}")

    if synthetic["Exercise Record ID"].duplicated().any():
        failures.append("Duplicate synthetic Exercise Record ID values found.")

    known_entities = set(personal["Entity Number"])
    unknown_entities = sorted(set(synthetic["Entity Number"]) - known_entities)
    if unknown_entities:
        failures.append(f"Synthetic rows contain unknown Entity Number values: {unknown_entities[:10]}")

    dates = synthetic.sort_values(["Entity Number", "Synthetic Session Number"])
    bad_date_members = []
    for entity_number, group in dates.groupby("Entity Number"):
        parsed_dates = pd.to_datetime(group["Record Date"], errors="coerce")
        if parsed_dates.isna().any() or not parsed_dates.is_monotonic_increasing:
            bad_date_members.append(entity_number)
        elif (parsed_dates.diff().dropna() <= pd.Timedelta(0)).any():
            bad_date_members.append(entity_number)
    if bad_date_members:
        failures.append(f"Dates do not increase for members: {bad_date_members[:10]}")

    range_checks = {
        "Pain Score": (0, 10),
        "RPE 1-10": (1, 10),
        "Sleep Hours": (3, 10),
        "Duration Min": (0, 120),
        "Distance Km": (0, np.inf),
        "Avg Heart Rate": (1, 220),
        "Max Heart Rate": (1, 240),
        "Resting Heart Rate": (1, 140),
        "Calories Burned": (0, 1500),
    }
    for column, (lower, upper) in range_checks.items():
        if column in synthetic:
            values = pd.to_numeric(synthetic[column], errors="coerce")
            if values.isna().any() or (values < lower).any() or (values > upper).any():
                failures.append(f"{column} contains missing or out-of-range values.")

    if not profiles["Has Setback"].mean() >= 0.10 or not profiles["Has Setback"].mean() <= 0.20:
        failures.append(
            "Setback member share is outside the requested 10% to 20% range: "
            f"{profiles['Has Setback'].mean():.2%}"
        )

    if (pd.to_numeric(synthetic["Duration Min"], errors="coerce") < 0).any():
        failures.append("Negative durations found.")
    if (pd.to_numeric(synthetic["Distance Km"], errors="coerce") < 0).any():
        failures.append("Negative distances found.")

    return failures


def original_with_metadata(
    exercise: pd.DataFrame, personal: pd.DataFrame, health: pd.DataFrame
) -> pd.DataFrame:
    original = exercise.merge(personal, on="Entity Number", how="left", validate="many_to_one")
    original = original.merge(health, on="Entity Number", how="left", validate="many_to_one")
    original["Is Synthetic"] = False
    original["Setback Flag"] = False
    for column in SYNTHETIC_COLUMNS:
        if column not in original.columns:
            original[column] = np.nan
    return original


def numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in SIMULATED_NUMERIC_COLUMNS:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            summary[column] = {
                "min": round(float(values.min()), 3),
                "max": round(float(values.max()), 3),
                "mean": round(float(values.mean()), 3),
                "median": round(float(values.median()), 3),
            }
    return summary


def save_outputs(
    synthetic: pd.DataFrame,
    combined: pd.DataFrame,
    profiles: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    synthetic.to_csv(OUTPUT_DIR / "synthetic_weekly_sessions.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "combined_exercise_data.csv", index=False)
    profiles.to_csv(OUTPUT_DIR / "member_recovery_profiles.csv", index=False)
    with (OUTPUT_DIR / "generation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Members processed: {summary['members_processed']}")
    print(f"Original exercise records: {summary['original_exercise_records']}")
    print(f"Synthetic sessions generated: {summary['synthetic_sessions_generated']}")
    print(f"Average sessions per member: {summary['average_sessions_per_member']:.2f}")

    print("\nRecovery profile counts:")
    for profile, count in summary["recovery_profile_counts"].items():
        print(f"{profile}: {count}")

    print(f"\nSetback sessions: {summary['setback_sessions']}")
    print("\nKey numeric field summary:")
    for column, values in summary["numeric_summary"].items():
        print(
            f"{column}: min={values['min']}, max={values['max']}, "
            f"mean={values['mean']}, median={values['median']}"
        )

    print("\nValidation failures:")
    if summary["validation_failures"]:
        for failure in summary["validation_failures"]:
            print(f"- {failure}")
    else:
        print("None")

    print("\nGenerated files:")
    for path in summary["generated_files"]:
        print(path)


def main() -> None:
    personal, exercise, health = load_source_data()
    synthetic, profiles = generate_weekly_sessions(personal, exercise, health)
    validation_failures = validate_generated_data(synthetic, personal, profiles)
    combined = pd.concat(
        [original_with_metadata(exercise, personal, health), synthetic],
        ignore_index=True,
        sort=False,
    )

    summary = {
        "synthetic_data_notice": (
            "Generated synthetic MVP data. This is not clinically validated."
        ),
        "members_processed": int(personal["Entity Number"].nunique()),
        "original_exercise_records": int(len(exercise)),
        "synthetic_sessions_generated": int(len(synthetic)),
        "average_sessions_per_member": float(len(synthetic) / personal["Entity Number"].nunique()),
        "recovery_profile_counts": profiles["Recovery Profile"].value_counts().to_dict(),
        "setback_sessions": int(synthetic["Setback Flag"].sum()),
        "numeric_summary": numeric_summary(synthetic),
        "validation_failures": validation_failures,
        "generated_files": [
            str((OUTPUT_DIR / "synthetic_weekly_sessions.csv").resolve()),
            str((OUTPUT_DIR / "combined_exercise_data.csv").resolve()),
            str((OUTPUT_DIR / "member_recovery_profiles.csv").resolve()),
            str((OUTPUT_DIR / "generation_summary.json").resolve()),
        ],
    }

    save_outputs(synthetic, combined, profiles, summary)
    print_summary(summary)

    if validation_failures:
        raise SystemExit("Synthetic data generation completed with validation failures.")


if __name__ == "__main__":
    main()
