from __future__ import annotations

import argparse
import calendar
from datetime import date
import json
import math
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import nnls
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TARGET_COL = "APPLICATIONS"
STATE_COL = "STATE_CD"
DIVISION_COL = "Division"
YEAR_COL = "ISO_YEAR"
WEEK_COL = "ISO_WEEK"

NON_DUMMY_PREDICTORS = [
    "DSP",
    "LeadGen",
    "Paid Search",
    "Paid Social",
    "Prescreen",
    "Referrals",
]
DEFAULT_MEDIA_PREDICTORS = list(NON_DUMMY_PREDICTORS)

DUMMY_FAMILIES: Dict[str, List[str]] = {
    "weekly": [f"W_{idx}" for idx in range(1, 53)],
    "f_dummy": [f"F_{idx}" for idx in range(26)],
}
FOURIER_COLS = ["sin_1", "cos_1", "sin_2", "cos_2"]
FEATURE_RUNS: Dict[str, Dict[str, object]] = {
    "weekly": {
        "extra_cols": DUMMY_FAMILIES["weekly"],
        "drop_one": True,
        "scaler": "minmax",
    },
    "f_dummy": {
        "extra_cols": DUMMY_FAMILIES["f_dummy"],
        "drop_one": True,
        "scaler": "minmax",
    },
    "fourier": {
        "extra_cols": FOURIER_COLS,
        "drop_one": False,
        "scaler": "standard",
    },
}

TRAIN_YEARS = {2024, 2025}
TEST_YEAR = 2026
TEST_WEEKS = set(range(1, 9))
EPSILON = 1e-9
BACKTEST_MODES = {"fixed_holdout", "rolling_one_step_expanding", "rolling_one_step_fixed_window"}
DEFAULT_FIXED_WINDOW_WEEKS = 104

TIME_INDEX_COL = "time_index"
TIME_INDEX_SQ_COL = "time_index_sq"
PRESCREEN_LAG1_COL = "Prescreen_lag1"
DSP_LAG1_COL = "DSP_lag1"
PAID_SEARCH_LAG1_COL = "Paid_Search_lag1"
DSP_TRAILING_4W_AVG_COL = "DSP_trailing_4w_avg"
PAID_SEARCH_TRAILING_4W_AVG_COL = "Paid_Search_trailing_4w_avg"
PRESCREEN_TRAILING_4W_AVG_COL = "Prescreen_trailing_4w_avg"
YEAR_INDICATOR_2025_COL = "year_indicator_2025"
YEAR_INDICATOR_2026_COL = "year_indicator_2026"
TARGET_LAG1_COL = f"{TARGET_COL}_lag1"
TARGET_TRAILING_4W_AVG_COL = f"{TARGET_COL}_trailing_4w_avg"

OPTIONAL_FEATURES: Dict[str, str] = {
    TIME_INDEX_COL: "Sequential week counter within each state or aggregated division series.",
    TIME_INDEX_SQ_COL: "Squared weekly trend term to capture curved long-run growth or decline.",
    YEAR_INDICATOR_2025_COL: "Indicator equal to 1 for 2025 rows and 0 otherwise.",
    YEAR_INDICATOR_2026_COL: "Indicator equal to 1 for 2026 rows and 0 otherwise.",
    PRESCREEN_LAG1_COL: "Prior week's Prescreen volume.",
    DSP_LAG1_COL: "Prior week's DSP volume.",
    PAID_SEARCH_LAG1_COL: "Prior week's Paid Search volume.",
    DSP_TRAILING_4W_AVG_COL: "Rolling 4-week average of DSP, inclusive of the current week.",
    PAID_SEARCH_TRAILING_4W_AVG_COL: "Rolling 4-week average of Paid Search, inclusive of the current week.",
    PRESCREEN_TRAILING_4W_AVG_COL: "Rolling 4-week average of Prescreen, inclusive of the current week.",
    TARGET_LAG1_COL: "Prior week's observed target value.",
    TARGET_TRAILING_4W_AVG_COL: "Rolling 4-week average of the observed target using only prior weeks.",
}
SATURATION_METHODS = {"none", "log1p"}
PLOTTABLE_TARGET_COLUMNS = ["APPLICATIONS", "NON_DM_APPLICATIONS"]
FUTURE_SPEND_RAW_REQUIRED_COLUMNS = [
    "DETAIL_TACTIC",
    "BUSINESS_DATE",
    "STATE_CD",
    "TOTAL_COST",
]
FUTURE_SPEND_EXTRA_TACTICS = ["Sweepstakes"]
FUTURE_SPEND_OUTPUT_TACTICS = list(NON_DUMMY_PREDICTORS) + FUTURE_SPEND_EXTRA_TACTICS


def target_lag_col(target_col: str) -> str:
    return f"{target_col}_lag1"


def target_trailing_avg_col(target_col: str) -> str:
    return f"{target_col}_trailing_4w_avg"


def prepare_future_spend_frame(
    raw_df: pd.DataFrame,
    tactic_alias_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Prepare a raw future spend extract into the model-ready weekly spend matrix.

    Accepts two input layouts:

    1. **Date-based** (original format) — ISO_YEAR and ISO_WEEK are derived from
       BUSINESS_DATE:
           DETAIL_TACTIC, BUSINESS_DATE, STATE_CD, TOTAL_COST

    2. **Pre-computed** (output of :func:`spread_monthly_spend_to_weekly`) — ISO_YEAR
       and ISO_WEEK are taken directly from the input; BUSINESS_DATE is optional:
           DETAIL_TACTIC, STATE_CD, TOTAL_COST, ISO_YEAR, ISO_WEEK

    When ISO_YEAR and ISO_WEEK are both present they always take precedence over
    BUSINESS_DATE.  Rows where ISO_WEEK is null (e.g. Prescreen monthly passthrough
    rows from :func:`spread_monthly_spend_to_weekly`) are dropped before aggregation.

    The returned frame contains:
        - ``ISO_YEAR``, ``ISO_WEEK``, ``STATE_CD``
        - one column per expected tactic: ``DSP``, ``LeadGen``, ``Paid Search``,
          ``Paid Social``, ``Prescreen``, ``Referrals``, ``Sweepstakes``

    Any expected tactic absent from the input is included in the output with 0.
    Pass ``tactic_alias_map`` to normalise raw DETAIL_TACTIC labels before aggregation.
    """
    cols = set(raw_df.columns)
    has_iso = {YEAR_COL, WEEK_COL}.issubset(cols)
    has_date = "BUSINESS_DATE" in cols

    base_required = {"DETAIL_TACTIC", "STATE_CD", "TOTAL_COST"}
    missing_base = sorted(base_required - cols)
    if missing_base:
        raise ValueError(f"prepare_future_spend_frame: missing required columns: {missing_base}")
    if not has_iso and not has_date:
        raise ValueError(
            "prepare_future_spend_frame: supply either BUSINESS_DATE or both "
            f"{YEAR_COL} and {WEEK_COL}."
        )

    keep = list(base_required)
    if has_iso:
        keep += [YEAR_COL, WEEK_COL]
    if has_date:
        keep.append("BUSINESS_DATE")

    spend_df = raw_df[keep].copy()
    spend_df = spend_df.drop_duplicates().reset_index(drop=True)
    spend_df["STATE_CD"] = spend_df["STATE_CD"].astype(str)
    spend_df["DETAIL_TACTIC"] = spend_df["DETAIL_TACTIC"].astype(str)
    spend_df["TOTAL_COST"] = pd.to_numeric(spend_df["TOTAL_COST"], errors="coerce").fillna(0.0)

    if tactic_alias_map:
        spend_df["DETAIL_TACTIC"] = spend_df["DETAIL_TACTIC"].replace(tactic_alias_map)

    if has_iso:
        spend_df[YEAR_COL] = pd.to_numeric(spend_df[YEAR_COL], errors="coerce")
        spend_df[WEEK_COL] = pd.to_numeric(spend_df[WEEK_COL], errors="coerce")
    else:
        spend_df["BUSINESS_DATE"] = pd.to_datetime(spend_df["BUSINESS_DATE"], errors="coerce")
        spend_df = spend_df.dropna(subset=["BUSINESS_DATE"]).copy()
        iso_calendar = spend_df["BUSINESS_DATE"].dt.isocalendar()
        spend_df[YEAR_COL] = iso_calendar.year.astype(float)
        spend_df[WEEK_COL] = iso_calendar.week.astype(float)

    # Drop rows with no valid ISO week (e.g. Prescreen monthly passthrough rows).
    spend_df = spend_df.dropna(subset=[YEAR_COL, WEEK_COL, "STATE_CD", "DETAIL_TACTIC"]).copy()
    spend_df[YEAR_COL] = spend_df[YEAR_COL].astype(int)
    spend_df[WEEK_COL] = spend_df[WEEK_COL].astype(int)

    grouped = (
        spend_df[[YEAR_COL, WEEK_COL, "DETAIL_TACTIC", STATE_COL, "TOTAL_COST"]]
        .groupby([YEAR_COL, WEEK_COL, "DETAIL_TACTIC", STATE_COL], as_index=False)["TOTAL_COST"]
        .sum()
    )

    pivoted = (
        grouped.pivot_table(
            index=[YEAR_COL, WEEK_COL, STATE_COL],
            columns="DETAIL_TACTIC",
            values="TOTAL_COST",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivoted.columns.name = None

    for tactic_col in FUTURE_SPEND_OUTPUT_TACTICS:
        if tactic_col not in pivoted.columns:
            pivoted[tactic_col] = 0.0

    ordered_cols = [YEAR_COL, WEEK_COL, STATE_COL, *FUTURE_SPEND_OUTPUT_TACTICS]
    prepared = pivoted[ordered_cols].copy()
    prepared = prepared.sort_values([YEAR_COL, WEEK_COL, STATE_COL]).reset_index(drop=True)
    return prepared


def prepare_future_spend_data(
    input_path: Union[str, Path, pd.DataFrame],
    output_path: Optional[str] = None,
    tactic_alias_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Read, standardize, and optionally write a future spend file for forecasting.

    This is the file-based wrapper around :func:`prepare_future_spend_frame`. Use it as the
    first step when a future spend file arrives in the raw format:
        ``DETAIL_TACTIC, BUSINESS_DATE, STATE_CD, TOTAL_COST``

    The prepared output follows the requested weekly wide layout:
        ``ISO_YEAR, ISO_WEEK, STATE_CD, DSP, LeadGen, Paid Search, Paid Social,
        Prescreen, Referrals, Sweepstakes``

    Parameters:
        input_path: Raw future spend input, provided either as a CSV path or a pandas
            DataFrame.
        output_path: Optional path where the prepared spend file should be written as CSV.
            If omitted, the function only returns the prepared DataFrame.
        tactic_alias_map: Optional mapping from raw tactic names to desired output names.

    Returns:
        The prepared weekly spend DataFrame.

    Example:
        spend_df = prepare_future_spend_data(
            input_path="/path/to/raw_future_spend.csv",
            output_path="/path/to/prepared_future_spend.csv",
            tactic_alias_map={"PaidSearch": "Paid Search", "PaidSocial": "Paid Social"},
        )

    Example with a DataFrame:
        spend_df = prepare_future_spend_data(
            input_path=raw_spend_df,
            tactic_alias_map={"PaidSearch": "Paid Search", "PaidSocial": "Paid Social"},
        )
    """
    raw_df = load_tabular_input(input_path, "Raw future spend input")
    prepared = prepare_future_spend_frame(raw_df, tactic_alias_map=tactic_alias_map)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(output_file, index=False)

    return prepared


def spread_monthly_spend_to_weekly(
    df: pd.DataFrame,
    monthly_tactics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convert a monthly spend file to weekly spend using pro-rata day allocation.

    Each calendar month is split across the ISO weeks that overlap it.  A week that
    falls entirely within one month receives the full 7/days_in_month share; a week
    that straddles a month boundary receives only the days that belong to that month.
    This mirrors the inverse of :func:`roll_up_weekly_forecast_to_monthly`.

    Tactics listed in ``monthly_tactics`` (default: ``["Prescreen"]``) are kept as a
    single row per month rather than being spread across weeks.  Their TOTAL_COST is
    unchanged and ISO_WEEK is left as None.

    Parameters:
        df: Monthly spend data with columns:
                DETAIL_TACTIC, BUSINESS_DATE, STATE_CD, TOTAL_COST
        monthly_tactics: DETAIL_TACTIC values to retain as one row per month.
            Defaults to ``["Prescreen"]``.

    Returns:
        DataFrame with columns:
            DETAIL_TACTIC, STATE_CD, BUSINESS_DATE,
            ISO_YEAR, ISO_WEEK, DAYS_ALLOCATED, TOTAL_COST

        Spread tactics produce one row per ISO week; monthly_tactics produce one
        row per original input row with ISO_WEEK = None.
    """
    if monthly_tactics is None:
        monthly_tactics = ["Prescreen"]

    required = {"DETAIL_TACTIC", "BUSINESS_DATE", "STATE_CD", "TOTAL_COST"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"spread_monthly_spend_to_weekly: missing required columns: {missing}")

    work = df[["DETAIL_TACTIC", "BUSINESS_DATE", "STATE_CD", "TOTAL_COST"]].copy()
    work["DETAIL_TACTIC"] = work["DETAIL_TACTIC"].astype(str)
    work["STATE_CD"] = work["STATE_CD"].astype(str)
    work["TOTAL_COST"] = pd.to_numeric(work["TOTAL_COST"], errors="coerce").fillna(0.0)

    parsed_dates = pd.to_datetime(work["BUSINESS_DATE"], errors="coerce")
    work["_cal_year"] = parsed_dates.dt.year
    work["_cal_month"] = parsed_dates.dt.month
    work = work.dropna(subset=["_cal_year", "_cal_month"]).copy()
    work["_cal_year"] = work["_cal_year"].astype(int)
    work["_cal_month"] = work["_cal_month"].astype(int)

    col_order = ["DETAIL_TACTIC", "STATE_CD", "BUSINESS_DATE", "ISO_YEAR", "ISO_WEEK", "DAYS_ALLOCATED", "TOTAL_COST"]

    spread_mask = ~work["DETAIL_TACTIC"].isin(monthly_tactics)
    spread_df = work[spread_mask].copy()
    monthly_df = work[~spread_mask].copy()

    # ── Weekly-spread rows ────────────────────────────────────────────────────
    weekly_rows: List[Dict[str, Any]] = []
    for _, row in spread_df.iterrows():
        cal_year = int(row["_cal_year"])
        cal_month = int(row["_cal_month"])
        monthly_cost = float(row["TOTAL_COST"])
        days_in_month = calendar.monthrange(cal_year, cal_month)[1]

        week_day_counts: Dict[Tuple[int, int], int] = {}
        for day_num in range(1, days_in_month + 1):
            d = date(cal_year, cal_month, day_num)
            iso = d.isocalendar()
            week_key = (int(iso.year), int(iso.week))
            week_day_counts[week_key] = week_day_counts.get(week_key, 0) + 1

        for (iso_year, iso_week), day_count in week_day_counts.items():
            weekly_rows.append({
                "DETAIL_TACTIC": row["DETAIL_TACTIC"],
                "STATE_CD": row["STATE_CD"],
                "BUSINESS_DATE": row["BUSINESS_DATE"],
                "ISO_YEAR": iso_year,
                "ISO_WEEK": iso_week,
                "DAYS_ALLOCATED": day_count,
                "TOTAL_COST": monthly_cost * (day_count / days_in_month),
            })

    # ── Monthly passthrough rows (e.g. Prescreen) ─────────────────────────────
    monthly_rows: List[Dict[str, Any]] = []
    for _, row in monthly_df.iterrows():
        biz_date = pd.to_datetime(row["BUSINESS_DATE"], errors="coerce")
        iso = biz_date.isocalendar() if not pd.isna(biz_date) else None
        monthly_rows.append({
            "DETAIL_TACTIC": row["DETAIL_TACTIC"],
            "STATE_CD": row["STATE_CD"],
            "BUSINESS_DATE": row["BUSINESS_DATE"],
            "ISO_YEAR": int(iso.year) if iso is not None else int(row["_cal_year"]),
            "ISO_WEEK": int(iso.week) if iso is not None else None,
            "DAYS_ALLOCATED": calendar.monthrange(int(row["_cal_year"]), int(row["_cal_month"]))[1],
            "TOTAL_COST": float(row["TOTAL_COST"]),
        })

    all_rows = weekly_rows + monthly_rows
    if not all_rows:
        return pd.DataFrame(columns=col_order)

    result = pd.DataFrame(all_rows)[col_order]
    result = result.sort_values(
        ["STATE_CD", "DETAIL_TACTIC", "ISO_YEAR", "ISO_WEEK"],
        na_position="last",
    ).reset_index(drop=True)
    return result


@dataclass
class RunMetadata:
    """Metadata saved with each trained model artifact or shown during inline review.

    Attributes:
        scope: Either ``state`` or ``division``.
        entity: State code or division name being modeled.
        model_type: ``OLS`` or ``NNLS``.
        dummy_family: One of ``weekly``, ``f_dummy``, or ``fourier``.
        dropped_dummy: Baseline dummy removed to avoid collinearity when a dummy family is used.
        train_rows: Number of training observations.
        test_rows: Number of holdout observations.
        predictors: Final predictor list after combining the default variables, seasonal terms,
            and any user-selected optional engineered variables.
        scaler_type: ``minmax`` for the default seasonal runs and ``standard`` for the Fourier run.
        backtest_mode: Backtest style used for evaluation.
        media_transform_config: Mapping of raw media variables to the transform settings used.
        target_col: Dependent variable used in the model.
    """

    scope: str
    entity: str
    model_type: str
    dummy_family: str
    dropped_dummy: Optional[str]
    train_rows: int
    test_rows: int
    predictors: List[str]
    scaler_type: str
    backtest_mode: str
    media_transform_config: Dict[str, Dict[str, Any]]
    target_col: str


class ForecastSkipError(Exception):
    """Non-fatal forecast skip used when one model slice cannot be scored."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The command-line interface keeps the current file-writing behavior as the default.
    For notebook usage, prefer importing :func:`run_model_pipeline` and passing keyword
    arguments directly instead of going through the CLI.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build OLS and NNLS models for each State and Division using 2024-2025 data, "
            "then test on 2026 weeks 1-8."
        )
    )
    parser.add_argument(
        "--input",
        default="/Users/Rahul/Desktop/Code/Working Codebase/ModelingFile_Digital.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--target-col",
        default=TARGET_COL,
        help="Dependent variable to model.",
    )
    parser.add_argument(
        "--dataset-group-by",
        nargs="*",
        default=None,
        help="Optional additional grouping keys such as CHANNEL_CD or DETAIL_TACTIC.",
    )
    parser.add_argument(
        "--output-dir",
        default="state_division_model_artifacts",
        help="Directory where diagnostics, models, scalers, and coefficient outputs will be saved.",
    )
    parser.add_argument(
        "--inline-output",
        action="store_true",
        help="Display inline notebook-style output instead of writing files. Requires selected states/divisions.",
    )
    parser.add_argument(
        "--selected-states",
        nargs="*",
        default=None,
        help="Optional list of state codes to run.",
    )
    parser.add_argument(
        "--selected-divisions",
        nargs="*",
        default=None,
        help="Optional list of division names to run.",
    )
    parser.add_argument(
        "--methodologies",
        nargs="*",
        default=None,
        help="Optional list from: OLS, NNLS, weekly, f_dummy, Fourier.",
    )
    parser.add_argument(
        "--optional-features",
        nargs="*",
        default=None,
        help=f"Optional engineered variables to add. Choices: {', '.join(sorted(OPTIONAL_FEATURES))}.",
    )
    parser.add_argument(
        "--media-predictors",
        nargs="*",
        default=None,
        help=f"Optional subset of base media predictors. Choices: {', '.join(NON_DUMMY_PREDICTORS)}.",
    )
    parser.add_argument(
        "--backtest-mode",
        default="fixed_holdout",
        choices=sorted(BACKTEST_MODES),
        help="Backtest style: fixed_holdout, rolling_one_step_expanding, or rolling_one_step_fixed_window.",
    )
    parser.add_argument(
        "--fixed-window-weeks",
        type=int,
        default=DEFAULT_FIXED_WINDOW_WEEKS,
        help="Training window length for rolling_one_step_fixed_window.",
    )
    return parser.parse_args()


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def group_by_suffix(dataset_group_by: Sequence[str]) -> str:
    """Build a filesystem-safe suffix from the selected group-by variables."""
    if not dataset_group_by:
        return ""
    cleaned = [safe_name(col) for col in dataset_group_by if str(col).strip()]
    return f"__group_by__{'__'.join(cleaned)}" if cleaned else ""


def format_model_key(grouping_keys: Sequence[str], entity_key: Sequence[object]) -> str:
    """Create the human-readable model key used across outputs."""
    return " | ".join(f"{col}={value}" for col, value in zip(grouping_keys, entity_key))


def _parse_key_string(key_str: object) -> Dict[str, str]:
    """Split a pipe-delimited 'COL=value | COL=value' Key string into a dict."""
    if not isinstance(key_str, str) or not key_str.strip():
        return {}
    out: Dict[str, str] = {}
    for segment in key_str.split(" | "):
        if "=" in segment:
            col, _, val = segment.partition("=")
            out[col.strip()] = val.strip()
    return out


def _key_col_label(col_name: str) -> str:
    """Convert a raw Key column name to a human-readable label."""
    label = re.sub(r"_CD$", "", col_name)   # STATE_CD -> STATE, CHANNEL_CD -> CHANNEL
    return label.replace("_", " ").strip().title()


def enrich_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the Key column and add a Modeling_Grain column.

    Parses every pipe-delimited 'COL=value' segment in the Key column into its own
    column, placed immediately after Key in sorted order.  A 'Modeling_Grain' column
    (e.g. 'By Customer Type, State') is inserted directly after those key columns.

    STATE_CD and CHANNEL_CD are intentionally excluded from expansion because the
    fixed output columns State and Channel already carry those values.

    Previously expanded key columns and any existing Modeling_Grain column are dropped
    first, so this function is idempotent and safe to call after pd.concat regardless
    of how many files contributed different key structures.
    """
    _SKIP_KEY_COLS = {"STATE_CD", "CHANNEL_CD"}

    if "Key" not in df.columns or df.empty:
        return df
    parsed = df["Key"].apply(_parse_key_string)
    key_cols = sorted({k for d in parsed for k in d} - _SKIP_KEY_COLS)
    if not key_cols:
        return df
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Per-row Modeling_Grain derived from the keys present in that row's Key string.
    def _grain(d: dict) -> str:
        labels = sorted(_key_col_label(k) for k in d)
        return ("By " + ", ".join(labels)) if labels else ""

    modeling_grain = parsed.apply(_grain)

    # Drop stale copies so re-insertion is always clean and correctly positioned.
    stale = [c for c in key_cols if c in df.columns]
    if "Modeling_Grain" in df.columns:
        stale.append("Modeling_Grain")
    result = df.drop(columns=stale).copy()

    insert_pos = list(result.columns).index("Key") + 1
    for offset, col in enumerate(key_cols):
        result.insert(insert_pos + offset, col, parsed_df[col])
    result.insert(insert_pos + len(key_cols), "Modeling_Grain", modeling_grain)
    return result


def consolidate_forecast_output_files(output_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Stack forecast output CSVs across multiple group-by runs.

    This consolidates all files that match these families inside ``output_dir``:
        - ``future_forecast__group_by__*.csv``
        - ``model_coefficients__group_by__*.csv``
        - ``model_training_summary__group_by__*.csv``
        - ``monthly_forecast__group_by__*.csv``

    For each family, a consolidated CSV is written back into ``output_dir``:
        - ``weekly_forecast_consolidated.csv``
        - ``model_coefficients_consolidated.csv``
        - ``model_training_summary_consolidated.csv``
        - ``monthly_forecast_consolidated.csv``

    Returns:
        A dictionary of the consolidated DataFrames keyed by output family.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # lead_columns: desired column order for the first N columns of that family's output.
    # Columns absent from the data are skipped; all remaining columns follow in their
    # natural order.
    file_specs = {
        "future_forecast": {
            "pattern": "future_forecast__group_by__*.csv",
            "output_name": "weekly_forecast_consolidated.csv",
            "lead_columns": [
                "Key",
                "Customer_Type",
                "State",
                "Channel",
                "H_Tactic",
                "Detail_Tactic",
                "Product",
            ],
        },
        "model_coefficients": {
            "pattern": "model_coefficients__group_by__*.csv",
            "output_name": "model_coefficients_consolidated.csv",
        },
        "model_training_summary": {
            "pattern": "model_training_summary__group_by__*.csv",
            "output_name": "model_training_summary_consolidated.csv",
        },
        "monthly_forecast": {
            "pattern": "monthly_forecast__group_by__*.csv",
            "output_name": "monthly_forecast_consolidated.csv",
        },
        "product_factors": {
            "pattern": "product_factors__group_by__*.csv",
            "output_name": "product_factors_consolidated.csv",
        },
    }

    consolidated_outputs: Dict[str, pd.DataFrame] = {}
    for output_key, spec in file_specs.items():
        matching_files = sorted(output_root.glob(spec["pattern"]))
        frames: List[pd.DataFrame] = []
        for file_path in matching_files:
            frame = pd.read_csv(file_path)
            frame["source_file"] = file_path.name
            frames.append(frame)

        consolidated_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        consolidated_df = enrich_key_columns(consolidated_df)

        lead_columns = spec.get("lead_columns")
        if lead_columns and not consolidated_df.empty:
            present_lead = [c for c in lead_columns if c in consolidated_df.columns]
            rest = [c for c in consolidated_df.columns if c not in lead_columns]
            consolidated_df = consolidated_df[present_lead + rest]

        if not consolidated_df.empty:
            consolidated_df.to_csv(output_root / spec["output_name"], index=False)
        consolidated_outputs[output_key] = consolidated_df

    return consolidated_outputs


def build_product_allocation_factors(
    history_input_source: Union[str, Path, pd.DataFrame],
    dataset_group_by: Sequence[str],
    selected_states: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build weighted product allocation factors from historical applications data.

    PRODUCT_FUNDED is not part of the modeling grain. Instead, the historical file is used
    to estimate how total applications should be split across products, and how approvals
    and originations should convert from those product-level applications.
    """
    raw_history = load_tabular_input(history_input_source, "Historical product allocation input")
    required_cols = {
        YEAR_COL,
        WEEK_COL,
        STATE_COL,
        "PRODUCT_FUNDED",
        "APPLICATIONS",
        "APPROVED",
        "ORIGINATIONS",
        *dataset_group_by,
    }
    missing = sorted(required_cols - set(raw_history.columns))
    if missing:
        return pd.DataFrame()

    frame = raw_history.copy()
    frame[YEAR_COL] = pd.to_numeric(frame[YEAR_COL], errors="coerce")
    frame[WEEK_COL] = pd.to_numeric(frame[WEEK_COL], errors="coerce")
    for metric_col in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"]:
        frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce").fillna(0.0)
    frame[STATE_COL] = frame[STATE_COL].astype(str)
    frame["PRODUCT_FUNDED"] = frame["PRODUCT_FUNDED"].astype(str)
    for col in dataset_group_by:
        frame[col] = frame[col].astype(str)

    frame = frame.dropna(subset=[YEAR_COL, WEEK_COL, STATE_COL, "PRODUCT_FUNDED"]).copy()
    frame[YEAR_COL] = frame[YEAR_COL].astype(int)
    frame[WEEK_COL] = frame[WEEK_COL].astype(int)
    if selected_states:
        frame = prepare_entity_subset(frame, STATE_COL, selected_states)
    if frame.empty:
        return pd.DataFrame()

    grouping_keys = list(dict.fromkeys([STATE_COL, *dataset_group_by]))
    weekly_group_cols = [*grouping_keys, YEAR_COL, WEEK_COL]
    detail = (
        frame[weekly_group_cols + ["PRODUCT_FUNDED", "APPLICATIONS", "APPROVED", "ORIGINATIONS"]]
        .groupby(weekly_group_cols + ["PRODUCT_FUNDED"], as_index=False, dropna=False)
        .sum(min_count=1)
    )
    product_totals = (
        detail[grouping_keys + ["PRODUCT_FUNDED", "APPLICATIONS", "APPROVED", "ORIGINATIONS"]]
        .groupby(grouping_keys + ["PRODUCT_FUNDED"], as_index=False, dropna=False)
        .sum(min_count=1)
    )
    model_totals = (
        detail[grouping_keys + ["APPLICATIONS"]]
        .groupby(grouping_keys, as_index=False, dropna=False)
        .sum(min_count=1)
        .rename(columns={"APPLICATIONS": "TOTAL_APPLICATIONS"})
    )
    # Key-level approval and origination totals (summed across all products)
    key_approval_totals = (
        product_totals.groupby(grouping_keys, as_index=False)[["APPROVED", "ORIGINATIONS"]]
        .sum(min_count=1)
        .rename(columns={"APPROVED": "TOTAL_APPROVED", "ORIGINATIONS": "TOTAL_ORIGINATIONS"})
    )
    factors = product_totals.merge(model_totals, on=grouping_keys, how="left")
    factors = factors.merge(key_approval_totals, on=grouping_keys, how="left")
    factors["APPLICATION_SHARE"] = np.where(
        factors["TOTAL_APPLICATIONS"].abs() > EPSILON,
        factors["APPLICATIONS"] / factors["TOTAL_APPLICATIONS"],
        0.0,
    )
    # Approval and origination rates are key-level (same across all products for a key)
    # PRODUCT_FUNDED only records originated loans so product-level rates are not meaningful
    factors["APPROVAL_RATE"] = np.where(
        factors["TOTAL_APPLICATIONS"].abs() > EPSILON,
        factors["TOTAL_APPROVED"] / factors["TOTAL_APPLICATIONS"],
        0.0,
    )
    factors["ORIGINATION_RATE"] = np.where(
        factors["TOTAL_APPLICATIONS"].abs() > EPSILON,
        factors["TOTAL_ORIGINATIONS"] / factors["TOTAL_APPLICATIONS"],
        0.0,
    )
    factors["Key"] = factors[grouping_keys].apply(lambda row: format_model_key(grouping_keys, tuple(row)), axis=1)
    return factors[["Key", "PRODUCT_FUNDED", "APPLICATION_SHARE", "APPROVAL_RATE", "ORIGINATION_RATE"]].copy()


def apply_product_allocation_to_forecast(
    forecast_df: pd.DataFrame,
    product_factors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Allocate predicted applications, approvals, and originations across products."""
    if forecast_df.empty or product_factors_df.empty:
        return forecast_df

    enriched = forecast_df.copy()
    success_mask = enriched["Run_Status"].astype(str) == "SUCCESS"
    if not success_mask.any():
        return enriched

    product_names = sorted(product_factors_df["PRODUCT_FUNDED"].dropna().astype(str).unique())
    for product_name in product_names:
        product_label = safe_name(product_name)
        enriched[f"APPLICATIONS_{product_label}"] = np.nan
        enriched[f"APPROVAL_{product_label}"] = np.nan
        enriched[f"ORIGINATIONS_{product_label}"] = np.nan

    enriched["Allocated_Approved"] = np.nan
    enriched["Allocated_Originations"] = np.nan

    factor_lookup = {
        key: group.copy()
        for key, group in product_factors_df.groupby("Key", dropna=False)
    }

    for row_idx in enriched.index[success_mask]:
        row_key = enriched.at[row_idx, "Key"]
        if row_key not in factor_lookup:
            continue
        predicted_apps_raw = pd.to_numeric(pd.Series([enriched.at[row_idx, "Predicted APPS Raw"]]), errors="coerce").iloc[0]
        if pd.isna(predicted_apps_raw):
            predicted_apps_raw = 0.0
        total_approved = 0.0
        total_originations = 0.0
        for _, factor_row in factor_lookup[row_key].iterrows():
            product_label = safe_name(factor_row["PRODUCT_FUNDED"])
            allocated_apps = float(predicted_apps_raw) * float(factor_row["APPLICATION_SHARE"])
            allocated_approved = float(predicted_apps_raw) * float(factor_row["APPROVAL_RATE"])
            allocated_originations = float(predicted_apps_raw) * float(factor_row["ORIGINATION_RATE"])
            enriched.at[row_idx, f"APPLICATIONS_{product_label}"] = allocated_apps
            enriched.at[row_idx, f"APPROVAL_{product_label}"] = allocated_approved
            enriched.at[row_idx, f"ORIGINATIONS_{product_label}"] = allocated_originations
            total_approved += allocated_approved
            total_originations += allocated_originations
        enriched.at[row_idx, "Allocated_Approved"] = total_approved
        enriched.at[row_idx, "Allocated_Originations"] = total_originations

    return enriched


def load_tabular_input(input_source: Union[str, Path, pd.DataFrame], source_name: str) -> pd.DataFrame:
    """Load a tabular input from either a CSV path or an in-memory DataFrame.

    Parameters:
        input_source: Either a CSV path or a pandas DataFrame.
        source_name: Friendly label used in validation error messages.

    Returns:
        A copy of the input DataFrame.
    """
    if isinstance(input_source, pd.DataFrame):
        return input_source.copy()
    if isinstance(input_source, (str, Path)):
        return pd.read_csv(input_source)
    raise TypeError(
        f"{source_name} must be a pandas DataFrame or a CSV file path, got {type(input_source).__name__}."
    )


def aggregate_input_dataset(
    df: pd.DataFrame,
    target_col: str,
    dataset_group_by: Sequence[str],
) -> pd.DataFrame:
    """Aggregate duplicate input rows at the requested grain.

    Rows are grouped by core identifiers plus any user-provided ``dataset_group_by`` columns
    and any available seasonal columns. The target and media variables are summed.
    """
    available_weekly = [col for col in DUMMY_FAMILIES["weekly"] if col in df.columns]
    available_f_dummy = [col for col in DUMMY_FAMILIES["f_dummy"] if col in df.columns]
    available_fourier = [col for col in FOURIER_COLS if col in df.columns]

    grouping_cols = [YEAR_COL, WEEK_COL, STATE_COL, DIVISION_COL, *dataset_group_by]
    grouping_cols.extend(available_weekly)
    grouping_cols.extend(available_f_dummy)
    grouping_cols.extend(available_fourier)
    grouping_cols = list(dict.fromkeys(grouping_cols))

    sum_cols = [target_col] + list(NON_DUMMY_PREDICTORS)
    sum_cols = [col for col in sum_cols if col in df.columns]

    aggregated = (
        df[grouping_cols + sum_cols]
        .groupby(grouping_cols, as_index=False, dropna=False)
        .sum(min_count=1)
    )
    return aggregated


def add_derived_seasonal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive seasonal columns from ISO week when they are not already present.

    This keeps future scoring robust when the incoming spend file only contains the core
    weekly identifiers and spend columns, but the trained models use weekly dummies,
    bi-weekly ``F_*`` dummies, or Fourier seasonality terms.
    """
    frame = df.copy()
    if WEEK_COL not in frame.columns:
        return frame

    iso_week = pd.to_numeric(frame[WEEK_COL], errors="coerce").fillna(0).astype(int)
    iso_week = iso_week.clip(lower=1, upper=53)

    for week_num in range(1, 53):
        col_name = f"W_{week_num}"
        if col_name not in frame.columns:
            frame[col_name] = (iso_week == week_num).astype(float)

    biweek_index = ((iso_week - 1) // 2).clip(lower=0, upper=25)
    for idx in range(26):
        col_name = f"F_{idx}"
        if col_name not in frame.columns:
            frame[col_name] = (biweek_index == idx).astype(float)

    week_position = (iso_week.astype(float) - 1.0) / 52.0
    if "sin_1" not in frame.columns:
        frame["sin_1"] = np.sin(2.0 * np.pi * week_position)
    if "cos_1" not in frame.columns:
        frame["cos_1"] = np.cos(2.0 * np.pi * week_position)
    if "sin_2" not in frame.columns:
        frame["sin_2"] = np.sin(4.0 * np.pi * week_position)
    if "cos_2" not in frame.columns:
        frame["cos_2"] = np.cos(4.0 * np.pi * week_position)

    return frame


def load_data(
    input_source: Union[str, Path, pd.DataFrame],
    target_col: str = TARGET_COL,
    dataset_group_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and validate the modeling dataset.

    Required columns include the selected target, the six default non-dummy marketing
    variables, and the core identifiers. Weekly, ``F_*``, and Fourier columns are optional.
    """
    df = load_tabular_input(input_source, "Modeling input")
    dataset_group_by = list(dataset_group_by or [])

    required_cols = {
        YEAR_COL,
        WEEK_COL,
        STATE_COL,
        DIVISION_COL,
        target_col,
        *NON_DUMMY_PREDICTORS,
        *dataset_group_by,
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[WEEK_COL] = pd.to_numeric(df[WEEK_COL], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    numeric_cols = list(NON_DUMMY_PREDICTORS)
    numeric_cols.extend(col for cols in DUMMY_FAMILIES.values() for col in cols if col in df.columns)
    numeric_cols.extend(col for col in FOURIER_COLS if col in df.columns)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.dropna(subset=[YEAR_COL, WEEK_COL, target_col, STATE_COL, DIVISION_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[WEEK_COL] = df[WEEK_COL].astype(int)
    df[STATE_COL] = df[STATE_COL].astype(str)
    df[DIVISION_COL] = df[DIVISION_COL].astype(str)
    for col in dataset_group_by:
        df[col] = df[col].astype(str)

    df = aggregate_input_dataset(df, target_col=target_col, dataset_group_by=dataset_group_by)
    return df.sort_values([YEAR_COL, WEEK_COL, STATE_COL, DIVISION_COL]).reset_index(drop=True)


def load_future_data(
    input_source: Union[str, Path, pd.DataFrame],
    target_col: str = TARGET_COL,
    dataset_group_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load future scoring data.

    Expected schema:
        - required identifiers: ``ISO_YEAR``, ``ISO_WEEK``, ``STATE_CD``
        - required raw media columns: the six base media variables
        - optional structural columns: ``Division`` and any requested ``dataset_group_by`` fields
        - optional seasonal columns: ``W_*``, ``F_*``, ``sin_1``, ``cos_1``, ``sin_2``, ``cos_2``

    The target column is optional for future rows and will be treated as missing if absent.
    """
    df = load_tabular_input(input_source, "Future scoring input")
    dataset_group_by = list(dataset_group_by or [])
    df = add_derived_seasonal_columns(df)

    required_cols = {
        YEAR_COL,
        WEEK_COL,
        STATE_COL,
        *NON_DUMMY_PREDICTORS,
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required future-data columns: {missing}")

    if target_col not in df.columns:
        df[target_col] = np.nan
    if DIVISION_COL not in df.columns:
        df[DIVISION_COL] = np.nan
    for col in dataset_group_by:
        if col not in df.columns:
            df[col] = np.nan

    df = df.copy()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[WEEK_COL] = pd.to_numeric(df[WEEK_COL], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    numeric_cols = list(NON_DUMMY_PREDICTORS)
    numeric_cols.extend(col for cols in DUMMY_FAMILIES.values() for col in cols if col in df.columns)
    numeric_cols.extend(col for col in FOURIER_COLS if col in df.columns)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.dropna(subset=[YEAR_COL, WEEK_COL, STATE_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[WEEK_COL] = df[WEEK_COL].astype(int)
    df[STATE_COL] = df[STATE_COL].astype(str)
    df[DIVISION_COL] = df[DIVISION_COL].astype(str)
    for col in dataset_group_by:
        df[col] = df[col].astype(str)
    df = aggregate_input_dataset(df, target_col=target_col, dataset_group_by=dataset_group_by)
    return df.sort_values([YEAR_COL, WEEK_COL, STATE_COL, DIVISION_COL]).reset_index(drop=True)


def validate_optional_features(
    optional_features: Optional[Sequence[str]],
    target_col: str = TARGET_COL,
) -> List[str]:
    """Validate requested engineered variables.

    Available optional variables:
        - ``time_index``
        - ``time_index_sq``
        - ``year_indicator_2025``
        - ``year_indicator_2026``
        - ``Prescreen_lag1``
        - ``DSP_lag1``
        - ``Paid_Search_lag1``
        - ``DSP_trailing_4w_avg``
        - ``Paid_Search_trailing_4w_avg``
        - ``Prescreen_trailing_4w_avg``
        - ``<target_col>_lag1`` for the selected target column
        - ``<target_col>_trailing_4w_avg`` for the selected target column
    """
    if optional_features is None:
        return []

    allowed_dynamic = {
        target_lag_col(target_col),
        target_trailing_avg_col(target_col),
    }
    requested = []
    for feature in optional_features:
        if feature not in OPTIONAL_FEATURES and feature not in allowed_dynamic:
            raise ValueError(
                f"Unsupported optional feature '{feature}'. Available options: {sorted(OPTIONAL_FEATURES)} "
                f"plus {sorted(allowed_dynamic)}"
            )
        requested.append(feature)
    return requested


def validate_media_predictors(media_predictors: Optional[Sequence[str]]) -> List[str]:
    """Validate the selected base media predictors.

    If ``media_predictors`` is omitted, the script uses all six default media variables:
        - ``DSP``
        - ``LeadGen``
        - ``Paid Search``
        - ``Paid Social``
        - ``Prescreen``
        - ``Referrals``
    """
    if media_predictors is None:
        return list(DEFAULT_MEDIA_PREDICTORS)

    requested = []
    for predictor in media_predictors:
        if predictor not in NON_DUMMY_PREDICTORS:
            raise ValueError(
                f"Unsupported media predictor '{predictor}'. Available options: {NON_DUMMY_PREDICTORS}"
            )
        requested.append(predictor)

    if not requested:
        raise ValueError("At least one media predictor must be selected.")

    return requested


def validate_media_transform_config(
    media_transform_config: Optional[Dict[str, Dict[str, Any]]],
    media_predictors: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Validate optional media transform settings.

    Design rules:
        - If a selected media variable is not listed here, it is used raw.
        - If a selected media variable is listed here, only the transformed variant is used.
        - Raw and transformed forms of the same variable are not both included in the model.

    Supported per-variable keys:
        - ``alpha``: adstock carryover value between 0 and 1
        - ``saturation``: ``none`` or ``log1p``
    """
    if media_transform_config is None:
        return {}

    validated: Dict[str, Dict[str, Any]] = {}
    for media_name, raw_config in media_transform_config.items():
        if media_name not in media_predictors:
            raise ValueError(
                f"Transform config provided for '{media_name}', but it is not in selected media_predictors."
            )
        if not isinstance(raw_config, dict):
            raise ValueError(f"Transform config for '{media_name}' must be a dictionary.")

        alpha = raw_config.get("alpha", 0.0)
        saturation = str(raw_config.get("saturation", "none")).lower()

        if alpha is None:
            alpha = 0.0
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha for '{media_name}' must be between 0 and 1.")
        if saturation not in SATURATION_METHODS:
            raise ValueError(
                f"Unsupported saturation method '{saturation}' for '{media_name}'. "
                f"Choices: {sorted(SATURATION_METHODS)}"
            )

        if alpha == 0.0 and saturation == "none":
            continue

        validated[media_name] = {
            "alpha": alpha,
            "saturation": saturation,
        }

    return validated


def validate_target_plot_columns(target_columns: Optional[Sequence[str]]) -> List[str]:
    """Validate user-selected target series for spend-history plots."""
    if target_columns is None:
        return list(PLOTTABLE_TARGET_COLUMNS)

    requested = []
    for col in target_columns:
        if col not in PLOTTABLE_TARGET_COLUMNS:
            raise ValueError(
                f"Unsupported plot target '{col}'. Available options: {PLOTTABLE_TARGET_COLUMNS}"
            )
        requested.append(col)
    return requested


def parse_methodology_selection(methodologies: Optional[Sequence[str]]) -> Tuple[Set[str], Set[str]]:
    """Parse a user-friendly methodology selection list.

    Supported values:
        - ``OLS`` and ``NNLS`` filter model types
        - ``weekly``, ``f_dummy``, and ``fourier`` filter seasonal feature runs
        - ``Fourier`` is accepted as a friendly alias for the ``fourier`` feature run

    If no model type is supplied, both ``OLS`` and ``NNLS`` are used.
    If no feature run is supplied, all feature runs are used.
    """
    if methodologies is None:
        return {"OLS", "NNLS"}, set(FEATURE_RUNS.keys())

    model_types: Set[str] = set()
    feature_runs: Set[str] = set()
    aliases = {"fourier": "fourier", "weekly": "weekly", "f_dummy": "f_dummy"}

    for item in methodologies:
        token = str(item).strip()
        upper_token = token.upper()
        lower_token = token.lower()
        if upper_token in {"OLS", "NNLS"}:
            model_types.add(upper_token)
        elif lower_token in aliases:
            feature_runs.add(aliases[lower_token])
        else:
            raise ValueError(
                "Unsupported methodology selection "
                f"'{item}'. Use OLS, NNLS, weekly, f_dummy, or Fourier."
            )

    if not model_types:
        model_types = {"OLS", "NNLS"}
    if not feature_runs:
        feature_runs = set(FEATURE_RUNS.keys())

    return model_types, feature_runs


def prepare_entity_subset(
    df: pd.DataFrame,
    scope_col: str,
    selected_entities: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Filter to a user-specified state or division subset when requested."""
    if not selected_entities:
        return df

    selected_lookup = {str(value) for value in selected_entities}
    return df[df[scope_col].astype(str).isin(selected_lookup)].copy()


def select_run_columns(df: pd.DataFrame, run_name: str) -> Tuple[List[str], Optional[str]]:
    run_config = FEATURE_RUNS[run_name]
    candidates = [col for col in run_config["extra_cols"] if col in df.columns]
    available = [col for col in candidates if df[col].fillna(0.0).abs().sum() > 0]

    if not available:
        return [], None

    if run_config["drop_one"]:
        dropped_col = available[0]
        selected = [col for col in available if col != dropped_col]
        return selected, dropped_col

    return available, None


def apply_recursive_adstock(series: pd.Series, alpha: float) -> pd.Series:
    """Apply simple recursive adstock to a media series."""
    values = series.fillna(0.0).astype(float).to_numpy()
    out = np.zeros(len(values), dtype=float)
    carry = 0.0
    for idx, value in enumerate(values):
        carry = value + alpha * carry
        out[idx] = carry
    return pd.Series(out, index=series.index, dtype=float)


def apply_saturation(series: pd.Series, saturation: str) -> pd.Series:
    """Apply the requested saturation transform."""
    values = series.fillna(0.0).astype(float)
    if saturation == "none":
        return values
    if saturation == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    raise ValueError(f"Unsupported saturation method '{saturation}'.")


def transform_media_variables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    media_predictors: Sequence[str],
    media_transform_config: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    """Create media design columns, using raw or transformed variants channel by channel.

    Any selected media variable not listed in ``media_transform_config`` is used raw.
    Any selected media variable listed there is replaced by its transformed counterpart.
    """
    combined = pd.concat(
        [
            train_df.assign(__dataset="train"),
            test_df.assign(__dataset="test"),
        ],
        axis=0,
        ignore_index=True,
    )
    combined = combined.sort_values([YEAR_COL, WEEK_COL, "__dataset"]).reset_index(drop=True)

    selected_feature_names: List[str] = []
    feature_name_map: Dict[str, str] = {}

    for media_name in media_predictors:
        if media_name not in media_transform_config:
            selected_feature_names.append(media_name)
            feature_name_map[media_name] = media_name
            continue

        config = media_transform_config[media_name]
        alpha = float(config.get("alpha", 0.0))
        saturation = str(config.get("saturation", "none")).lower()

        transformed = combined[media_name].astype(float)
        feature_suffixes: List[str] = []
        if alpha > 0.0:
            transformed = apply_recursive_adstock(transformed, alpha)
            feature_suffixes.append(f"adstock_{alpha:g}")
        if saturation != "none":
            transformed = apply_saturation(transformed, saturation)
            feature_suffixes.append(saturation)

        transformed_name = f"{safe_name(media_name)}__{'__'.join(feature_suffixes)}"
        combined[transformed_name] = transformed.astype(float)
        selected_feature_names.append(transformed_name)
        feature_name_map[media_name] = transformed_name

    train_out = combined[combined["__dataset"] == "train"].drop(columns=["__dataset"]).copy()
    test_out = combined[combined["__dataset"] == "test"].drop(columns=["__dataset"]).copy()
    return train_out, test_out, selected_feature_names, feature_name_map


def split_train_test(entity_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = entity_df[entity_df[YEAR_COL].isin(TRAIN_YEARS)].copy()
    test_df = entity_df[
        (entity_df[YEAR_COL] == TEST_YEAR) & (entity_df[WEEK_COL].isin(TEST_WEEKS))
    ].copy()
    return train_df, test_df


def generate_backtest_splits(
    entity_df: pd.DataFrame,
    backtest_mode: str,
    fixed_window_weeks: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build train/test splits for the requested backtest mode.

    Modes:
        - ``fixed_holdout``: one model fit on 2024-2025, scored on 2026 weeks 1-8.
        - ``rolling_one_step_expanding``: refit each week using all prior observed rows.
        - ``rolling_one_step_fixed_window``: refit each week using only the most recent
          ``fixed_window_weeks`` prior rows.
    """
    if backtest_mode not in BACKTEST_MODES:
        raise ValueError(f"Unsupported backtest mode '{backtest_mode}'.")

    entity_df = entity_df.sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)
    if backtest_mode == "fixed_holdout":
        train_df, test_df = split_train_test(entity_df)
        return [(train_df, test_df)] if (not train_df.empty and not test_df.empty) else []

    candidate_idx = entity_df.index[
        (entity_df[YEAR_COL] == TEST_YEAR) & (entity_df[WEEK_COL].isin(TEST_WEEKS))
    ].tolist()
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for idx in candidate_idx:
        train_df = entity_df.iloc[:idx].copy()
        if backtest_mode == "rolling_one_step_fixed_window" and fixed_window_weeks > 0:
            train_df = train_df.tail(fixed_window_weeks).copy()
        test_df = entity_df.iloc[[idx]].copy()
        if not train_df.empty and not test_df.empty:
            splits.append((train_df, test_df))
    return splits


def aggregate_division_weekly(
    entity_df: pd.DataFrame,
    run_name: str,
    target_col: str = TARGET_COL,
    dataset_group_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate state rows to one row per division-week for division fallback models.

    Additive series are summed across component states, while seasonal columns are carried
    forward once per division-week because they describe the week itself rather than volume.
    All raw media columns are retained in the aggregated frame so optional lag and rolling
    features can still be engineered even when the user later models only a subset.
    """
    dataset_group_by = list(dataset_group_by or [])
    run_cols = [col for col in FEATURE_RUNS[run_name]["extra_cols"] if col in entity_df.columns]
    group_cols = [DIVISION_COL, YEAR_COL, WEEK_COL, *dataset_group_by]
    additive_media_cols = list(NON_DUMMY_PREDICTORS)
    aggregation_map: Dict[str, Any] = {
        target_col: lambda s: s.sum(min_count=1),
    }
    aggregation_map.update({col: "sum" for col in additive_media_cols})
    aggregation_map.update({col: "first" for col in run_cols})

    aggregated = (
        entity_df[group_cols + [target_col] + additive_media_cols + run_cols]
        .groupby(group_cols, as_index=False, dropna=False)
        .agg(aggregation_map)
    )

    return aggregated.sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)


def add_optional_features(
    entity_df: pd.DataFrame,
    optional_features: Sequence[str],
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Create engineered variables that users can optionally include in the model.

    Available optional variables:
        - ``time_index``
        - ``time_index_sq``
        - ``year_indicator_2025``
        - ``year_indicator_2026``
        - ``Prescreen_lag1``
        - ``DSP_lag1``
        - ``Paid_Search_lag1``
        - ``DSP_trailing_4w_avg``
        - ``Paid_Search_trailing_4w_avg``
        - ``Prescreen_trailing_4w_avg``
        - ``<target_col>_lag1`` for the selected target column
        - ``<target_col>_trailing_4w_avg`` for the selected target column

    The function expects data for a single modeling series, meaning either one state over
    time or one aggregated division over time. Lagged terms are filled with ``0.0`` for the
    earliest week so the design matrix stays dense. Target trailing averages use only prior
    observed weeks to avoid leaking the current target into the predictor set.
    """
    if not optional_features:
        return entity_df

    frame = entity_df.sort_values([YEAR_COL, WEEK_COL]).copy()
    time_index = np.arange(1, len(frame) + 1, dtype=float)
    lagged_target = frame[target_col].shift(1)

    if TIME_INDEX_COL in optional_features:
        frame[TIME_INDEX_COL] = time_index
    if TIME_INDEX_SQ_COL in optional_features:
        frame[TIME_INDEX_SQ_COL] = np.square(time_index)
    if YEAR_INDICATOR_2025_COL in optional_features:
        frame[YEAR_INDICATOR_2025_COL] = (frame[YEAR_COL] == 2025).astype(float)
    if YEAR_INDICATOR_2026_COL in optional_features:
        frame[YEAR_INDICATOR_2026_COL] = (frame[YEAR_COL] == 2026).astype(float)
    if PRESCREEN_LAG1_COL in optional_features:
        frame[PRESCREEN_LAG1_COL] = frame["Prescreen"].shift(1).fillna(0.0)
    if DSP_LAG1_COL in optional_features:
        frame[DSP_LAG1_COL] = frame["DSP"].shift(1).fillna(0.0)
    if PAID_SEARCH_LAG1_COL in optional_features:
        frame[PAID_SEARCH_LAG1_COL] = frame["Paid Search"].shift(1).fillna(0.0)
    if DSP_TRAILING_4W_AVG_COL in optional_features:
        frame[DSP_TRAILING_4W_AVG_COL] = frame["DSP"].rolling(window=4, min_periods=1).mean()
    if PAID_SEARCH_TRAILING_4W_AVG_COL in optional_features:
        frame[PAID_SEARCH_TRAILING_4W_AVG_COL] = frame["Paid Search"].rolling(window=4, min_periods=1).mean()
    if PRESCREEN_TRAILING_4W_AVG_COL in optional_features:
        frame[PRESCREEN_TRAILING_4W_AVG_COL] = frame["Prescreen"].rolling(window=4, min_periods=1).mean()
    dynamic_target_lag = target_lag_col(target_col)
    dynamic_target_trailing = target_trailing_avg_col(target_col)
    if dynamic_target_lag in optional_features:
        frame[dynamic_target_lag] = lagged_target.fillna(0.0)
    if dynamic_target_trailing in optional_features:
        frame[dynamic_target_trailing] = lagged_target.rolling(window=4, min_periods=1).mean().fillna(0.0)

    return frame


def build_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str,
    media_predictors: Sequence[str],
    media_transform_config: Dict[str, Dict[str, Any]],
    optional_features: Sequence[str],
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, object, List[str], Optional[str], Dict[str, str]]:
    """Build aligned train and test design matrices for one model run.

    Default media predictors:
        - DSP, LeadGen, Paid Search, Paid Social, Prescreen, Referrals

    Users can provide any subset of those six media variables via ``media_predictors``.
    Selected media variables can also be transformed through ``media_transform_config``.
    A transformed media variable replaces its raw version in the design matrix.

    Seasonal feature runs:
        - ``weekly``: uses ``W_*`` dummies with one dummy dropped
        - ``f_dummy``: uses ``F_*`` dummies with one dummy dropped
        - ``fourier``: uses ``sin_1``, ``cos_1``, ``sin_2``, ``cos_2`` and excludes dummy families

    Optional engineered variables that can be added on top of the defaults:
        - ``time_index``
        - ``time_index_sq``
        - ``year_indicator_2025``
        - ``year_indicator_2026``
        - ``Prescreen_lag1``
        - ``DSP_lag1``
        - ``Paid_Search_lag1``
        - ``DSP_trailing_4w_avg``
        - ``Paid_Search_trailing_4w_avg``
        - ``Prescreen_trailing_4w_avg``
        - ``APPLICATIONS_lag1`` or ``NON_DM_APPLICATIONS_lag1`` depending on the active target
        - ``APPLICATIONS_trailing_4w_avg`` or ``NON_DM_APPLICATIONS_trailing_4w_avg`` depending on the active target
    """
    train_df, test_df, media_feature_cols, media_feature_map = transform_media_variables(
        train_df=train_df,
        test_df=test_df,
        media_predictors=media_predictors,
        media_transform_config=media_transform_config,
    )
    run_cols, dropped_col = select_run_columns(pd.concat([train_df, test_df], axis=0), run_name)
    feature_cols = list(media_feature_cols) + list(optional_features) + run_cols
    feature_cols = [col for col in feature_cols if train_df[col].nunique(dropna=False) > 1]

    x_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(float).copy()
    x_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(float).copy()

    x_train = x_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler_type = FEATURE_RUNS[run_name]["scaler"]
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    scale_cols = list(media_feature_cols) + list(optional_features)
    if run_name == "fourier":
        scale_cols.extend(run_cols)
    scale_cols = list(dict.fromkeys(scale_cols))
    scale_cols = [col for col in scale_cols if col in x_train.columns]

    if scale_cols:
        x_train.loc[:, scale_cols] = scaler.fit_transform(x_train[scale_cols])
        x_test.loc[:, scale_cols] = scaler.transform(x_test[scale_cols])

    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return x_train, y_train, x_test, y_test, scaler, feature_cols, dropped_col, media_feature_map


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.clip(np.abs(y_true_arr), EPSILON, None))) * 100.0)

    metrics = {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
    }

    if len(y_true_arr) >= 2 and not np.allclose(y_true_arr, y_true_arr[0]):
        metrics["R2_test"] = r2_score(y_true_arr, y_pred_arr)
    else:
        metrics["R2_test"] = np.nan

    return metrics


def adjusted_r2(r2_value: float, n_obs: int, n_predictors: int) -> float:
    if n_obs <= n_predictors + 1:
        return np.nan
    return 1.0 - (1.0 - r2_value) * ((n_obs - 1) / (n_obs - n_predictors - 1))


def compute_information_criteria(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_params: int,
) -> Tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n_obs = len(y_true_arr)

    if n_obs == 0:
        return np.nan, np.nan

    rss = float(np.sum(np.square(y_true_arr - y_pred_arr)))
    rss = max(rss, EPSILON)
    aic = n_obs * math.log(rss / n_obs) + 2 * n_params
    bic = n_obs * math.log(rss / n_obs) + math.log(n_obs) * n_params
    return aic, bic


def fit_ols(x_train: pd.DataFrame, y_train: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    x_train_with_const = sm.add_constant(x_train, has_constant="add")
    return sm.OLS(y_train, x_train_with_const, missing="drop").fit()


def fit_nnls(x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    x_values = x_train.to_numpy(dtype=float)
    y_values = y_train.to_numpy(dtype=float)
    coef, residual_norm = nnls(x_values, y_values)

    train_pred = x_values @ coef
    train_r2 = r2_score(y_values, train_pred) if len(y_values) >= 2 else np.nan
    n_obs = len(y_values)
    n_predictors = x_values.shape[1]
    aic, bic = compute_information_criteria(y_values, train_pred, n_predictors)

    return {
        "coef": coef,
        "intercept": 0.0,
        "residual_norm": float(residual_norm),
        "train_pred": train_pred,
        "train_r2": train_r2,
        "train_adj_r2": adjusted_r2(train_r2, n_obs, n_predictors),
        "aic": aic,
        "bic": bic,
        "feature_names": list(x_train.columns),
        "n_obs": n_obs,
        "n_predictors": n_predictors,
    }


def predict_nnls(nnls_result: Dict[str, object], x_frame: pd.DataFrame) -> np.ndarray:
    feature_names = nnls_result["feature_names"]
    coef = np.asarray(nnls_result["coef"], dtype=float)
    x_aligned = x_frame.reindex(columns=feature_names, fill_value=0.0)
    return x_aligned.to_numpy(dtype=float) @ coef


def model_diagnostics_row(
    metadata: RunMetadata,
    train_y: pd.Series,
    train_pred: Sequence[float],
    test_y: pd.Series,
    test_pred: Sequence[float],
    train_r2: float,
    train_adj_r2: float,
    aic: float,
    bic: float,
    spend_coefficients: str,
) -> Dict[str, object]:
    test_metrics = regression_metrics(test_y, test_pred)
    average_bias = float(np.mean(np.asarray(test_pred, dtype=float) - np.asarray(test_y, dtype=float)))
    row = asdict(metadata)
    row.update(
        {
            "n_observations": metadata.train_rows,
            "n_test_observations": metadata.test_rows,
            "R2": train_r2,
            "AdjR2": train_adj_r2,
            "MAE": test_metrics["MAE"],
            "MAPE": test_metrics["MAPE"],
            "RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2_test"],
            "Average_Bias": average_bias,
            "AIC": aic,
            "BIC": bic,
            "Spend_Coefficients": spend_coefficients,
        }
    )
    return row


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def entity_artifact_dir(output_root: Path, metadata: RunMetadata) -> Path:
    """Original per-entity artifact layout."""
    return output_root / metadata.dummy_family / metadata.scope / safe_name(metadata.entity) / metadata.model_type


def grouped_artifact_path(output_root: Path, metadata: RunMetadata, artifact_name: str) -> Path:
    """Grouped artifact layout for cross-entity comparison by artifact type.

    Files are stored directly in the artifact folder, with the entity name embedded in the
    filename so users do not need to click through one subfolder per state or division.
    """
    return (
        output_root
        / "by_artifact"
        / metadata.dummy_family
        / metadata.scope
        / metadata.model_type
        / artifact_name
    )


def scaler_metadata(scaler: object, scaler_type: str) -> Dict[str, object]:
    if not hasattr(scaler, "feature_names_in_"):
        return {
            "scaler_type": scaler_type,
            "scaled_columns": [],
        }

    feature_names = scaler.feature_names_in_.tolist()
    if scaler_type == "minmax":
        return {
            "scaler_type": scaler_type,
            "scaled_columns": feature_names,
            "scaler_data_min": dict(zip(feature_names, scaler.data_min_.tolist())),
            "scaler_data_max": dict(zip(feature_names, scaler.data_max_.tolist())),
        }

    return {
        "scaler_type": scaler_type,
        "scaled_columns": feature_names,
        "scaler_mean": dict(zip(feature_names, scaler.mean_.tolist())),
        "scaler_scale": dict(zip(feature_names, scaler.scale_.tolist())),
    }


def minmax_scaler_lookup(scaler: object) -> Dict[str, Dict[str, float]]:
    """Return per-variable min/range metadata for MinMax-scaled predictors."""
    if not isinstance(scaler, MinMaxScaler) or not hasattr(scaler, "feature_names_in_"):
        return {}
    return {
        str(name): {
            "min": float(min_value),
            "range": float(range_value),
        }
        for name, min_value, range_value in zip(
            scaler.feature_names_in_.tolist(),
            scaler.data_min_.tolist(),
            scaler.data_range_.tolist(),
        )
    }


def training_summary_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    r2_value: float,
    adj_r2_value: float,
) -> Dict[str, float]:
    """Compute training diagnostics for a fitted model."""
    base_metrics = regression_metrics(y_true, y_pred)
    return {
        "R2": float(r2_value),
        "AdjR2": float(adj_r2_value),
        "MAE": float(base_metrics["MAE"]),
        "MAPE": float(base_metrics["MAPE"]),
        "RMSE": float(base_metrics["RMSE"]),
    }


def iso_week_month(iso_year: int, iso_week: int) -> int:
    """Return the calendar month for the Monday of an ISO year/week."""
    return int(date.fromisocalendar(int(iso_year), int(iso_week), 1).month)


def format_predicted_apps(value: float) -> int:
    """Round forecasts up to whole applications and floor negative values at zero."""
    if not np.isfinite(value):
        return 0
    return int(max(0, math.ceil(float(value))))


def coefficient_lookup(model_type: str, fitted_model: object) -> Dict[str, float]:
    """Return a term-to-coefficient mapping for the fitted model."""
    if model_type == "OLS":
        return {str(term): float(value) for term, value in fitted_model.params.items() if term != "const"}
    return {
        str(term): float(value)
        for term, value in zip(
            fitted_model["feature_names"],
            np.asarray(fitted_model["coef"], dtype=float).tolist(),
        )
    }


def roll_up_weekly_forecast_to_monthly(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Roll weekly forecast rows into calendar months using equal daily allocation.

    Each ISO week forecast is split across its seven calendar dates. If a week spans two
    months, the forecast is allocated proportionally by the number of days in each month.

    The function prefers ``Predicted APPS Raw`` when available so monthly rollups are based
    on the unrounded weekly forecast. If that column is absent, it falls back to
    ``Predicted APPS``.
    """
    if forecast_df.empty:
        return pd.DataFrame()

    if "Predicted APPS Raw" in forecast_df.columns:
        prediction_col = "Predicted APPS Raw"
    elif "Predicted APPS" in forecast_df.columns:
        prediction_col = "Predicted APPS"
    else:
        raise ValueError("forecast_df must include 'Predicted APPS Raw' or 'Predicted APPS'.")

    success_df = forecast_df[forecast_df.get("Run_Status", "").astype(str) == "SUCCESS"].copy()
    if success_df.empty:
        return pd.DataFrame()

    prorated_metric_cols = [
        col for col in success_df.columns
        if col.startswith("APPLICATIONS_")
        or col.startswith("APPROVAL_")
        or col.startswith("ORIGINATIONS_")
    ]

    allocation_rows: List[Dict[str, Any]] = []
    for _, row in success_df.iterrows():
        iso_year = int(row["ISO_Year"])
        iso_week = int(row["ISO_Week"])
        weekly_prediction = float(row[prediction_col])
        daily_prediction = weekly_prediction / 7.0

        month_day_counts: Dict[Tuple[int, int], int] = {}
        for iso_day in range(1, 8):
            calendar_day = date.fromisocalendar(iso_year, iso_week, iso_day)
            month_key = (calendar_day.year, calendar_day.month)
            month_day_counts[month_key] = month_day_counts.get(month_key, 0) + 1

        for (calendar_year, calendar_month), day_count in month_day_counts.items():
            prorate_factor = day_count / 7.0
            allocation_row: Dict[str, Any] = {
                "Key": row.get("Key"),
                "Run_Status": row.get("Run_Status"),
                "State": row.get("State"),
                "Calendar_Year": int(calendar_year),
                "Calendar_Month": int(calendar_month),
                "Allocated_Days": int(day_count),
                "Allocated_Predicted_APPS": daily_prediction * day_count,
                "Scope": row.get("Scope"),
                "Model_Type": row.get("Model_Type"),
                "Feature_Run": row.get("Feature_Run"),
                "Channel": row.get("Channel"),
                "H_Tactic": row.get("H_Tactic"),
                "Detail_Tactic": row.get("Detail_Tactic"),
                "Product": row.get("Product"),
            }
            for metric_col in prorated_metric_cols:
                metric_value = pd.to_numeric(pd.Series([row.get(metric_col)]), errors="coerce").iloc[0]
                allocation_row[metric_col] = float(metric_value) * prorate_factor if pd.notna(metric_value) else 0.0
            # Carry marginal columns through unchanged — they are rates
            # not counts, so they must NOT be multiplied by prorate_factor
            MARGINAL_COLS = [f"Marginal_{safe_name(t)}" for t in NON_DUMMY_PREDICTORS]
            for _mc in MARGINAL_COLS:
                if _mc in row.index and pd.notna(row[_mc]):
                    allocation_row[_mc] = float(row[_mc])

            allocation_rows.append(
                allocation_row
            )

    monthly_df = pd.DataFrame(allocation_rows)
    if monthly_df.empty:
        return monthly_df

    group_cols = [
        "Key",
        "Run_Status",
        "State",
        "Calendar_Year",
        "Calendar_Month",
        "Scope",
        "Model_Type",
        "Feature_Run",
        "Channel",
        "H_Tactic",
        "Detail_Tactic",
        "Product",
    ]
    aggregation_map = {
        "Allocated_Days": "sum",
        "Allocated_Predicted_APPS": "sum",
    }
    aggregation_map.update({col: "sum" for col in prorated_metric_cols})
    # Marginal columns: use mean across weeks (they are rates, not counts)
    _marginal_cols_present = [
        c for c in [
            "Marginal_DSP", "Marginal_LeadGen", "Marginal_Paid_Search",
            "Marginal_Paid_Social", "Marginal_Prescreen", "Marginal_Referrals",
        ]
        if c in monthly_df.columns
    ]
    aggregation_map.update({col: "mean" for col in _marginal_cols_present})
    monthly_df = (
        monthly_df.groupby(group_cols, as_index=False, dropna=False)
        .agg(aggregation_map)
        .sort_values(["Key", "Calendar_Year", "Calendar_Month", "Feature_Run", "Model_Type"])
        .reset_index(drop=True)
    )
    monthly_df["Allocated_Predicted_APPS_Rounded"] = monthly_df["Allocated_Predicted_APPS"].apply(format_predicted_apps)
    application_cols = [col for col in prorated_metric_cols if col.startswith("APPLICATIONS_")]
    approval_cols = [col for col in prorated_metric_cols if col.startswith("APPROVAL_")]
    origination_cols = [col for col in prorated_metric_cols if col.startswith("ORIGINATIONS_")]
    if approval_cols:
        monthly_df["Allocated_Approved"] = monthly_df[approval_cols].sum(axis=1)
        monthly_df["Allocated_Approved_Rounded"] = monthly_df["Allocated_Approved"].apply(format_predicted_apps)
    if origination_cols:
        monthly_df["Allocated_Originations"] = monthly_df[origination_cols].sum(axis=1)
        monthly_df["Allocated_Originations_Rounded"] = monthly_df["Allocated_Originations"].apply(format_predicted_apps)
    if application_cols:
        monthly_df["Allocated_Product_Applications"] = monthly_df[application_cols].sum(axis=1)
    return monthly_df


def future_output_value(row: pd.Series, column_name: str) -> Optional[object]:
    """Safely fetch an output attribute from a future row."""
    if column_name not in row.index:
        return None
    value = row[column_name]
    if pd.isna(value):
        return None
    return value


def column_has_meaningful_values(frame: pd.DataFrame, column_name: str) -> bool:
    """Return True when a column contains at least one non-blank value."""
    if column_name not in frame.columns:
        return False
    values = frame[column_name]
    if values.isna().all():
        return False
    tokens = values.astype(str).str.strip()
    return (~tokens.isin({"", "nan", "None", "<NA>"})).any()


def expand_future_rows_to_model_grain(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    grouping_keys: Sequence[str],
) -> pd.DataFrame:
    """Expand future rows so every historical model slice receives the same future spend.

    This is meant for cases where ``dataset_group_by`` columns define separate models from
    history, but those fields are not predictors and therefore are not present in the future
    spend file. The future state-week rows are duplicated across all historical model keys
    that match on the available identifying columns, typically the scope column.
    """
    if future_df.empty:
        return future_df.copy()

    history_keys = history_df[list(dict.fromkeys(grouping_keys))].drop_duplicates().copy()
    if history_keys.empty:
        return future_df.copy()

    future_base = future_df.copy()
    merge_cols = [col for col in grouping_keys if column_has_meaningful_values(future_base, col)]
    if not merge_cols and grouping_keys:
        merge_cols = [grouping_keys[0]]

    for col in grouping_keys:
        if col not in future_base.columns:
            future_base[col] = np.nan

    expanded = future_base.merge(history_keys, on=merge_cols, how="left", suffixes=("", "__history"))
    for col in grouping_keys:
        history_col = f"{col}__history"
        if history_col in expanded.columns:
            expanded[col] = expanded[history_col].where(
                expanded[history_col].notna()
                & ~expanded[history_col].astype(str).str.strip().isin({"", "nan", "None", "<NA>"}),
                expanded[col],
            )
            expanded = expanded.drop(columns=[history_col])

    expanded = expanded.drop_duplicates().reset_index(drop=True)
    return expanded


def build_future_coefficients_frame(
    metadata: RunMetadata,
    model_type: str,
    fitted_model: object,
    scaler: object,
) -> pd.DataFrame:
    """Create the coefficient export for one fitted future-forecast model.

    The output is one row per model, with coefficient terms pivoted wide into columns.
    Scaler parameters (MinMax min/range or Standard mean/scale) are embedded as companion
    columns so the row is self-contained for downstream scoring without re-fitting.
    Additional metadata columns ``Scaler_Type``, ``Dropped_Dummy``, and
    ``Media_Transform_Config`` (JSON string) are included so a scoring function can
    fully reconstruct the feature pipeline from this row alone.
    """
    minmax_lookup = minmax_scaler_lookup(scaler)

    # Build StandardScaler lookup for Fourier/standard runs.
    std_lookup: Dict[str, Dict[str, float]] = {}
    if isinstance(scaler, StandardScaler) and hasattr(scaler, "feature_names_in_"):
        std_lookup = {
            str(name): {"mean": float(mean_val), "scale": float(scale_val)}
            for name, mean_val, scale_val in zip(
                scaler.feature_names_in_.tolist(),
                scaler.mean_.tolist(),
                scaler.scale_.tolist(),
            )
        }

    if model_type == "OLS":
        intercept = float(fitted_model.params.get("const", 0.0))
        coef_items = [(term, float(value)) for term, value in fitted_model.params.items() if term != "const"]
    else:
        intercept = float(fitted_model["intercept"])
        coef_items = list(
            zip(
                fitted_model["feature_names"],
                np.asarray(fitted_model["coef"], dtype=float).tolist(),
            )
        )

    _key_parts = _parse_key_string(metadata.entity)
    row: Dict[str, Any] = {
        "Key":           metadata.entity,
        "State":         _key_parts.get("STATE_CD",      None),
        "Channel":       _key_parts.get("CHANNEL_CD",    "--Default--"),
        "H_Tactic":      _key_parts.get("H_TACTIC",      "--Default--"),
        "Detail_Tactic": _key_parts.get("DETAIL_TACTIC", "--Default--"),
        "Scope":         metadata.scope,
        "Model_Type":    metadata.model_type,
        "Feature_Run":   metadata.dummy_family,
        "Scaler_Type":   metadata.scaler_type,
        "Dropped_Dummy": metadata.dropped_dummy,
        "Media_Transform_Config": json.dumps(metadata.media_transform_config, default=str),
        "Target":        metadata.target_col,
        "Intercept":     intercept,
    }
    for term, coefficient in coef_items:
        row[term] = float(coefficient)
        minmax_info = minmax_lookup.get(term, {})
        if minmax_info:
            row[f"{term}__MinMax_Min"] = minmax_info.get("min")
            row[f"{term}__MinMax_Range"] = minmax_info.get("range")
        std_info = std_lookup.get(term, {})
        if std_info:
            row[f"{term}__Std_Mean"] = std_info.get("mean")
            row[f"{term}__Std_Scale"] = std_info.get("scale")
    return pd.DataFrame([row])


def build_future_training_summary_row(
    metadata: RunMetadata,
    train_y: Sequence[float],
    train_pred: Sequence[float],
    r2_value: float,
    adj_r2_value: float,
) -> Dict[str, Any]:
    """Create the training-summary export row for one fitted future-forecast model."""
    metrics = training_summary_metrics(train_y, train_pred, r2_value, adj_r2_value)
    return {
        "Key": metadata.entity,
        "Scope": metadata.scope,
        "Model_Type": metadata.model_type,
        "Feature_Run": metadata.dummy_family,
        "Target": metadata.target_col,
        **metrics,
    }


def score_from_coefficients_row(
    spend_df: pd.DataFrame,
    coeff_row: pd.Series,
) -> pd.DataFrame:
    """Score a weekly spend DataFrame using one embedded coefficient row.

    Parameters
    ----------
    spend_df : pd.DataFrame
        Weekly spend in wide format — ISO_YEAR, ISO_WEEK, STATE_CD, and raw media
        columns (DSP, LeadGen, Paid Search, Paid Social, Prescreen, Referrals,
        Sweepstakes).  Rows must be sorted by ISO_YEAR, ISO_WEEK so that adstock
        carryover is computed correctly.
    coeff_row : pd.Series
        One row from :func:`build_future_coefficients_frame`.  Must contain the
        ``Scaler_Type``, ``Dropped_Dummy``, and ``Media_Transform_Config`` columns
        added in the extended version, plus the embedded scaler-parameter companions
        (``__MinMax_Min``/``__MinMax_Range`` or ``__Std_Mean``/``__Std_Scale``) for
        every scaled predictor.

    Returns
    -------
    pd.DataFrame
        One row per input spend row with columns:
        ISO_YEAR, ISO_WEEK, STATE_CD, Predicted_APPS_Raw, Predicted_APPS.
    """
    _SCALER_SUFFIXES = ("__MinMax_Min", "__MinMax_Range", "__Std_Mean", "__Std_Scale")
    _META_COLS = {
        "Key", "Scope", "Model_Type", "Feature_Run", "Target", "Intercept",
        "Dropped_Dummy", "Scaler_Type", "Media_Transform_Config",
        "Modeling_Grain", "source_file",
    }

    intercept = float(coeff_row.get("Intercept", 0.0))
    dropped_dummy = coeff_row.get("Dropped_Dummy")
    if pd.isna(dropped_dummy) if not isinstance(dropped_dummy, str) else dropped_dummy in ("", "nan", "None"):
        dropped_dummy = None

    mtc_raw = coeff_row.get("Media_Transform_Config", "{}")
    if pd.isna(mtc_raw) if not isinstance(mtc_raw, str) else False:
        mtc_raw = "{}"
    try:
        media_transform_config: Dict[str, Any] = json.loads(str(mtc_raw))
    except (json.JSONDecodeError, TypeError):
        media_transform_config = {}

    # Build coefficient dict — skip metadata and scaler param columns.
    coef_dict: Dict[str, float] = {}
    for col, val in coeff_row.items():
        if col in _META_COLS:
            continue
        if any(col.endswith(sfx) for sfx in _SCALER_SUFFIXES):
            continue
        if pd.notna(val):
            try:
                coef_dict[col] = float(val)
            except (ValueError, TypeError):
                pass

    if not coef_dict:
        empty = spend_df[[YEAR_COL, WEEK_COL, STATE_COL]].copy()
        empty["Predicted_APPS_Raw"] = 0.0
        empty["Predicted_APPS"] = 0
        return empty

    frame = spend_df.sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True).copy()
    frame = add_derived_seasonal_columns(frame)

    # Apply media transforms (adstock then saturation).
    for media_name, config in media_transform_config.items():
        if media_name not in frame.columns:
            continue
        alpha = float(config.get("alpha", 0.0))
        saturation = str(config.get("saturation", "none")).lower()
        transformed = frame[media_name].astype(float)
        feature_suffixes: List[str] = []
        if alpha > 0.0:
            transformed = apply_recursive_adstock(transformed, alpha)
            feature_suffixes.append(f"adstock_{alpha:g}")
        if saturation != "none":
            transformed = apply_saturation(transformed, saturation)
            feature_suffixes.append(saturation)
        transformed_name = f"{safe_name(media_name)}__{'__'.join(feature_suffixes)}"
        frame[transformed_name] = transformed.astype(float)

    # Reconstruct deterministic optional features when they appear in the model.
    # time_index must match the training-window convention: week 1 of 2024 = 1,
    # so index = (year - 2024) * 52 + week + 1.  Using row position here would
    # give wrong values when scoring future data that doesn't start at week 1 of 2024.
    if TIME_INDEX_COL in coef_dict:
        frame[TIME_INDEX_COL] = ((frame[YEAR_COL] - 2024) * 52 + frame[WEEK_COL] + 1).astype(float)
    if TIME_INDEX_SQ_COL in coef_dict:
        frame[TIME_INDEX_SQ_COL] = np.square(frame[TIME_INDEX_COL] if TIME_INDEX_COL in frame.columns else ((frame[YEAR_COL] - 2024) * 52 + frame[WEEK_COL] + 1).astype(float))
    if YEAR_INDICATOR_2025_COL in coef_dict:
        frame[YEAR_INDICATOR_2025_COL] = (frame[YEAR_COL] == 2025).astype(float)
    if YEAR_INDICATOR_2026_COL in coef_dict:
        frame[YEAR_INDICATOR_2026_COL] = (frame[YEAR_COL] == 2026).astype(float)
    if PRESCREEN_LAG1_COL in coef_dict and "Prescreen" in frame.columns:
        frame[PRESCREEN_LAG1_COL] = frame["Prescreen"].shift(1).fillna(0.0)
    if DSP_LAG1_COL in coef_dict and "DSP" in frame.columns:
        frame[DSP_LAG1_COL] = frame["DSP"].shift(1).fillna(0.0)
    if PAID_SEARCH_LAG1_COL in coef_dict and "Paid Search" in frame.columns:
        frame[PAID_SEARCH_LAG1_COL] = frame["Paid Search"].shift(1).fillna(0.0)
    if DSP_TRAILING_4W_AVG_COL in coef_dict and "DSP" in frame.columns:
        frame[DSP_TRAILING_4W_AVG_COL] = frame["DSP"].rolling(window=4, min_periods=1).mean()
    if PAID_SEARCH_TRAILING_4W_AVG_COL in coef_dict and "Paid Search" in frame.columns:
        frame[PAID_SEARCH_TRAILING_4W_AVG_COL] = frame["Paid Search"].rolling(window=4, min_periods=1).mean()
    if PRESCREEN_TRAILING_4W_AVG_COL in coef_dict and "Prescreen" in frame.columns:
        frame[PRESCREEN_TRAILING_4W_AVG_COL] = frame["Prescreen"].rolling(window=4, min_periods=1).mean()
    # Target-lag features are set to 0 in the scoring context (no observed target history).

    for col in coef_dict:
        if col not in frame.columns:
            frame[col] = 0.0

    feature_cols = list(coef_dict.keys())
    x = frame.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Apply scaling using the embedded per-column parameters.
    for col in feature_cols:
        min_key = f"{col}__MinMax_Min"
        range_key = f"{col}__MinMax_Range"
        mean_key = f"{col}__Std_Mean"
        scale_key = f"{col}__Std_Scale"
        if min_key in coeff_row.index and pd.notna(coeff_row.get(min_key)):
            range_val = float(coeff_row[range_key])
            if range_val > 0:
                x[col] = (x[col] - float(coeff_row[min_key])) / range_val
        elif mean_key in coeff_row.index and pd.notna(coeff_row.get(mean_key)):
            scale_val = float(coeff_row[scale_key])
            if scale_val > 0:
                x[col] = (x[col] - float(coeff_row[mean_key])) / scale_val

    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    coef_array = np.array([coef_dict[col] for col in feature_cols], dtype=float)
    raw_preds = np.maximum(0.0, x.to_numpy() @ coef_array + intercept)

    result = frame[[YEAR_COL, WEEK_COL, STATE_COL]].copy()
    result["Predicted_APPS_Raw"] = raw_preds
    result["Predicted_APPS"] = result["Predicted_APPS_Raw"].apply(format_predicted_apps)
    return result.sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)


def score_spend_with_coefficients(
    spend_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    model_type: str,
    feature_run: str,
) -> pd.DataFrame:
    """Score all states in spend_df using matching coefficient rows.

    Matches each state in spend_df to its coefficient row by parsing ``STATE_CD``
    from the Key column.  States with no matching coefficient row are silently skipped.

    Parameters
    ----------
    spend_df : pd.DataFrame
        Weekly wide spend, as returned by :func:`prepare_future_spend_frame`.
    coeff_df : pd.DataFrame
        Coefficient table as written by :func:`run_future_forecast`, typically loaded
        from ``model_coefficients_consolidated.csv``.
    model_type : str
        ``"OLS"`` or ``"NNLS"``.
    feature_run : str
        One of ``"weekly"``, ``"f_dummy"``, ``"fourier"``.

    Returns
    -------
    pd.DataFrame
        Predictions with columns: Key, ISO_YEAR, ISO_WEEK, STATE_CD,
        Predicted_APPS_Raw, Predicted_APPS, Model_Type, Feature_Run.
    """
    mask = (
        (coeff_df["Model_Type"].astype(str) == model_type)
        & (coeff_df["Feature_Run"].astype(str) == feature_run)
    )
    matching_coeffs = coeff_df[mask].copy()
    matching_coeffs["_state_cd"] = matching_coeffs["Key"].apply(
        lambda k: _parse_key_string(k).get("STATE_CD", "")
    )

    # Aggregate spend to one row per (STATE_CD, ISO_YEAR, ISO_WEEK) so that any
    # tactic-level or channel-level rows in the spend file are collapsed before scoring.
    spend_cols = [c for c in spend_df.columns if c not in [YEAR_COL, WEEK_COL, STATE_COL]]
    spend_agg = (
        spend_df.groupby([STATE_COL, YEAR_COL, WEEK_COL], as_index=False)[spend_cols]
        .sum(numeric_only=True)
    )
    # Carry any non-numeric columns (e.g. Division) forward from the first row.
    non_numeric = [c for c in spend_cols if c not in spend_agg.columns]
    if non_numeric:
        first_vals = spend_df.groupby([STATE_COL, YEAR_COL, WEEK_COL])[non_numeric].first().reset_index()
        spend_agg = spend_agg.merge(first_vals, on=[STATE_COL, YEAR_COL, WEEK_COL], how="left")

    # Iterate over every coefficient row so that all models within the selected
    # grain are scored (e.g. CHANNEL_CD=PHYSICAL and CHANNEL_CD=DIGITAL both
    # produce predictions for the same state).  Cross-grain deduplication is
    # handled upstream by the Modeling Grain selector in the app.
    result_frames: List[pd.DataFrame] = []
    for _, coeff_row in matching_coeffs.iterrows():
        state_cd = coeff_row["_state_cd"]
        if not state_cd:
            continue
        state_spend = spend_agg[spend_agg[STATE_COL] == state_cd]
        if state_spend.empty:
            continue
        preds = score_from_coefficients_row(state_spend.copy(), coeff_row)
        preds["Key"] = coeff_row["Key"]
        preds["Model_Type"] = model_type
        preds["Feature_Run"] = feature_run
        result_frames.append(preds)

    if not result_frames:
        return pd.DataFrame()
    combined = pd.concat(result_frames, axis=0, ignore_index=True)
    lead = ["Key", YEAR_COL, WEEK_COL, STATE_COL, "Predicted_APPS_Raw", "Predicted_APPS", "Model_Type", "Feature_Run"]
    present = [c for c in lead if c in combined.columns]
    return combined[present].sort_values(["Key", YEAR_COL, WEEK_COL]).reset_index(drop=True)



def compute_marginals(
    spend_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    model_type: str = "OLS",
    feature_run: str = "weekly",
    increment: float = 1_000.0,
) -> pd.DataFrame:
    """Compute marginal predicted applications per $1,000 per tactic at each Key.

    Uses the analytical OLS formula rather than numerical perturbation:

        marginal_apps_per_$1K = coefficient / MinMax_Range × 1000

    This is exact for linear OLS models with MinMax scaling and avoids
    the time_index contamination that affects numerical perturbation
    (score_from_coefficients_row rebuilds time_index as row position,
    which differs from the training-window time_index used in the
    baseline forecast, causing large spurious deltas).

    Parameters
    ----------
    spend_df : pd.DataFrame
        Weekly spend file (used only to identify which tactics are present).
    coeff_df : pd.DataFrame
        Coefficient table (model_coefficients_consolidated.csv).
    baseline_preds : pd.DataFrame
        Output of score_spend_with_coefficients + product allocation.
        Marginal columns are added to a copy of this DataFrame.
    model_type : str
        Retained for API compatibility — not used in analytical path.
    feature_run : str
        Retained for API compatibility — not used in analytical path.
    increment : float
        Retained for API compatibility — not used in analytical path.

    Returns
    -------
    pd.DataFrame
        baseline_preds with additional columns:
        Marginal_DSP, Marginal_LeadGen, Marginal_Paid_Search,
        Marginal_Paid_Social, Marginal_Prescreen, Marginal_Referrals.
        Values represent incremental predicted applications per $1,000.
    """
    TACTIC_TO_MARGINAL_COL = {t: f"Marginal_{safe_name(t)}" for t in NON_DUMMY_PREDICTORS}

    # Filter to OLS / weekly models only
    _coeff = coeff_df.copy()
    if "Model_Type" in _coeff.columns:
        _coeff = _coeff[_coeff["Model_Type"].astype(str).str.upper() == "OLS"]
    if "Feature_Run" in _coeff.columns:
        _coeff = _coeff[_coeff["Feature_Run"].astype(str).str.lower() == "weekly"]

    # Build {Key: {tactic: marginal_per_$1K}} lookup analytically
    # marginal = coef / range × 1000
    # This is exact for linear OLS with MinMax scaling and no transforms.
    marginal_lookup: Dict[str, Dict[str, float]] = {}
    for _, row in _coeff.iterrows():
        key = row.get("Key", "")
        if not key:
            continue
        marginal_lookup[key] = {}
        for tactic, marginal_col in TACTIC_TO_MARGINAL_COL.items():
            coef_val = row.get(tactic, np.nan)
            rng_val  = row.get(f"{tactic}__MinMax_Range", np.nan)
            if pd.isna(coef_val) or pd.isna(rng_val) or float(rng_val) == 0:
                marginal_lookup[key][marginal_col] = 0.0
            else:
                marginal_lookup[key][marginal_col] = (
                    float(coef_val) / float(rng_val) * 1_000.0
                )

    # Add marginal columns to baseline_preds
    result = baseline_preds.copy()
    for marginal_col in TACTIC_TO_MARGINAL_COL.values():
        result[marginal_col] = result["Key"].map(
            lambda k, mc=marginal_col: marginal_lookup.get(k, {}).get(mc, 0.0)
        )

    return result

def format_spend_coefficients(
    coefficient_map: Dict[str, float],
    media_feature_map: Dict[str, str],
) -> str:
    """Format the modeled spend coefficients for inclusion in consolidated diagnostics."""
    parts = []
    for raw_name, modeled_name in media_feature_map.items():
        if modeled_name in coefficient_map:
            parts.append(f"{raw_name}->{modeled_name}={float(coefficient_map[modeled_name]):.6g}")
    return "; ".join(parts)


def save_ols_artifacts(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    scaler: object,
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_model_path = grouped_artifact_path(output_root, metadata, "model") / f"{entity_prefix}_model.pkl"
    grouped_scaler_path = grouped_artifact_path(output_root, metadata, "scaler") / f"{entity_prefix}_scaler.joblib"
    grouped_coef_path = grouped_artifact_path(output_root, metadata, "coefficients_with_pvalues") / f"{entity_prefix}_coefficients_with_pvalues.csv"
    grouped_summary_path = grouped_artifact_path(output_root, metadata, "statsmodels_summary") / f"{entity_prefix}_statsmodels_summary.txt"
    grouped_metadata_path = grouped_artifact_path(output_root, metadata, "run_metadata") / f"{entity_prefix}_run_metadata.json"
    grouped_model_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_coef_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    result.save(str(artifact_dir / "model.pkl"))
    result.save(str(grouped_model_path))
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(scaler, grouped_scaler_path)

    coefficients = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.values,
            "p_value": result.pvalues.reindex(result.params.index).values,
            "std_error": result.bse.reindex(result.params.index).values,
            "t_value": result.tvalues.reindex(result.params.index).values,
        }
    )
    coefficients.to_csv(artifact_dir / "coefficients_with_pvalues.csv", index=False)
    coefficients.to_csv(grouped_coef_path, index=False)

    save_text(artifact_dir / "statsmodels_summary.txt", result.summary().as_text())
    save_text(grouped_summary_path, result.summary().as_text())
    metadata_payload = {
        **asdict(metadata),
        **scaler_metadata(scaler, metadata.scaler_type),
    }
    save_json(
        artifact_dir / "run_metadata.json",
        metadata_payload,
    )
    save_json(grouped_metadata_path, metadata_payload)


def save_nnls_artifacts(
    nnls_result: Dict[str, object],
    scaler: object,
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_model_path = grouped_artifact_path(output_root, metadata, "model") / f"{entity_prefix}_model.joblib"
    grouped_scaler_path = grouped_artifact_path(output_root, metadata, "scaler") / f"{entity_prefix}_scaler.joblib"
    grouped_coef_path = grouped_artifact_path(output_root, metadata, "coefficients") / f"{entity_prefix}_coefficients.csv"
    grouped_metadata_path = grouped_artifact_path(output_root, metadata, "run_metadata") / f"{entity_prefix}_run_metadata.json"
    grouped_model_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_coef_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(scaler, grouped_scaler_path)
    joblib.dump(nnls_result, artifact_dir / "model.joblib")
    joblib.dump(nnls_result, grouped_model_path)

    coefficients = pd.DataFrame(
        {
            "term": nnls_result["feature_names"],
            "coefficient": np.asarray(nnls_result["coef"], dtype=float),
        }
    )
    coefficients.to_csv(artifact_dir / "coefficients.csv", index=False)
    coefficients.to_csv(grouped_coef_path, index=False)

    metadata_payload = {
        **asdict(metadata),
        "intercept": nnls_result["intercept"],
        "residual_norm": nnls_result["residual_norm"],
        **scaler_metadata(scaler, metadata.scaler_type),
    }
    save_json(artifact_dir / "run_metadata.json", metadata_payload)
    save_json(grouped_metadata_path, metadata_payload)


def save_predictions(
    train_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    train_pred: Sequence[float],
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_predictions_path = grouped_artifact_path(output_root, metadata, "predictions") / f"{entity_prefix}_predictions.csv"
    grouped_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    train_output = train_df[[YEAR_COL, WEEK_COL, metadata.target_col]].copy()
    train_output["prediction"] = np.asarray(train_pred, dtype=float)
    train_output["dataset"] = "train"

    prediction_frame = pd.concat([train_output, test_predictions_df], axis=0, ignore_index=True)
    prediction_frame.to_csv(artifact_dir / "predictions.csv", index=False)
    prediction_frame.to_csv(grouped_predictions_path, index=False)


def create_actual_vs_predicted_figure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    metadata: RunMetadata,
) -> Tuple[plt.Figure, plt.Axes]:
    train_weeks = train_df[YEAR_COL].astype(str) + "-W" + train_df[WEEK_COL].astype(str).str.zfill(2)
    test_weeks = test_df[YEAR_COL].astype(str) + "-W" + test_df[WEEK_COL].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train_weeks, train_df[metadata.target_col], label="Train Actual", color="#1f77b4", linewidth=2)
    ax.plot(train_weeks, np.asarray(train_pred, dtype=float), label="Train Predicted", color="#ff7f0e", linestyle="--")
    ax.plot(test_weeks, test_df[metadata.target_col], label="Test Actual", color="#2ca02c", linewidth=2)
    ax.plot(test_weeks, np.asarray(test_pred, dtype=float), label="Test Predicted", color="#d62728", linestyle="--")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(metadata.target_col)
    ax.set_title(f"Actual vs Predicted: {metadata.scope}={metadata.entity} | {metadata.model_type} | {metadata.dummy_family}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_actual_vs_predicted(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_plot_path = grouped_artifact_path(output_root, metadata, "actual_vs_predicted") / f"{entity_prefix}_actual_vs_predicted.png"
    grouped_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = create_actual_vs_predicted_figure(train_df, test_df, train_pred, test_pred, metadata)
    fig.savefig(artifact_dir / "actual_vs_predicted.png", dpi=150)
    fig.savefig(grouped_plot_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_plot_path = grouped_artifact_path(output_root, metadata, "residuals") / f"{entity_prefix}_residuals.png"
    grouped_plot_path.parent.mkdir(parents=True, exist_ok=True)

    train_residuals = train_df[metadata.target_col].to_numpy(dtype=float) - np.asarray(train_pred, dtype=float)
    test_residuals = test_df[metadata.target_col].to_numpy(dtype=float) - np.asarray(test_pred, dtype=float)

    plt.figure(figsize=(12, 6))
    plt.scatter(np.asarray(train_pred, dtype=float), train_residuals, label="Train", color="#1f77b4", alpha=0.75)
    plt.scatter(np.asarray(test_pred, dtype=float), test_residuals, label="Test", color="#d62728", alpha=0.85)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot: {metadata.scope}={metadata.entity} | {metadata.model_type} | {metadata.dummy_family}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "residuals.png", dpi=150)
    plt.savefig(grouped_plot_path, dpi=150)
    plt.close()


def build_contribution_frame(
    x_frame: pd.DataFrame,
    coefficients: Dict[str, float],
    intercept: float,
) -> pd.DataFrame:
    contribution_frame = x_frame.copy()

    contribution_cols: List[str] = []
    for col in x_frame.columns:
        coef_value = float(coefficients.get(col, 0.0))
        contribution_col = f"{col}__contribution"
        contribution_frame[contribution_col] = x_frame[col].astype(float) * coef_value
        contribution_cols.append(contribution_col)

    contribution_frame["Intercept__contribution"] = intercept
    contribution_cols.append("Intercept__contribution")
    contribution_frame["Predicted_Total"] = contribution_frame[contribution_cols].sum(axis=1)
    return contribution_frame


def save_contribution_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    metadata: RunMetadata,
    coefficients: Dict[str, float],
    intercept: float,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_csv_path = grouped_artifact_path(output_root, metadata, "contribution_decomposition_csv") / f"{entity_prefix}_contribution_decomposition.csv"
    grouped_png_path = grouped_artifact_path(output_root, metadata, "contribution_decomposition") / f"{entity_prefix}_contribution_decomposition.png"
    grouped_csv_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_png_path.parent.mkdir(parents=True, exist_ok=True)

    train_contrib = build_contribution_frame(x_train, coefficients, intercept)
    train_contrib[YEAR_COL] = train_df[YEAR_COL].to_numpy()
    train_contrib[WEEK_COL] = train_df[WEEK_COL].to_numpy()
    train_contrib["dataset"] = "train"

    test_contrib = build_contribution_frame(x_test, coefficients, intercept)
    test_contrib[YEAR_COL] = test_df[YEAR_COL].to_numpy()
    test_contrib[WEEK_COL] = test_df[WEEK_COL].to_numpy()
    test_contrib["dataset"] = "test"

    contribution_df = pd.concat([train_contrib, test_contrib], axis=0, ignore_index=True)
    contribution_df.to_csv(artifact_dir / "contribution_decomposition.csv", index=False)
    contribution_df.to_csv(grouped_csv_path, index=False)

    contribution_columns = [
        col
        for col in contribution_df.columns
        if col.endswith("__contribution") and not col.startswith("Intercept__")
    ]

    contribution_summary = contribution_df.groupby("dataset")[contribution_columns + ["Intercept__contribution"]].sum().T
    contribution_summary.columns = [str(col).title() for col in contribution_summary.columns]
    contribution_summary["abs_total"] = contribution_summary.abs().sum(axis=1)
    contribution_summary = contribution_summary.sort_values("abs_total", ascending=False).drop(columns=["abs_total"])
    contribution_summary = contribution_summary.head(12)

    fig, ax = plt.subplots(figsize=(12, 7))
    contribution_summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("Contribution")
    ax.set_title(
        f"Contribution Decomposition: {metadata.scope}={metadata.entity} | {metadata.model_type} | {metadata.dummy_family}"
    )
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(artifact_dir / "contribution_decomposition.png", dpi=150)
    fig.savefig(grouped_png_path, dpi=150)
    plt.close(fig)


def display_inline_review(
    metadata: RunMetadata,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    test_y: pd.Series,
    ols_result: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None,
) -> None:
    """Display notebook-friendly inline output for a user-selected subset.

    Inline review is intended for interactive inspection in Jupyter. For OLS, the full
    statsmodels summary is shown.
    For all inline reviews, an actual-vs-predicted chart and an average-bias diagnostic
    are displayed.
    """
    average_bias = float(np.mean(np.asarray(test_pred, dtype=float) - np.asarray(test_y, dtype=float)))

    try:
        from IPython.display import Markdown, display

        display(
            Markdown(
                f"### {metadata.scope.title()}: `{metadata.entity}` | {metadata.model_type} | {metadata.dummy_family}\n"
                f"- Train rows: {metadata.train_rows}\n"
                f"- Test rows: {metadata.test_rows}\n"
                f"- Scaler: `{metadata.scaler_type}`\n"
                f"- Average Bias (`predicted - actual`): `{average_bias:.4f}`"
            )
        )
        if ols_result is not None:
            display(Markdown("#### Statsmodels Summary"))
            print(ols_result.summary().as_text())

        fig, _ = create_actual_vs_predicted_figure(train_df, test_df, train_pred, test_pred, metadata)
        display(fig)
        plt.close(fig)
    except ImportError:
        print(
            f"{metadata.scope.title()}: {metadata.entity} | {metadata.model_type} | {metadata.dummy_family} | "
            f"Average Bias (predicted - actual): {average_bias:.4f}"
        )
        if ols_result is not None:
            print(ols_result.summary().as_text())


def display_forecast_skip_notice(
    scope_name: str,
    entity_name: str,
    model_type: str,
    run_name: str,
    reason: str,
) -> None:
    """Display a non-fatal forecast skip message inline and continue processing."""
    message = (
        f"Skipped forecast for {scope_name} `{entity_name}` | {model_type} | {run_name}: {reason}"
    )
    try:
        from IPython.display import Markdown, display

        display(Markdown(f"**Notice**: {message}"))
    except ImportError:
        print(message)


def build_forecast_predictions_for_entity(
    history_entity_df: pd.DataFrame,
    future_entity_df: pd.DataFrame,
    scope_name: str,
    entity_name: str,
    run_name: str,
    model_type: str,
    media_predictors: Sequence[str],
    media_transform_config: Dict[str, Dict[str, Any]],
    optional_features: Sequence[str],
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RunMetadata]:
    """Generate recursive future forecasts for one entity.

    The model is fit once on observed history only. Future predictions are generated
    sequentially, and prior predicted targets are written back only for lag-feature
    construction. Predicted targets are never added to the training set.
    """
    combined = pd.concat(
        [
            history_entity_df.assign(__is_observed_target=True, __is_future=False),
            future_entity_df.assign(__is_observed_target=False, __is_future=True),
        ],
        axis=0,
        ignore_index=True,
    ).sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)

    future_indices = combined.index[combined["__is_future"]].tolist()
    if not future_indices:
        raise ForecastSkipError("no future rows available")

    forecast_rows: List[Dict[str, Any]] = []
    last_metadata: Optional[RunMetadata] = None
    last_fitted_model: Optional[object] = None
    last_scaler: Optional[object] = None
    last_y_train: Optional[pd.Series] = None
    last_train_pred: Optional[np.ndarray] = None

    for current_idx in future_indices:
        feature_df = add_optional_features(
            combined.drop(columns=["__is_observed_target", "__is_future"]),
            optional_features,
            target_col=target_col,
        )
        feature_df["__is_observed_target"] = combined["__is_observed_target"].to_numpy()
        feature_df["__is_future"] = combined["__is_future"].to_numpy()

        train_df = feature_df[feature_df["__is_observed_target"]].drop(columns=["__is_observed_target", "__is_future"]).copy()
        test_df = feature_df.loc[future_indices[0]:current_idx].drop(columns=["__is_observed_target", "__is_future"]).copy()

        x_train, y_train, x_test, _, scaler, predictors, dropped_dummy, media_feature_map = build_design_matrices(
            train_df=train_df,
            test_df=test_df,
            run_name=run_name,
            media_predictors=media_predictors,
            media_transform_config=media_transform_config,
            optional_features=optional_features,
            target_col=target_col,
        )
        if x_train.empty:
            raise ForecastSkipError("no usable predictors remain after design-matrix construction")

        x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
        x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_current = x_test.tail(1).copy()
        raw_future_row = test_df.tail(1).copy()

        if model_type == "OLS":
            if len(y_train) <= x_train.shape[1] + 1:
                raise ForecastSkipError(
                    f"not enough observed history to fit OLS ({len(y_train)} rows for {x_train.shape[1]} predictors)"
                )
            fitted_model = fit_ols(x_train, y_train)
            x_current_const = sm.add_constant(x_current, has_constant="add").reindex(
                fitted_model.model.exog_names, axis=1, fill_value=0.0
            )
            current_pred = float(
                fitted_model.predict(x_current_const).iloc[0]
            )
            prediction_frame = fitted_model.get_prediction(x_current_const).summary_frame(alpha=0.05)
            lower_limit = float(
                prediction_frame.iloc[0].get(
                    "obs_ci_lower",
                    prediction_frame.iloc[0].get("mean_ci_lower", np.nan),
                )
            )
            upper_limit = float(
                prediction_frame.iloc[0].get(
                    "obs_ci_upper",
                    prediction_frame.iloc[0].get("mean_ci_upper", np.nan),
                )
            )
            train_pred = np.asarray(
                fitted_model.predict(sm.add_constant(x_train, has_constant="add")).to_numpy(),
                dtype=float,
            )
        else:
            fitted_model = fit_nnls(x_train, y_train)
            current_pred = float(predict_nnls(fitted_model, x_current)[0])
            lower_limit = np.nan
            upper_limit = np.nan
            train_pred = np.asarray(fitted_model["train_pred"], dtype=float)

        if not np.isfinite(current_pred):
            raise ForecastSkipError("prediction evaluated to a non-finite value")

        model_coefficients = coefficient_lookup(model_type, fitted_model)
        current_feature_values = x_current.iloc[0].to_dict()
        time_index_contribution = float(
            current_feature_values.get(TIME_INDEX_COL, 0.0) * model_coefficients.get(TIME_INDEX_COL, 0.0)
        )
        time_index_sq_contribution = float(
            current_feature_values.get(TIME_INDEX_SQ_COL, 0.0) * model_coefficients.get(TIME_INDEX_SQ_COL, 0.0)
        )
        trend_contribution = time_index_contribution + time_index_sq_contribution

        combined.loc[current_idx, target_col] = current_pred

        forecast_rows.append(
            {
                "Key": entity_name,
                "Run_Status": "SUCCESS",
                "State": future_output_value(raw_future_row.iloc[0], STATE_COL),
                "ISO_Year": int(raw_future_row.iloc[0][YEAR_COL]),
                "ISO_Week": int(raw_future_row.iloc[0][WEEK_COL]),
                "Month": iso_week_month(int(raw_future_row.iloc[0][YEAR_COL]), int(raw_future_row.iloc[0][WEEK_COL])),
                **{col: float(raw_future_row.iloc[0].get(col, 0.0)) for col in FUTURE_SPEND_OUTPUT_TACTICS},
                "Channel": future_output_value(raw_future_row.iloc[0], "CHANNEL_CD"),
                "H_Tactic": future_output_value(raw_future_row.iloc[0], "H_TACTIC"),
                "Detail_Tactic": future_output_value(raw_future_row.iloc[0], "DETAIL_TACTIC"),
                "Product": future_output_value(raw_future_row.iloc[0], "PRODUCT_CD"),
                "Predicted APPS Raw": max(0.0, float(current_pred)),
                "Predicted APPS": format_predicted_apps(current_pred),
                "95% Confidence Lower Limit": lower_limit,
                "95% Confidence Upper Limit": upper_limit,
                "Time_Index_Contribution": time_index_contribution,
                "Time_Index_Sq_Contribution": time_index_sq_contribution,
                "Trend_Contribution": trend_contribution,
                "Scope": scope_name,
                "Model_Type": model_type,
                "Feature_Run": run_name,
            }
        )

        last_metadata = RunMetadata(
            scope=scope_name,
            entity=entity_name,
            model_type=model_type,
            dummy_family=run_name,
            dropped_dummy=dropped_dummy,
            train_rows=len(train_df),
            test_rows=len(future_indices),
            predictors=predictors,
            scaler_type=str(FEATURE_RUNS[run_name]["scaler"]),
            backtest_mode="future_recursive_forecast",
            media_transform_config=media_transform_config,
            target_col=target_col,
        )
        last_fitted_model = fitted_model
        last_scaler = scaler
        last_y_train = y_train.copy()
        last_train_pred = train_pred.copy()

    if last_metadata is None:
        raise ForecastSkipError("unable to generate future forecasts from the available history")
    if last_fitted_model is None or last_scaler is None or last_y_train is None or last_train_pred is None:
        raise ForecastSkipError("incomplete forecast artifacts were produced")

    forecast_df = pd.DataFrame(forecast_rows).sort_values(["ISO_Year", "ISO_Week"]).reset_index(drop=True)
    coefficients_df = build_future_coefficients_frame(
        metadata=last_metadata,
        model_type=model_type,
        fitted_model=last_fitted_model,
        scaler=last_scaler,
    )
    if model_type == "OLS":
        training_summary_row = build_future_training_summary_row(
            metadata=last_metadata,
            train_y=last_y_train,
            train_pred=last_train_pred,
            r2_value=float(last_fitted_model.rsquared),
            adj_r2_value=float(last_fitted_model.rsquared_adj),
        )
    else:
        training_summary_row = build_future_training_summary_row(
            metadata=last_metadata,
            train_y=last_y_train,
            train_pred=last_train_pred,
            r2_value=float(last_fitted_model["train_r2"]),
            adj_r2_value=float(last_fitted_model["train_adj_r2"]),
        )
    training_summary_df = pd.DataFrame([training_summary_row])
    return forecast_df, coefficients_df, training_summary_df, last_metadata


def build_skipped_forecast_row(
    scope_name: str,
    entity_name: str,
    model_type: str,
    run_name: str,
    reason: str,
) -> pd.DataFrame:
    """Create a tracking row for a model slice that was skipped during forecasting."""
    return pd.DataFrame(
        [
            {
                "Key": entity_name,
                "Run_Status": f"SKIPPED: {reason}",
                "State": "",
                "ISO_Year": np.nan,
                "ISO_Week": np.nan,
                "Month": np.nan,
                **{col: np.nan for col in FUTURE_SPEND_OUTPUT_TACTICS},
                "Channel": "",
                "H_Tactic": "",
                "Detail_Tactic": "",
                "Product": "",
                "Predicted APPS Raw": np.nan,
                "Predicted APPS": np.nan,
                "95% Confidence Lower Limit": np.nan,
                "95% Confidence Upper Limit": np.nan,
                "Time_Index_Contribution": np.nan,
                "Time_Index_Sq_Contribution": np.nan,
                "Trend_Contribution": np.nan,
                "Scope": scope_name,
                "Model_Type": model_type,
                "Feature_Run": run_name,
            }
        ]
    )


def run_scope(
    df: pd.DataFrame,
    scope_col: str,
    scope_name: str,
    output_root: Optional[Path],
    model_types: Set[str],
    feature_runs: Set[str],
    media_predictors: Sequence[str],
    media_transform_config: Dict[str, Dict[str, Any]],
    dataset_group_by: Sequence[str],
    selected_entities: Optional[Sequence[str]],
    optional_features: Sequence[str],
    inline_output: bool,
    backtest_mode: str,
    fixed_window_weeks: int,
    target_col: str = TARGET_COL,
) -> List[Dict[str, object]]:
    """Run the requested model families for one scope.

    Parameters:
        df: Full validated modeling dataset.
        scope_col: ``STATE_CD`` or ``Division``.
        scope_name: Human-readable scope label used in outputs.
        output_root: Root folder for saved artifacts. Pass ``None`` when using inline notebook mode.
        model_types: Subset of ``{"OLS", "NNLS"}``.
        feature_runs: Subset of ``{"weekly", "f_dummy", "fourier"}``.
        media_predictors: Selected subset of the six base media variables.
        media_transform_config: Optional per-media transform settings. Any transformed media
            variable replaces its raw form in the design matrix.
        dataset_group_by: Additional grouping keys such as CHANNEL_CD or DETAIL_TACTIC.
        selected_entities: Optional list of states or divisions to run.
        optional_features: Optional engineered variables to include in addition to the default predictors.
        inline_output: If ``True``, show notebook-friendly summaries instead of writing files.
        backtest_mode: Backtest style used for evaluation.
        fixed_window_weeks: Training window size for fixed-window rolling backtests.
    """
    diagnostics: List[Dict[str, object]] = []
    scope_df = prepare_entity_subset(df, scope_col, selected_entities)

    for run_name, run_config in FEATURE_RUNS.items():
        if run_name not in feature_runs:
            continue

        if run_name == "fourier" and not all(col in scope_df.columns for col in FOURIER_COLS):
            continue
        if run_name in {"weekly", "f_dummy"} and not any(
            col in scope_df.columns for col in FEATURE_RUNS[run_name]["extra_cols"]
        ):
            continue

        grouping_keys = list(dict.fromkeys([scope_col, *dataset_group_by]))

        for entity_key, entity_df in scope_df.groupby(grouping_keys, dropna=False):
            modeling_df = entity_df
            if scope_name == "division":
                modeling_df = aggregate_division_weekly(
                    entity_df,
                    run_name,
                    target_col=target_col,
                    dataset_group_by=dataset_group_by,
                )
            modeling_df = add_optional_features(modeling_df, optional_features, target_col=target_col)

            backtest_splits = generate_backtest_splits(modeling_df, backtest_mode, fixed_window_weeks)
            if not backtest_splits:
                continue

            if not isinstance(entity_key, tuple):
                entity_key = (entity_key,)
            entity_name = " | ".join(
                f"{col}={value}" for col, value in zip(grouping_keys, entity_key)
            )

            if "OLS" in model_types:
                ols_result = None
                ols_last_scaler = None
                ols_last_train_df = None
                ols_last_x_train = None
                ols_last_test_df = None
                ols_last_x_test = None
                ols_last_train_pred = None
                ols_test_frames: List[pd.DataFrame] = []

                for train_df, test_df in backtest_splits:
                    x_train, y_train, x_test, y_test, scaler, predictors, dropped_dummy, media_feature_map = build_design_matrices(
                        train_df=train_df,
                        test_df=test_df,
                        run_name=run_name,
                        media_predictors=media_predictors,
                        media_transform_config=media_transform_config,
                        optional_features=optional_features,
                        target_col=target_col,
                    )
                    if x_train.empty or len(y_train) <= x_train.shape[1] + 1:
                        continue

                    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
                    current_result = fit_ols(x_train, y_train)
                    current_train_pred = current_result.predict(sm.add_constant(x_train, has_constant="add"))
                    current_test_pred = current_result.predict(
                        sm.add_constant(x_test, has_constant="add").reindex(current_result.model.exog_names, axis=1, fill_value=0.0)
                    )

                    test_output = test_df[[YEAR_COL, WEEK_COL, target_col]].copy()
                    test_output["prediction"] = np.asarray(current_test_pred, dtype=float)
                    test_output["dataset"] = "test"
                    ols_test_frames.append(test_output)

                    ols_result = current_result
                    ols_last_scaler = scaler
                    ols_last_train_df = train_df.copy()
                    ols_last_x_train = x_train.copy()
                    ols_last_test_df = test_df.copy()
                    ols_last_x_test = x_test.copy()
                    ols_last_train_pred = np.asarray(current_train_pred, dtype=float)
                    ols_media_feature_map = dict(media_feature_map)

                if (
                    ols_result is None
                    or ols_last_scaler is None
                    or ols_last_train_df is None
                    or ols_last_x_train is None
                    or ols_last_test_df is None
                    or ols_last_x_test is None
                ):
                    continue

                ols_test_output = pd.concat(ols_test_frames, axis=0, ignore_index=True).sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)
                y_test = ols_test_output[target_col].astype(float)
                ols_test_pred = ols_test_output["prediction"].astype(float).to_numpy()
                ols_metadata = RunMetadata(
                    scope=scope_name,
                    entity=entity_name,
                    model_type="OLS",
                    dummy_family=run_name,
                    dropped_dummy=dropped_dummy,
                    train_rows=len(ols_last_train_df),
                    test_rows=len(ols_test_output),
                    predictors=predictors,
                    scaler_type=str(run_config["scaler"]),
                    backtest_mode=backtest_mode,
                    media_transform_config=media_transform_config,
                    target_col=target_col,
                )
                diagnostics.append(
                    model_diagnostics_row(
                        metadata=ols_metadata,
                        train_y=ols_last_train_df[target_col].astype(float),
                        train_pred=ols_last_train_pred,
                        test_y=y_test,
                        test_pred=ols_test_pred,
                        train_r2=float(ols_result.rsquared),
                        train_adj_r2=float(ols_result.rsquared_adj),
                        aic=float(ols_result.aic),
                        bic=float(ols_result.bic),
                        spend_coefficients=format_spend_coefficients(
                            {name: value for name, value in ols_result.params.items() if name != "const"},
                            ols_media_feature_map,
                        ),
                    )
                )
                if inline_output:
                    display_inline_review(
                        ols_metadata,
                        ols_last_train_df,
                        ols_test_output,
                        ols_last_train_pred,
                        ols_test_pred,
                        y_test,
                        ols_result,
                    )
                else:
                    save_ols_artifacts(ols_result, ols_last_scaler, ols_metadata, output_root)
                    save_predictions(ols_last_train_df, ols_test_output, ols_last_train_pred, ols_metadata, output_root)
                    plot_actual_vs_predicted(ols_last_train_df, ols_test_output, ols_last_train_pred, ols_test_pred, ols_metadata, output_root)
                    plot_residuals(ols_last_train_df, ols_test_output, ols_last_train_pred, ols_test_pred, ols_metadata, output_root)
                    save_contribution_outputs(
                        train_df=ols_last_train_df,
                        test_df=ols_last_test_df,
                        x_train=ols_last_x_train,
                        x_test=ols_last_x_test.reindex(columns=ols_last_x_train.columns, fill_value=0.0),
                        metadata=ols_metadata,
                        coefficients={name: value for name, value in ols_result.params.items() if name != "const"},
                        intercept=float(ols_result.params.get("const", 0.0)),
                        output_root=output_root,
                    )

            if "NNLS" in model_types:
                nnls_result = None
                nnls_last_scaler = None
                nnls_last_train_df = None
                nnls_last_x_train = None
                nnls_last_test_df = None
                nnls_last_x_test = None
                nnls_last_train_pred = None
                nnls_test_frames: List[pd.DataFrame] = []

                for train_df, test_df in backtest_splits:
                    x_train, y_train, x_test, y_test, scaler, predictors, dropped_dummy, media_feature_map = build_design_matrices(
                        train_df=train_df,
                        test_df=test_df,
                        run_name=run_name,
                        media_predictors=media_predictors,
                        media_transform_config=media_transform_config,
                        optional_features=optional_features,
                        target_col=target_col,
                    )
                    if x_train.empty:
                        continue
                    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
                    current_result = fit_nnls(x_train, y_train)
                    current_train_pred = np.asarray(current_result["train_pred"], dtype=float)
                    current_test_pred = predict_nnls(current_result, x_test)

                    test_output = test_df[[YEAR_COL, WEEK_COL, target_col]].copy()
                    test_output["prediction"] = np.asarray(current_test_pred, dtype=float)
                    test_output["dataset"] = "test"
                    nnls_test_frames.append(test_output)

                    nnls_result = current_result
                    nnls_last_scaler = scaler
                    nnls_last_train_df = train_df.copy()
                    nnls_last_x_train = x_train.copy()
                    nnls_last_test_df = test_df.copy()
                    nnls_last_x_test = x_test.copy()
                    nnls_last_train_pred = current_train_pred
                    nnls_media_feature_map = dict(media_feature_map)

                if (
                    nnls_result is None
                    or nnls_last_scaler is None
                    or nnls_last_train_df is None
                    or nnls_last_x_train is None
                    or nnls_last_test_df is None
                    or nnls_last_x_test is None
                ):
                    continue

                nnls_test_output = pd.concat(nnls_test_frames, axis=0, ignore_index=True).sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)
                y_test = nnls_test_output[target_col].astype(float)
                nnls_test_pred = nnls_test_output["prediction"].astype(float).to_numpy()
                nnls_metadata = RunMetadata(
                    scope=scope_name,
                    entity=entity_name,
                    model_type="NNLS",
                    dummy_family=run_name,
                    dropped_dummy=dropped_dummy,
                    train_rows=len(nnls_last_train_df),
                    test_rows=len(nnls_test_output),
                    predictors=predictors,
                    scaler_type=str(run_config["scaler"]),
                    backtest_mode=backtest_mode,
                    media_transform_config=media_transform_config,
                    target_col=target_col,
                )
                diagnostics.append(
                    model_diagnostics_row(
                        metadata=nnls_metadata,
                        train_y=nnls_last_train_df[target_col].astype(float),
                        train_pred=nnls_last_train_pred,
                        test_y=y_test,
                        test_pred=nnls_test_pred,
                        train_r2=float(nnls_result["train_r2"]),
                        train_adj_r2=float(nnls_result["train_adj_r2"]),
                        aic=float(nnls_result["aic"]),
                        bic=float(nnls_result["bic"]),
                        spend_coefficients=format_spend_coefficients(
                            dict(zip(nnls_result["feature_names"], np.asarray(nnls_result["coef"], dtype=float))),
                            nnls_media_feature_map,
                        ),
                    )
                )
                if inline_output:
                    display_inline_review(
                        nnls_metadata,
                        nnls_last_train_df,
                        nnls_test_output,
                        nnls_last_train_pred,
                        nnls_test_pred,
                        y_test,
                    )
                else:
                    save_nnls_artifacts(nnls_result, nnls_last_scaler, nnls_metadata, output_root)
                    save_predictions(nnls_last_train_df, nnls_test_output, nnls_last_train_pred, nnls_metadata, output_root)
                    plot_actual_vs_predicted(nnls_last_train_df, nnls_test_output, nnls_last_train_pred, nnls_test_pred, nnls_metadata, output_root)
                    plot_residuals(nnls_last_train_df, nnls_test_output, nnls_last_train_pred, nnls_test_pred, nnls_metadata, output_root)
                    save_contribution_outputs(
                        train_df=nnls_last_train_df,
                        test_df=nnls_last_test_df,
                        x_train=nnls_last_x_train,
                        x_test=nnls_last_x_test.reindex(columns=nnls_last_x_train.columns, fill_value=0.0),
                        metadata=nnls_metadata,
                        coefficients=dict(zip(nnls_result["feature_names"], np.asarray(nnls_result["coef"], dtype=float))),
                        intercept=float(nnls_result["intercept"]),
                        output_root=output_root,
                    )

    return diagnostics


def run_model_pipeline(
    input_path: str = "/Users/Rahul/Desktop/Code/Working Codebase/ModelingFile_Digital.csv",
    output_dir: Optional[str] = "state_division_model_artifacts",
    target_col: str = TARGET_COL,
    dataset_group_by: Optional[Sequence[str]] = None,
    selected_states: Optional[Sequence[str]] = None,
    selected_divisions: Optional[Sequence[str]] = None,
    methodologies: Optional[Sequence[str]] = None,
    media_predictors: Optional[Sequence[str]] = None,
    media_transform_config: Optional[Dict[str, Dict[str, Any]]] = None,
    optional_features: Optional[Sequence[str]] = None,
    inline_output: bool = False,
    backtest_mode: str = "fixed_holdout",
    fixed_window_weeks: int = DEFAULT_FIXED_WINDOW_WEEKS,
) -> pd.DataFrame:
    """
    Run the full modeling workflow and return the consolidated diagnostics DataFrame.

    Default behavior:
        Writes model artifacts to ``output_dir`` exactly as before, and also creates
        grouped comparison folders under ``output_dir/by_artifact`` so the same artifact
        type for all states or divisions sits together for easier review.

    Notebook inline review mode:
        Set ``inline_output=True`` to display detailed OLS statsmodels summaries, actual-vs-predicted
        charts, and average-bias diagnostics directly inside Jupyter instead of writing files.

    Parameters:
        input_path: CSV file to model.
        output_dir: Folder for saved artifacts. Leave as the default for file outputs. Pass ``None``
            or keep it unused when ``inline_output=True``.
        target_col: Dependent variable to model.
        dataset_group_by: Optional extra grouping keys such as ``CHANNEL_CD`` or ``DETAIL_TACTIC``.
            Separate models are built for each unique combination of scope and these keys.
        selected_states: Optional list of state codes to run. When omitted, all eligible states run.
        selected_divisions: Optional list of division names to run. When omitted, all eligible divisions run.
        methodologies: Optional list controlling model types and feature runs.
            Supported values:
                - ``OLS``
                - ``NNLS``
                - ``weekly``
                - ``f_dummy``
                - ``Fourier`` or ``fourier``
        media_predictors: Optional subset of the six base media variables. If omitted, the
            script uses all six by default:
                - ``DSP``
                - ``LeadGen``
                - ``Paid Search``
                - ``Paid Social``
                - ``Prescreen``
                - ``Referrals``
        media_transform_config: Optional per-media transform settings. Example:
            ``{"DSP": {"alpha": 0.5, "saturation": "log1p"}, "Prescreen": {"alpha": 0.7}}``
            Any media variable not listed here is used raw. Any media variable listed here is
            used only in transformed form.
        optional_features: Optional engineered variables to add on top of the default predictors.
            Available options:
                - ``time_index``
                - ``time_index_sq``
                - ``year_indicator_2025``
                - ``year_indicator_2026``
                - ``Prescreen_lag1``
                - ``DSP_lag1``
                - ``Paid_Search_lag1``
                - ``DSP_trailing_4w_avg``
                - ``Paid_Search_trailing_4w_avg``
                - ``Prescreen_trailing_4w_avg``
                - ``APPLICATIONS_lag1`` or ``NON_DM_APPLICATIONS_lag1`` depending on the active target
                - ``APPLICATIONS_trailing_4w_avg`` or ``NON_DM_APPLICATIONS_trailing_4w_avg`` depending on the active target
        inline_output: If ``True``, show inline notebook output and skip writing model folders.
        backtest_mode: One of ``fixed_holdout``, ``rolling_one_step_expanding``, or
            ``rolling_one_step_fixed_window``.
        fixed_window_weeks: Number of prior rows retained for the fixed-window rolling backtest.

    Example for Jupyter:
        from build_state_division_models import run_model_pipeline
        diagnostics_df = run_model_pipeline(
            selected_states=["CA", "TX"],
            methodologies=["OLS", "Fourier"],
            media_predictors=["DSP", "Prescreen", "Paid Search"],
            media_transform_config={"DSP": {"alpha": 0.5, "saturation": "log1p"}},
            optional_features=["time_index", "Prescreen_lag1"],
            inline_output=True,
            output_dir=None,
            backtest_mode="rolling_one_step_expanding",
        )
    """
    validated_media_predictors = validate_media_predictors(media_predictors)
    validated_media_transform_config = validate_media_transform_config(
        media_transform_config,
        validated_media_predictors,
    )
    validated_optional_features = validate_optional_features(optional_features, target_col=target_col)
    model_types, feature_runs = parse_methodology_selection(methodologies)
    if backtest_mode not in BACKTEST_MODES:
        raise ValueError(f"Unsupported backtest mode '{backtest_mode}'. Choices: {sorted(BACKTEST_MODES)}")

    if inline_output:
        selected_count = len(selected_states or []) + len(selected_divisions or [])
        if selected_count == 0:
            raise ValueError("Inline output requires at least one selected state or division.")
        output_root = None
    else:
        output_root = Path(output_dir or "state_division_model_artifacts")
        output_root.mkdir(parents=True, exist_ok=True)

    dataset_group_by = list(dataset_group_by or [])
    df = load_data(input_path, target_col=target_col, dataset_group_by=dataset_group_by)

    diagnostics_rows: List[Dict[str, object]] = []
    diagnostics_rows.extend(
        run_scope(
            df,
            STATE_COL,
            "state",
            output_root,
            model_types,
            feature_runs,
            validated_media_predictors,
            validated_media_transform_config,
            dataset_group_by,
            selected_states,
            validated_optional_features,
            inline_output,
            backtest_mode,
            fixed_window_weeks,
            target_col,
        )
    )
    diagnostics_rows.extend(
        run_scope(
            df,
            DIVISION_COL,
            "division",
            output_root,
            model_types,
            feature_runs,
            validated_media_predictors,
            validated_media_transform_config,
            dataset_group_by,
            selected_divisions,
            validated_optional_features,
            inline_output,
            backtest_mode,
            fixed_window_weeks,
            target_col,
        )
    )

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    if not diagnostics_df.empty:
        diagnostics_df = diagnostics_df.sort_values(
            ["dummy_family", "scope", "entity", "model_type"]
        ).reset_index(drop=True)
    if output_root is not None:
        diagnostics_df.to_csv(output_root / "consolidated_model_diagnostics.csv", index=False)

    if output_root is not None:
        with (output_root / "run_manifest.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "input_path": input_path,
                    "output_dir": str(output_root.resolve()),
                    "target_col": target_col,
                    "dataset_group_by": dataset_group_by,
                    "diagnostics_rows": len(diagnostics_df),
                    "feature_runs": sorted(feature_runs),
                    "model_types": sorted(model_types),
                    "media_predictors": validated_media_predictors,
                    "media_transform_config": validated_media_transform_config,
                    "optional_features": validated_optional_features,
                    "selected_states": list(selected_states or []),
                    "selected_divisions": list(selected_divisions or []),
                    "backtest_mode": backtest_mode,
                    "fixed_window_weeks": fixed_window_weeks,
                    "train_years": sorted(TRAIN_YEARS),
                    "test_year": TEST_YEAR,
                    "test_weeks": sorted(TEST_WEEKS),
                },
                handle,
            )

    return diagnostics_df


def run_future_forecast(
    history_input_path: Union[str, Path, pd.DataFrame],
    future_input_path: Union[str, Path, pd.DataFrame],
    target_col: str = TARGET_COL,
    dataset_group_by: Optional[Sequence[str]] = None,
    selected_states: Optional[Sequence[str]] = None,
    selected_divisions: Optional[Sequence[str]] = None,
    methodologies: Optional[Sequence[str]] = None,
    media_predictors: Optional[Sequence[str]] = None,
    media_transform_config: Optional[Dict[str, Dict[str, Any]]] = None,
    optional_features: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate future week-by-week forecasts from planned spend.

    This function is designed for the case where you have observed historical data plus
    a future weekly spend plan, for example the next four weeks. Forecasting is sequential:
    week 1 is predicted first, then that prediction is fed into later target-lag features
    when required.

    Key behavior:
        - uses the same dataset aggregation logic as ``run_model_pipeline``
        - honors ``target_col`` and ``dataset_group_by`` exactly the same way
        - tolerates datasets that do not include Fourier or ``F_*`` seasonal columns
        - silently skips unavailable seasonal methodologies rather than failing
        - observed history is used for model fitting
        - predicted future targets are used only to build later lag/rolling target features
        - predicted future targets are not added back into the training set
        - future forecasting is state-based only and does not use division-level fallback models

    Parameters:
        history_input_path: Historical dataset with observed target values. This can be either
            a CSV path or a pandas DataFrame.
        future_input_path: Future weekly spend plan with the same structural columns, but the
            target can be omitted or left empty. This can be either a CSV path or a pandas
            DataFrame.
        target_col: Dependent variable used for recursive forecasting.
        dataset_group_by: Optional extra grouping keys such as ``CHANNEL_CD`` or ``DETAIL_TACTIC``.
            These keys define separate aggregated modeling series exactly as in
            ``run_model_pipeline``.
        selected_states: Optional list of states to forecast. When provided, both model
            fitting history and future predictions are restricted to those states only.
        selected_divisions: Accepted for API compatibility, but ignored for future
            forecasting because future predictions are state-based only.
        methodologies: Optional list controlling model types and feature runs.
        media_predictors: Optional subset of the six base media variables.
        media_transform_config: Optional per-media transform settings.
        optional_features: Optional engineered variables, including target lags if desired.
        output_dir: Optional folder where the forecast output files will be written:
            ``future_forecast.csv``, ``monthly_forecast.csv``,
            ``model_coefficients.csv``, and ``model_training_summary.csv``.

    Example:
        forecast_outputs = run_future_forecast(
            history_input_path="/path/to/history.csv",
            future_input_path="/path/to/future_4_weeks.csv",
            selected_states=["CA", "TX"],
            methodologies=["OLS", "Fourier"],
            media_predictors=["DSP", "Prescreen", "Paid Search"],
            optional_features=["time_index", "APPLICATIONS_lag1", "APPLICATIONS_trailing_4w_avg"],
        )

    Example with DataFrames:
        forecast_outputs = run_future_forecast(
            history_input_path=history_df,
            future_input_path=future_df,
            target_col="NON_DM_APPLICATIONS",
            selected_states=["CA", "TX"],
        )

    Future spend prep:
        If the future spend file arrives in the raw transaction-level layout with
        ``DETAIL_TACTIC``, ``BUSINESS_DATE``, ``STATE_CD``, ``PRODUCT_CD``, and
        ``TOTAL_COST``, first call :func:`prepare_future_spend_data` to create the
        weekly wide spend matrix expected by the forecasting pipeline.
    """
    requested_dataset_group_by = list(dataset_group_by or [])
    validated_media_predictors = validate_media_predictors(media_predictors)
    validated_media_transform_config = validate_media_transform_config(
        media_transform_config,
        validated_media_predictors,
    )
    validated_optional_features = validate_optional_features(optional_features, target_col=target_col)
    model_types, feature_runs = parse_methodology_selection(methodologies)

    history_df = load_data(
        history_input_path,
        target_col=target_col,
        dataset_group_by=requested_dataset_group_by,
    )
    future_df = load_future_data(
        future_input_path,
        target_col=target_col,
        dataset_group_by=requested_dataset_group_by,
    )

    if selected_states:
        history_df = prepare_entity_subset(history_df, STATE_COL, selected_states)
        future_df = prepare_entity_subset(future_df, STATE_COL, selected_states)
    product_factors_df = build_product_allocation_factors(
        history_input_source=history_input_path,
        dataset_group_by=requested_dataset_group_by,
        selected_states=selected_states,
    )

    forecast_outputs: List[pd.DataFrame] = []
    coefficient_outputs: List[pd.DataFrame] = []
    training_summary_outputs: List[pd.DataFrame] = []
    scopes = [
        (STATE_COL, "state", selected_states),
    ]

    for scope_col, scope_name, selected_entities in scopes:
        history_scope_df = prepare_entity_subset(history_df, scope_col, selected_entities)
        future_scope_df = prepare_entity_subset(future_df, scope_col, selected_entities)

        for run_name in FEATURE_RUNS:
            if run_name not in feature_runs:
                continue
            if run_name == "fourier" and not all(col in history_scope_df.columns for col in FOURIER_COLS):
                continue
            if run_name in {"weekly", "f_dummy"} and not any(
                col in history_scope_df.columns for col in FEATURE_RUNS[run_name]["extra_cols"]
            ):
                continue

            if scope_name == "division":
                history_scope_run_df = aggregate_division_weekly(
                    history_scope_df,
                    run_name,
                    target_col=target_col,
                    dataset_group_by=requested_dataset_group_by,
                )
                future_scope_run_df = aggregate_division_weekly(
                    future_scope_df,
                    run_name,
                    target_col=target_col,
                    dataset_group_by=requested_dataset_group_by,
                )
            else:
                history_scope_run_df = history_scope_df.copy()
                future_scope_run_df = future_scope_df.copy()

            grouping_keys = list(dict.fromkeys([scope_col, *requested_dataset_group_by]))
            future_scope_run_df = expand_future_rows_to_model_grain(
                history_df=history_scope_run_df,
                future_df=future_scope_run_df,
                grouping_keys=grouping_keys,
            )
            for entity_key, history_entity_df in history_scope_run_df.groupby(grouping_keys, dropna=False):
                if not isinstance(entity_key, tuple):
                    entity_key = (entity_key,)
                entity_mask = np.ones(len(future_scope_run_df), dtype=bool)
                for key_col, key_value in zip(grouping_keys, entity_key):
                    entity_mask &= future_scope_run_df[key_col].astype(str).to_numpy() == str(key_value)
                future_entity_df = future_scope_run_df[entity_mask].copy()
                if history_entity_df.empty or future_entity_df.empty:
                    continue

                entity_name = " | ".join(
                    f"{col}={value}" for col, value in zip(grouping_keys, entity_key)
                )

                for model_type in sorted(model_types):
                    try:
                        forecast_df, coefficients_df, training_summary_df, metadata = build_forecast_predictions_for_entity(
                            history_entity_df=history_entity_df,
                            future_entity_df=future_entity_df,
                            scope_name=scope_name,
                            entity_name=entity_name,
                            run_name=run_name,
                            model_type=model_type,
                            media_predictors=validated_media_predictors,
                            media_transform_config=validated_media_transform_config,
                            optional_features=validated_optional_features,
                            target_col=target_col,
                        )
                    except ForecastSkipError as exc:
                        display_forecast_skip_notice(
                            scope_name=scope_name,
                            entity_name=entity_name,
                            model_type=model_type,
                            run_name=run_name,
                            reason=str(exc),
                        )
                        forecast_outputs.append(
                            build_skipped_forecast_row(
                                scope_name=scope_name,
                                entity_name=entity_name,
                                model_type=model_type,
                                run_name=run_name,
                                reason=str(exc),
                            )
                        )
                        continue
                    forecast_outputs.append(forecast_df)
                    coefficient_outputs.append(coefficients_df)
                    training_summary_outputs.append(training_summary_df)

    non_empty_forecast_outputs = [frame for frame in forecast_outputs if not frame.empty]
    forecast_result = (
        pd.concat(non_empty_forecast_outputs, axis=0, ignore_index=True)
        if non_empty_forecast_outputs
        else pd.DataFrame()
    )
    if not forecast_result.empty:
        forecast_result = forecast_result.sort_values(
            ["Scope", "Key", "Feature_Run", "Model_Type", "ISO_Year", "ISO_Week"]
        ).reset_index(drop=True)
        forecast_column_order = [
            "Key",
            "Run_Status",
            "State",
            "ISO_Year",
            "ISO_Week",
            "Month",
            "DSP",
            "LeadGen",
            "Paid Search",
            "Paid Social",
            "Prescreen",
            "Referrals",
            "Sweepstakes",
            "Channel",
            "H_Tactic",
            "Detail_Tactic",
            "Product",
            "Predicted APPS Raw",
            "Predicted APPS",
            "95% Confidence Lower Limit",
            "95% Confidence Upper Limit",
            "Time_Index_Contribution",
            "Time_Index_Sq_Contribution",
            "Trend_Contribution",
            "Scope",
            "Model_Type",
            "Feature_Run",
        ]
        forecast_result = forecast_result.reindex(
            columns=[col for col in forecast_column_order if col in forecast_result.columns]
        )
        forecast_result = apply_product_allocation_to_forecast(forecast_result, product_factors_df)
        static_cols = [col for col in forecast_column_order if col in forecast_result.columns]
        dynamic_cols = [col for col in forecast_result.columns if col not in static_cols]
        forecast_result = forecast_result.reindex(columns=static_cols + sorted(dynamic_cols))
    # Compute marginal apps per $1K per tactic before monthly rollup
    # This is methodology-agnostic — calls the scorer with +$1K per tactic
    if not forecast_result.empty:
        try:
            _spend_for_marginals = load_future_data(
                future_input_path,
                target_col=target_col,
                dataset_group_by=requested_dataset_group_by,
            )
            _ols_coeffs = pd.concat(
                [f for f in coefficient_outputs if not f.empty],
                axis=0, ignore_index=True
            ) if [f for f in coefficient_outputs if not f.empty] else pd.DataFrame()

            if not _ols_coeffs.empty:
                forecast_result = compute_marginals(
                    _spend_for_marginals,
                    _ols_coeffs,
                    forecast_result,
                    model_type="OLS",
                    feature_run="weekly",
                )
        except Exception as _e:
            import warnings
            warnings.warn(f"Marginal computation skipped: {_e}")

    monthly_forecast_result = roll_up_weekly_forecast_to_monthly(forecast_result)
    non_empty_coefficient_outputs = [frame for frame in coefficient_outputs if not frame.empty]
    coefficients_result = (
        pd.concat(non_empty_coefficient_outputs, axis=0, ignore_index=True)
        if non_empty_coefficient_outputs
        else pd.DataFrame()
    )
    if not coefficients_result.empty:
        coefficients_result = coefficients_result.sort_values(
            ["Scope", "Key", "Feature_Run", "Model_Type"]
        ).reset_index(drop=True)
    non_empty_training_summary_outputs = [frame for frame in training_summary_outputs if not frame.empty]
    training_summary_result = (
        pd.concat(non_empty_training_summary_outputs, axis=0, ignore_index=True)
        if non_empty_training_summary_outputs
        else pd.DataFrame()
    )
    if not training_summary_result.empty:
        training_summary_result = training_summary_result.sort_values(
            ["Scope", "Key", "Feature_Run", "Model_Type"]
        ).reset_index(drop=True)

    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        file_suffix = group_by_suffix(requested_dataset_group_by)
        forecast_result = enrich_key_columns(forecast_result)
        monthly_forecast_result = enrich_key_columns(monthly_forecast_result)
        coefficients_result = enrich_key_columns(coefficients_result)
        training_summary_result = enrich_key_columns(training_summary_result)
        forecast_result.to_csv(output_root / f"future_forecast{file_suffix}.csv", index=False)
        monthly_forecast_result.to_csv(output_root / f"monthly_forecast{file_suffix}.csv", index=False)
        coefficients_result.to_csv(output_root / f"model_coefficients{file_suffix}.csv", index=False)
        training_summary_result.to_csv(output_root / f"model_training_summary{file_suffix}.csv", index=False)
        if not product_factors_df.empty:
            product_factors_df.to_csv(output_root / f"product_factors{file_suffix}.csv", index=False)
        _consolidated = consolidate_forecast_output_files(output_root)
        _coeff_cons = _consolidated.get("model_coefficients", pd.DataFrame())
        _pf_cons    = _consolidated.get("product_factors",    pd.DataFrame())
        if not _coeff_cons.empty and not _pf_cons.empty:
            _pf_slim = _pf_cons[["Key", "PRODUCT_FUNDED", "APPLICATION_SHARE", "APPROVAL_RATE", "ORIGINATION_RATE"]].copy()
            _joined  = _coeff_cons.merge(_pf_slim, on="Key", how="left")
            _joined.to_csv(output_root / "modelcoeff_and_prodfactors.csv", index=False)

    return {
        "future_forecast": forecast_result,
        "monthly_forecast": monthly_forecast_result,
        "model_coefficients": coefficients_result,
        "model_training_summary": training_summary_result,
    }


def create_spend_history_figure(
    entity_df: pd.DataFrame,
    scope_name: str,
    entity_name: str,
    target_columns: Sequence[str],
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a chart with spend on the primary axis and selected targets on the secondary axis."""
    plot_df = entity_df.sort_values([YEAR_COL, WEEK_COL]).copy()
    x_labels = plot_df[YEAR_COL].astype(str) + "-W" + plot_df[WEEK_COL].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(14, 7))
    for col in NON_DUMMY_PREDICTORS:
        if col in plot_df.columns:
            ax.plot(x_labels, plot_df[col].astype(float), label=col, linewidth=2)

    ax2 = ax.twinx()
    app_colors = {
        "APPLICATIONS": "#111111",
        "NON_DM_APPLICATIONS": "#8c564b",
    }
    for col in target_columns:
        if col in plot_df.columns:
            ax2.plot(
                x_labels,
                pd.to_numeric(plot_df[col], errors="coerce").astype(float),
                label=col,
                linewidth=2.5,
                linestyle="--",
                color=app_colors.get(col, None),
            )

    ax.set_title(f"Spend Over Time: {scope_name}={entity_name}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Spend")
    ax2.set_ylabel("Applications")
    ax.tick_params(axis="x", rotation=90)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    return fig, ax


def run_spend_history_plot(
    input_path: str,
    selected_states: Optional[Sequence[str]] = None,
    selected_divisions: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
    prediction_start_year: Optional[int] = None,
    prediction_start_week: Optional[int] = None,
    inline_output: bool = False,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Plot spend history for selected states or divisions up to a prediction cutoff.

    This is a standalone utility for visual inspection of the six spend tactics:
        - ``DSP``
        - ``LeadGen``
        - ``Paid Search``
        - ``Paid Social``
        - ``Prescreen``
        - ``Referrals``

    The same chart also overlays user-selected target series on a secondary Y axis for
    easier comparison against spend patterns.

    Parameters:
        input_path: Historical dataset path.
        selected_states: Optional list of states to plot.
        selected_divisions: Optional list of divisions to plot.
        target_columns: Optional target series to plot on the secondary axis. Choices:
            ``APPLICATIONS`` and ``NON_DM_APPLICATIONS``. If omitted, both are plotted.
        prediction_start_year: Optional forecast start year. If provided together with
            ``prediction_start_week``, the plot only includes data strictly before that point.
        prediction_start_week: Optional forecast start week.
        inline_output: If ``True``, display plots in Jupyter instead of writing files.
        output_dir: Optional folder for saved plots when ``inline_output`` is ``False``.

    Returns:
        A manifest DataFrame listing the scope, entity, last plotted week, and saved path
        when files are written.
    """
    validated_target_columns = validate_target_plot_columns(target_columns)
    loader_target = validated_target_columns[0] if validated_target_columns else TARGET_COL
    df = load_data(input_path, target_col=loader_target)
    if (prediction_start_year is None) ^ (prediction_start_week is None):
        raise ValueError("Provide both prediction_start_year and prediction_start_week, or neither.")

    if prediction_start_year is not None and prediction_start_week is not None:
        cutoff_mask = (
            (df[YEAR_COL] < int(prediction_start_year))
            | (
                (df[YEAR_COL] == int(prediction_start_year))
                & (df[WEEK_COL] < int(prediction_start_week))
            )
        )
        df = df[cutoff_mask].copy()

    manifests: List[Dict[str, Any]] = []
    scopes = [
        (STATE_COL, "state", selected_states, False),
        (DIVISION_COL, "division", selected_divisions, True),
    ]

    for scope_col, scope_name, selected_entities, aggregate_division in scopes:
        scope_df = prepare_entity_subset(df, scope_col, selected_entities)
        if scope_df.empty:
            continue

        if aggregate_division:
            group_cols = [DIVISION_COL, YEAR_COL, WEEK_COL]
            aggregation_map = {col: "sum" for col in NON_DUMMY_PREDICTORS}
            for col in validated_target_columns:
                if col in scope_df.columns:
                    aggregation_map[col] = "sum"
            scope_df = (
                scope_df[group_cols + list(aggregation_map.keys())]
                .groupby(group_cols, as_index=False)
                .agg(aggregation_map)
                .sort_values([DIVISION_COL, YEAR_COL, WEEK_COL])
                .reset_index(drop=True)
            )

        for entity, entity_df in scope_df.groupby(scope_col, dropna=False):
            if entity_df.empty:
                continue

            fig, _ = create_spend_history_figure(entity_df, scope_name, str(entity), validated_target_columns)
            saved_path = None

            if inline_output:
                try:
                    from IPython.display import display

                    display(fig)
                finally:
                    plt.close(fig)
            else:
                if output_dir is None:
                    output_root = Path("spend_history_plots")
                else:
                    output_root = Path(output_dir)
                output_root.mkdir(parents=True, exist_ok=True)
                save_path = output_root / scope_name / f"{safe_name(entity)}_spend_history.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=150)
                saved_path = str(save_path)
                plt.close(fig)

            last_row = entity_df.sort_values([YEAR_COL, WEEK_COL]).iloc[-1]
            manifests.append(
                {
                    "scope": scope_name,
                    "entity": str(entity),
                    "last_plotted_year": int(last_row[YEAR_COL]),
                    "last_plotted_week": int(last_row[WEEK_COL]),
                    "saved_path": saved_path,
                }
            )

    return pd.DataFrame(manifests)


def main() -> None:
    args = parse_args()
    run_model_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        target_col=args.target_col,
        dataset_group_by=args.dataset_group_by,
        selected_states=args.selected_states,
        selected_divisions=args.selected_divisions,
        methodologies=args.methodologies,
        media_predictors=args.media_predictors,
        optional_features=args.optional_features,
        inline_output=args.inline_output,
        backtest_mode=args.backtest_mode,
        fixed_window_weeks=args.fixed_window_weeks,
    )


if __name__ == "__main__":
    main()
