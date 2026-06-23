"""
Shared initialisation logic for Oracle — imported by both the home page and
Scenario Runs so that data loading and predictions are pre-computed the moment
the app starts, before the user navigates to any page.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.client import Config as _BotoConfig

sys.path.insert(0, str(Path(__file__).parent))
from build_state_division_models import roll_up_weekly_forecast_to_monthly, spread_monthly_spend_to_weekly

# ── Config & constants ────────────────────────────────────────────────────────
_FALLBACK_MEDIA_PREDICTORS = [
    "DSP", "LeadGen", "Paid Search", "Paid Social", "Prescreen", "Referrals",
]
_EXTRA_SPEND_TACTICS = ["Sweepstakes"]


def _load_config() -> dict:
    config_path = Path(__file__).parent / "model_config.json"
    try:
        with open(config_path) as _f:
            return json.load(_f)
    except Exception:
        return {}

_CONFIG = _load_config()


def _load_media_predictors() -> list[str]:
    predictors = _CONFIG.get("media_predictors", [])
    return predictors if predictors else list(_FALLBACK_MEDIA_PREDICTORS)

_MEDIA_PREDICTORS = _load_media_predictors()

SPEND_COLUMNS = [f"{t} ($)" for t in _MEDIA_PREDICTORS] + [f"{t} ($)" for t in _EXTRA_SPEND_TACTICS]

TACTIC_MAP = {
    f"{t} ($)": (t, f"{t.replace(' ', '_')}_contrib")
    for t in _MEDIA_PREDICTORS + _EXTRA_SPEND_TACTICS
}

_COL_TO_TACTIC = {col: names[0] for col, names in TACTIC_MAP.items()}
_TACTIC_TO_COL = {v: k for k, v in _COL_TO_TACTIC.items()}

_PF_COLS = ["PRODUCT_FUNDED", "APPLICATION_SHARE"]

_SCENARIO_NAMES = ["Baseline", "Scenario 1", "Scenario 2", "Scenario 3"]

_UPLOAD_ALIASES: dict[str, str] = {"date": "Date", "state": "State", "state_cd": "State"}
for _col in SPEND_COLUMNS:
    _tactic = _col[: -len(" ($)")]
    _UPLOAD_ALIASES[_col.lower()] = _col
    _UPLOAD_ALIASES[_tactic.lower()] = _col
if "LeadGen ($)" in SPEND_COLUMNS:
    _UPLOAD_ALIASES["lead gen"] = "LeadGen ($)"
    _UPLOAD_ALIASES["lead gen ($)"] = "LeadGen ($)"

REQUIRED_COLS = ["Date", "State"] + SPEND_COLUMNS


# ── Data helpers ──────────────────────────────────────────────────────────────
def build_tactic_map(coeff_df: pd.DataFrame) -> dict:
    _NON_TACTIC_PREFIXES = (
        "W_", "F_", "time_index", "year_indicator",
        "sin_", "cos_",
        "Prescreen_lag", "DSP_lag", "Paid_Search_lag",
        "DSP_trailing", "Paid_Search_trailing", "Prescreen_trailing",
        "APPLICATIONS_", "NON_DM_APPLICATIONS_",
    )
    modelled = sorted(
        col[: -len("__MinMax_Min")]
        for col in coeff_df.columns
        if col.endswith("__MinMax_Min")
        and not any(col.startswith(p) for p in _NON_TACTIC_PREFIXES)
    )
    tactics = list(modelled)
    for extra in _EXTRA_SPEND_TACTICS:
        if extra not in tactics:
            tactics.append(extra)
    return {
        f"{t} ($)": (t, f"{t.replace(' ', '_')}_contrib")
        for t in tactics
    }


def load_coeff_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop(columns=[c for c in _PF_COLS if c in df.columns])
        .drop_duplicates(subset=["Key"])
        .reset_index(drop=True)
    )


def load_product_factors(df: pd.DataFrame) -> pd.DataFrame:
    if "PRODUCT_FUNDED" not in df.columns or "APPLICATION_SHARE" not in df.columns:
        return pd.DataFrame()
    return (
        df[["Key", "PRODUCT_FUNDED", "APPLICATION_SHARE"]]
        .dropna(subset=["PRODUCT_FUNDED"])
        .drop_duplicates(subset=["Key", "PRODUCT_FUNDED"])
        .reset_index(drop=True)
    )


def normalise_upload(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns={c: _UPLOAD_ALIASES.get(c.lower().strip(), c) for c in raw.columns})
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    out = raw[REQUIRED_COLS].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    out["State"] = out["State"].astype(str).str.strip().str.upper()
    for col in SPEND_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


# ── Spaces helpers ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_spaces_client():
    key    = os.environ.get("SPACES_KEY", "")
    secret = os.environ.get("SPACES_SECRET", "")
    region = os.environ.get("SPACES_REGION", "lon1").lower().strip()
    bucket = os.environ.get("SPACES_BUCKET", "")
    if not (key and secret and bucket):
        return None, ""
    client = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=_BotoConfig(signature_version="s3v4"),
    )
    return client, bucket


def load_df_from_spaces(
    file_env_var: str,
    default_filename: str,
    excel_sheet: str | None = None,
) -> tuple[pd.DataFrame | None, str]:
    client, bucket = get_spaces_client()
    if client is None:
        key    = os.environ.get("SPACES_KEY", "")
        secret = os.environ.get("SPACES_SECRET", "")
        bkt    = os.environ.get("SPACES_BUCKET", "")
        missing = [n for n, v in [("SPACES_KEY", key), ("SPACES_SECRET", secret), ("SPACES_BUCKET", bkt)] if not v]
        return None, f"Missing env vars: {', '.join(missing)}"
    filename = os.environ.get(file_env_var, default_filename)
    try:
        obj  = client.get_object(Bucket=bucket, Key=filename)
        data = obj["Body"].read()
        if filename.lower().endswith((".xlsx", ".xls")):
            xl    = pd.ExcelFile(BytesIO(data))
            sheet = (
                excel_sheet
                if excel_sheet and excel_sheet in xl.sheet_names
                else xl.sheet_names[0]
            )
            return xl.parse(sheet), ""
        return pd.read_csv(BytesIO(data)), ""
    except Exception as e:
        return None, f"{filename}: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_model() -> tuple[pd.DataFrame | None, str]:
    return load_df_from_spaces("SPACES_MODEL_FILE", "modelcoeff_and_prodfactors.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_spend() -> tuple[pd.DataFrame | None, str]:
    return load_df_from_spaces("SPACES_SPEND_FILE", "FutureSpend.csv")


# ── Prediction engine ──────────────────────────────────────────────────────────
def monthly_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    long = (
        df.rename(columns={"Date": "BUSINESS_DATE", "State": "STATE_CD"})
        .melt(
            id_vars=["BUSINESS_DATE", "STATE_CD"],
            value_vars=SPEND_COLUMNS,
            var_name="_col",
            value_name="TOTAL_COST",
        )
    )
    long["DETAIL_TACTIC"] = long["_col"].map(_COL_TO_TACTIC)
    long = long.drop(columns=["_col"])
    long["TOTAL_COST"] = pd.to_numeric(long["TOTAL_COST"], errors="coerce").fillna(0.0)
    long["BUSINESS_DATE"] = pd.to_datetime(long["BUSINESS_DATE"])
    weekly = spread_monthly_spend_to_weekly(long, monthly_tactics=["Prescreen"])
    weekly = weekly.dropna(subset=["ISO_WEEK"])
    weekly["ISO_WEEK"] = weekly["ISO_WEEK"].astype(int)
    weekly["ISO_YEAR"] = weekly["ISO_YEAR"].astype(int)
    weekly["_col"] = weekly["DETAIL_TACTIC"].map(_TACTIC_TO_COL)
    weekly = weekly.dropna(subset=["_col"])
    wide = (
        weekly.groupby(["STATE_CD", "ISO_YEAR", "ISO_WEEK", "_col"])["TOTAL_COST"]
        .sum()
        .unstack("_col")
        .reset_index()
    )
    for col in SPEND_COLUMNS:
        if col not in wide.columns:
            wide[col] = 0.0
    wide[SPEND_COLUMNS] = wide[SPEND_COLUMNS].fillna(0.0)
    wide["Date"] = wide.apply(
        lambda r: date.fromisocalendar(int(r["ISO_YEAR"]), int(r["ISO_WEEK"]), 1),
        axis=1,
    )
    wide = wide.rename(columns={"STATE_CD": "State"})
    return wide[["Date", "State"] + SPEND_COLUMNS]


def parse_key(key: str) -> dict:
    result: dict = {}
    for seg in str(key).split("|"):
        seg = seg.strip()
        if "=" in seg:
            col, _, val = seg.partition("=")
            result[col.strip()] = val.strip()
    return result


def grain_level(parsed: dict) -> int:
    if "DETAIL_TACTIC" in parsed: return 3
    if "H_TACTIC"      in parsed: return 2
    if "CHANNEL_CD"    in parsed: return 1
    return 0


def score_coeff_row(coeff: pd.Series, spend_row: pd.Series, iso_year: int, iso_week: int) -> dict:
    def scale(val: float, col_name: str) -> float:
        mn  = coeff.get(f"{col_name}__MinMax_Min",   0)
        rng = coeff.get(f"{col_name}__MinMax_Range", 1)
        mn  = 0.0 if pd.isna(mn)  else float(mn)
        rng = 1.0 if pd.isna(rng) else float(rng)
        return 0.0 if rng == 0 else (val - mn) / rng

    intercept  = float(coeff.get("Intercept", 0) or 0)
    prediction = intercept
    contrib: dict = {}

    for input_col, (coeff_col, contrib_key) in TACTIC_MAP.items():
        raw_val = float(spend_row.get(input_col, 0) or 0)
        c_raw   = coeff.get(coeff_col, np.nan)
        if pd.isna(c_raw):
            contrib[contrib_key] = 0.0
            continue
        c            = float(c_raw)
        contribution = c * scale(raw_val, coeff_col)
        contrib[contrib_key] = round(contribution, 6)
        prediction  += contribution

    _start_year   = _CONFIG.get("training_start_year", 2024)
    time_index    = (iso_year - _start_year) * 52 + iso_week + 1
    time_index_sq = time_index ** 2

    ti_c_raw   = coeff.get("time_index", np.nan)
    ti_contrib = 0.0
    if not pd.isna(ti_c_raw):
        ti_contrib = float(ti_c_raw) * scale(time_index, "time_index")
    prediction += ti_contrib

    ti_sq_c_raw   = coeff.get("time_index_sq", np.nan)
    ti_sq_contrib = 0.0
    if not pd.isna(ti_sq_c_raw):
        ti_sq_contrib = float(ti_sq_c_raw) * scale(time_index_sq, "time_index_sq")
    prediction += ti_sq_contrib

    w_contrib  = float(coeff.get(f"W_{iso_week}", 0) or 0) if iso_week > 1 else 0.0
    prediction += w_contrib

    if np.isnan(prediction):
        prediction = 0.0

    sigma    = float(coeff.get("Sigma", 0) or 0)
    lower_ci = max(0.0, prediction - 1.96 * sigma)
    upper_ci = prediction + 1.96 * sigma

    return {
        "Predicted APPS":             max(0, int(round(prediction))),
        "Predicted APPS Raw":         max(0.0, round(prediction, 6)),
        "95% Confidence Lower Limit": int(round(lower_ci)),
        "95% Confidence Upper Limit": int(round(upper_ci)),
        "time_index":                 time_index,
        "time_index_sq":              time_index_sq,
        **contrib,
        "time_index_contrib":         round(ti_contrib,    6),
        "time_index_sq_contrib":      round(ti_sq_contrib, 6),
        "weekly_dummy_contrib":       round(w_contrib,     6),
        "Intercept":                  round(intercept,     6),
        "Sigma":                      round(sigma,         6),
    }


def run_predictions(input_df: pd.DataFrame, coeff_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    df = input_df.copy()
    df["Date"]     = pd.to_datetime(df["Date"])
    df["ISO_YEAR"] = df["Date"].apply(lambda d: d.isocalendar()[0])
    df["ISO_WEEK"] = df["Date"].apply(lambda d: d.isocalendar()[1])
    df["Month"]    = df["Date"].dt.month

    spend_cols = list(SPEND_COLUMNS)
    grouped = (
        df.groupby(["State", "ISO_YEAR", "ISO_WEEK"], as_index=False)
        .agg({**{c: "sum" for c in spend_cols}, "Month": "first"})
    )

    for _, row in grouped.iterrows():
        state    = str(row["State"])
        iso_year = int(row["ISO_YEAR"])
        iso_week = int(row["ISO_WEEK"])
        month    = int(row["Month"])

        state_coeffs = coeff_df[
            coeff_df["Key"].astype(str).str.startswith(f"STATE_CD={state}")
        ]

        if state_coeffs.empty:
            results.append({
                "State": state, "ISO_Year": iso_year, "ISO_Week": iso_week, "Month": month,
                **{c: row[c] for c in spend_cols},
                "Channel": None, "H_Tactic": None, "Detail_Tactic": None, "Product": None,
                "Predicted APPS": None, "Predicted APPS Raw": None,
                "95% Confidence Lower Limit": None, "95% Confidence Upper Limit": None,
                "Run_Status": "SKIPPED",
                "_grain": -1, "_ch": "", "_ht": "", "_dt": "",
                "Model_Key": f"STATE_CD={state}", "Model_Status": "No coefficient found",
            })
            continue

        for _, coeff in state_coeffs.iterrows():
            key    = str(coeff["Key"])
            parsed = parse_key(key)
            if parsed.get("STATE_CD", "") != state:
                continue

            grain         = grain_level(parsed)
            channel       = parsed.get("CHANNEL_CD",    None)
            h_tactic      = parsed.get("H_TACTIC",      None)
            detail_tactic = parsed.get("DETAIL_TACTIC", None)
            scored        = score_coeff_row(coeff, row, iso_year, iso_week)

            results.append({
                "State": state, "ISO_Year": iso_year, "ISO_Week": iso_week, "Month": month,
                **{c: row[c] for c in spend_cols},
                "Channel": channel, "H_Tactic": h_tactic,
                "Detail_Tactic": detail_tactic, "Product": None,
                **scored,
                "Run_Status": "SUCCESS",
                "_grain": grain,
                "_ch": channel       or "",
                "_ht": h_tactic      or "",
                "_dt": detail_tactic or "",
                "Model_Key": key, "Model_Status": "OK",
            })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out.sort_values(
        ["State", "ISO_Year", "ISO_Week", "_grain", "_ch", "_ht", "_dt"],
        ascending=True, na_position="first",
    ).drop(columns=["_grain", "_ch", "_ht", "_dt"]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def run_scenario(spend_df: pd.DataFrame, coeff_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a spend DataFrame. Returns (results_df, monthly_df). Cached at process level."""
    weekly_df  = monthly_to_weekly(spend_df.copy())
    results_df = run_predictions(weekly_df, coeff_df)

    # Zero out predictions for any State×Week where all spend is 0.
    # Applies only to the user's run — the internal zero-spend baseline below is intentionally untouched.
    _zero_spend_mask = weekly_df[SPEND_COLUMNS].sum(axis=1) == 0
    if _zero_spend_mask.any():
        _zero_keys = (
            weekly_df[_zero_spend_mask][["State", "ISO_YEAR", "ISO_WEEK"]]
            .rename(columns={"ISO_YEAR": "ISO_Year", "ISO_WEEK": "ISO_Week"})
            .drop_duplicates()
        )
        _zero_idx = results_df.merge(
            _zero_keys, on=["State", "ISO_Year", "ISO_Week"], how="inner"
        ).index
        results_df.loc[_zero_idx, ["Predicted APPS", "Predicted APPS Raw"]] = 0

    zero_df = weekly_df.copy()
    for _c in SPEND_COLUMNS:
        zero_df[_c] = 0.0
    baseline_df = run_predictions(zero_df, coeff_df)

    baseline_lookup = (
        baseline_df[["State", "ISO_Year", "ISO_Week", "Model_Key", "Predicted APPS"]]
        .rename(columns={"Predicted APPS": "Baseline APPS"})
    )
    results_df = results_df.merge(
        baseline_lookup, on=["State", "ISO_Year", "ISO_Week", "Model_Key"], how="left"
    )
    results_df["Baseline APPS"] = results_df[["Predicted APPS", "Baseline APPS"]].min(axis=1)
    results_df["Incremental APPS"] = (
        results_df["Predicted APPS"] - results_df["Baseline APPS"].fillna(0)
    ).clip(lower=0).round().astype("Int64")

    if "APPROVAL_RATE" in coeff_df.columns and "ORIGINATION_RATE" in coeff_df.columns:
        _key_rates  = coeff_df[["Key", "APPROVAL_RATE", "ORIGINATION_RATE"]].copy()
        _rates      = results_df[["Model_Key"]].merge(
            _key_rates, left_on="Model_Key", right_on="Key", how="left"
        )
        _raw_clipped = results_df["Predicted APPS Raw"].clip(lower=0)
        results_df["APPROVAL_Total"]     = (_raw_clipped * _rates["APPROVAL_RATE"].fillna(0).values).fillna(0)
        results_df["ORIGINATIONS_Total"] = (_raw_clipped * _rates["ORIGINATION_RATE"].fillna(0).values).fillna(0)

    _rollup_input        = results_df.copy()
    _rollup_input["Key"] = _rollup_input["Model_Key"]
    _monthly_pred        = roll_up_weekly_forecast_to_monthly(_rollup_input)

    _baseline_rollup        = baseline_df.drop(columns=["Predicted APPS Raw"], errors="ignore").copy()
    _baseline_rollup["Key"] = _baseline_rollup["Model_Key"]
    _monthly_base           = roll_up_weekly_forecast_to_monthly(_baseline_rollup)

    if not _monthly_pred.empty and not _monthly_base.empty:
        _merge_keys = ["Key", "State", "Calendar_Year", "Calendar_Month", "Channel", "H_Tactic", "Detail_Tactic"]
        _merge_keys = [k for k in _merge_keys if k in _monthly_pred.columns and k in _monthly_base.columns]
        _base_slim  = (
            _monthly_base[_merge_keys + ["Allocated_Predicted_APPS"]]
            .rename(columns={"Allocated_Predicted_APPS": "_Baseline_raw"})
        )
        _monthly_pred = _monthly_pred.merge(_base_slim, on=_merge_keys, how="left")
        _monthly_pred["Baseline APPS"]     = _monthly_pred[["Allocated_Predicted_APPS", "_Baseline_raw"]].min(axis=1).clip(lower=0)
        _monthly_pred["Incremental APPS"]  = (
            _monthly_pred["Allocated_Predicted_APPS"] - _monthly_pred["Baseline APPS"].fillna(0)
        ).clip(lower=0)
        _monthly_pred["Baseline_APPS_Rounded"]    = _monthly_pred["Baseline APPS"].round().astype("Int64")
        _monthly_pred["Incremental_APPS_Rounded"] = _monthly_pred["Incremental APPS"].round().astype("Int64")
        _pred_denom = _monthly_pred["Allocated_Predicted_APPS"].replace(0, np.nan)
        if "Allocated_Approved" in _monthly_pred.columns:
            _appr_rate = (_monthly_pred["Allocated_Approved"] / _pred_denom).fillna(0)
            _monthly_pred["Baseline_Approved_Rounded"]    = (_monthly_pred["Baseline APPS"]    * _appr_rate).round().astype("Int64")
            _monthly_pred["Incremental_Approved_Rounded"] = (_monthly_pred["Incremental APPS"] * _appr_rate).round().astype("Int64")
        if "Allocated_Originations" in _monthly_pred.columns:
            _orig_rate = (_monthly_pred["Allocated_Originations"] / _pred_denom).fillna(0)
            _monthly_pred["Baseline_Originations_Rounded"]    = (_monthly_pred["Baseline APPS"]    * _orig_rate).round().astype("Int64")
            _monthly_pred["Incremental_Originations_Rounded"] = (_monthly_pred["Incremental APPS"] * _orig_rate).round().astype("Int64")
        _monthly_pred = _monthly_pred.drop(columns=["_Baseline_raw"])

    return results_df, _monthly_pred


# ── Session state ──────────────────────────────────────────────────────────────
def blank_scenario(name: str) -> dict:
    return {
        "name":            name,
        "id":              name,
        "upload_df":       None,
        "results_df":      None,
        "monthly_df":      None,
        "display_df":      None,
        "input_snap":      None,
        "upload_version":  0,
        "last_input_name": None,
        "spend_source":    None,
    }


def init_session_state() -> None:
    if "coeff_df"           not in st.session_state: st.session_state.coeff_df           = None
    if "coeff_source"       not in st.session_state: st.session_state.coeff_source        = None
    if "product_factors_df" not in st.session_state: st.session_state.product_factors_df  = None
    if "spaces_errors"      not in st.session_state: st.session_state.spaces_errors       = {}
    if "scenarios"          not in st.session_state:
        st.session_state.scenarios = [blank_scenario(n) for n in _SCENARIO_NAMES]
    else:
        for _sc, _default_id in zip(st.session_state.scenarios, _SCENARIO_NAMES):
            if "id" not in _sc:
                _sc["id"] = _default_id


def load_from_spaces() -> None:
    """Load model and spend from Spaces into session state. No-op if Spaces not configured."""
    if st.session_state.coeff_df is None:
        _model_raw, _err = cached_load_model()
        if _model_raw is not None:
            st.session_state.coeff_df           = load_coeff_df(_model_raw)
            st.session_state.product_factors_df = load_product_factors(_model_raw)
            st.session_state.coeff_source       = "spaces"
            st.session_state.spaces_errors.pop("model", None)
        elif _err:
            st.session_state.spaces_errors["model"] = _err

    _sc0 = st.session_state.scenarios[0]
    if _sc0["upload_df"] is None:
        _spend_raw, _err = cached_load_spend()
        if _spend_raw is not None:
            try:
                _sc0["upload_df"]      = normalise_upload(_spend_raw)
                _sc0["spend_source"]   = "spaces"
                _sc0["upload_version"] += 1
                st.session_state.spaces_errors.pop("spend", None)
                for _other in st.session_state.scenarios[1:]:
                    if _other["upload_df"] is None:
                        _other["upload_df"]      = _sc0["upload_df"].copy()
                        _other["upload_version"] += 1
            except Exception as e:
                st.session_state.spaces_errors["spend"] = f"FutureSpend.csv normalise failed: {e}"
        elif _err:
            st.session_state.spaces_errors["spend"] = _err


def prewarm_predictions() -> None:
    """Run predictions for every scenario that has spend data but no results yet.
    With @st.cache_data on run_scenario, only the first call computes — all
    subsequent sessions (and identical scenarios) get instant cache hits."""
    if st.session_state.coeff_df is None:
        return
    for _sc in st.session_state.scenarios:
        if _sc["upload_df"] is not None and _sc["results_df"] is None:
            try:
                _rows = _sc["upload_df"].dropna(subset=["Date", "State"])
                _rows = _rows[_rows["State"].astype(str).str.strip() != ""]
                if not _rows.empty:
                    _res, _mon          = run_scenario(_rows, st.session_state.coeff_df)
                    _sc["results_df"]   = _res
                    _sc["input_snap"]   = _rows.copy()
                    _sc["monthly_df"]   = _mon
            except Exception:
                pass
