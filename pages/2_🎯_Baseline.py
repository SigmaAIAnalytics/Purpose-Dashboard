from __future__ import annotations

import calendar as _calendar
import json
import os
import sys
import uuid
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
import numpy as np
import pandas as pd
import streamlit as st

pd.set_option("styler.render.max_elements", 1_000_000)
from botocore.client import Config as _BotoConfig
from build_state_division_models import roll_up_weekly_forecast_to_monthly, spread_monthly_spend_to_weekly

st.set_page_config(
    page_title="Baseline — Oracle",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    /* ══ FONTS ══════════════════════════════════════════════════════════════ */
    html, body, .stApp, .stMarkdown, .stText,
    p, li, td, th, input, textarea, select {
        font-family: 'DM Sans', sans-serif !important;
    }
    h1, h2, h3, .section-header {
        font-family: 'DM Serif Display', serif !important;
    }

    /* ══ GLOBAL — let Streamlit's own theme control all base text/bg ════════ */
    .stApp { background-color: var(--background-color); }
    .block-container { padding-top: 2rem; }

    /* Theme-aware text for common elements */
    p, li, td, th,
    .stMarkdown p, .stMarkdown li,
    [data-testid="stWidgetLabel"] p {
        color: var(--text-color) !important;
    }

    /* ══ SIDEBAR ════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: var(--text-color) !important;
    }

    /* ══ SECTION HEADERS ════════════════════════════════════════════════════ */
    .section-header {
        font-size: 1.35rem;
        color: var(--text-color) !important;
        margin-bottom: 0.25rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(148, 163, 184, 0.4);
    }

    /* ══ NOTE / INFO BOX ════════════════════════════════════════════════════ */
    .note-box {
        background: rgba(59, 130, 246, 0.13) !important;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: var(--text-color) !important;
        margin-bottom: 0.75rem;
    }
    .note-box strong { color: var(--text-color) !important; }

    /* ══ METRIC CARDS — intentionally dark on both themes ══════════════════ */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
    }
    /* force ALL child text white so dark card never inherits light-theme black */
    div[data-testid="metric-container"],
    div[data-testid="metric-container"] *,
    div[data-testid="metric-container"] p,
    div[data-testid="metric-container"] span,
    div[data-testid="metric-container"] div,
    div[data-testid="metric-container"] label {
        color: #cbd5e1 !important;
    }
    div[data-testid="metric-container"] [data-testid="metric-label"] p,
    div[data-testid="metric-container"] [data-testid="metric-label"] div {
        color: #94a3b8 !important;
        font-size: 0.73rem !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: #38bdf8 !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #64748b !important;
        font-size: 0.76rem !important;
    }
    div[data-testid="metric-container"] [data-testid="metric-delta"] svg {
        display: none;
    }

    /* ══ RUN BUTTON — gradient, always white text ═══════════════════════════ */
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.84 !important; }
    .stButton > button:focus { outline: none; box-shadow: 0 0 0 3px rgba(99,102,241,0.35); }

    /* ══ DOWNLOAD BUTTONS — theme-aware ════════════════════════════════════ */
    .stDownloadButton > button {
        background: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        border-radius: 8px !important;
        font-size: 0.875rem !important;
        transition: opacity 0.2s;
    }
    .stDownloadButton > button:hover { opacity: 0.72 !important; }

    /* ══ EXPANDER ════════════════════════════════════════════════════════════ */
    details summary,
    details summary p,
    .streamlit-expanderHeader p {
        color: var(--text-color) !important;
    }

    /* ══ DATAFRAME / TABLE ══════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] * {
        color: var(--text-color) !important;
    }

    /* ══ APPROVAL RATE OVERRIDE — small checkbox label ══════════════════════ */
    div[data-testid="metric-container"] ~ div [data-testid="stCheckbox"] label p {
        font-size: 0.72rem !important;
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_OPTIONS = list(
    dict.fromkeys(
        [
            "AL", "CA", "CO", "DE", "FL", "IA", "ID", "IN", "KS", "KY",
            "LA", "MI", "MO", "MS", "NV", "OH", "OK", "RI", "SC",
            "TN", "TX", "UT", "WI", "WY",
        ]
    )
)

SPEND_COLUMNS = [
    "DSP ($)",
    "LeadGen ($)",
    "Paid Search ($)",
    "Paid Social ($)",
    "Prescreen ($)",
    "Referrals ($)",
    "Sweepstakes ($)",
]

TACTIC_MAP = {
    "DSP ($)":          ("DSP",          "DSP_contrib"),
    "LeadGen ($)":      ("LeadGen",      "LeadGen_contrib"),
    "Paid Search ($)":  ("Paid Search",  "Paid_Search_contrib"),
    "Paid Social ($)":  ("Paid Social",  "Paid_Social_contrib"),
    "Prescreen ($)":    ("Prescreen",    "Prescreen_contrib"),
    "Referrals ($)":    ("Referrals",    "Referrals_contrib"),
    "Sweepstakes ($)":  ("Sweepstakes",  "Sweepstakes_contrib"),
}

_COL_TO_TACTIC = {col: names[0] for col, names in TACTIC_MAP.items()}
_TACTIC_TO_COL = {v: k for k, v in _COL_TO_TACTIC.items()}

_PF_COLS = ["PRODUCT_FUNDED", "APPLICATION_SHARE"]


def _load_coeff_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the coefficient rows from the model file, dropping product factor columns."""
    return (
        df.drop(columns=[c for c in _PF_COLS if c in df.columns])
        .drop_duplicates(subset=["Key"])
        .reset_index(drop=True)
    )


def _load_product_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Key → PRODUCT_FUNDED + APPLICATION_SHARE lookup from the model file."""
    if "PRODUCT_FUNDED" not in df.columns or "APPLICATION_SHARE" not in df.columns:
        return pd.DataFrame()
    return (
        df[["Key", "PRODUCT_FUNDED", "APPLICATION_SHARE"]]
        .dropna(subset=["PRODUCT_FUNDED"])
        .drop_duplicates(subset=["Key", "PRODUCT_FUNDED"])
        .reset_index(drop=True)
    )


# ── Upload column aliases & normaliser ───────────────────────────────────────
_UPLOAD_ALIASES: dict[str, str] = {
    "date":             "Date",
    "state":            "State",
    "state_cd":         "State",
    "dsp":              "DSP ($)",
    "dsp ($)":          "DSP ($)",
    "leadgen":          "LeadGen ($)",
    "leadgen ($)":      "LeadGen ($)",
    "lead gen":         "LeadGen ($)",
    "lead gen ($)":     "LeadGen ($)",
    "paid search":      "Paid Search ($)",
    "paid search ($)":  "Paid Search ($)",
    "paid social":      "Paid Social ($)",
    "paid social ($)":  "Paid Social ($)",
    "prescreen":        "Prescreen ($)",
    "prescreen ($)":    "Prescreen ($)",
    "referrals":        "Referrals ($)",
    "referrals ($)":    "Referrals ($)",
    "sweepstakes":      "Sweepstakes ($)",
    "sweepstakes ($)":  "Sweepstakes ($)",
}

_REQUIRED_COLS = ["Date", "State"] + SPEND_COLUMNS


def _normalise_upload(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using alias map, fill missing spend cols with 0, coerce types."""
    raw = raw.rename(columns={c: _UPLOAD_ALIASES.get(c.lower().strip(), c) for c in raw.columns})
    missing = [c for c in _REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    out = raw[_REQUIRED_COLS].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    out["State"] = out["State"].astype(str).str.strip().str.upper()
    for col in SPEND_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


# ── DigitalOcean Spaces helpers ───────────────────────────────────────────────
def _get_spaces_client():
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


def _load_df_from_spaces(
    file_env_var: str,
    default_filename: str,
    excel_sheet: str | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Fetch a CSV or Excel file from DO Spaces. Returns (df, error_message)."""
    client, bucket = _get_spaces_client()
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


# ── Comments helpers ──────────────────────────────────────────────────────────
def _comments_key() -> str:
    return os.environ.get("SPACES_COMMENTS_FILE", "comments.json")


@st.cache_data(ttl=60, show_spinner=False)
def _load_comments() -> list:
    client, bucket = _get_spaces_client()
    if client is None:
        return []
    try:
        obj = client.get_object(Bucket=bucket, Key=_comments_key())
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return []


def _save_comments(comments: list) -> tuple[bool, str]:
    client, bucket = _get_spaces_client()
    if client is None:
        return False, "Spaces client not configured"
    try:
        data = json.dumps(comments, indent=2, default=str).encode("utf-8")
        client.put_object(
            Bucket=bucket,
            Key=_comments_key(),
            Body=data,
            ContentType="application/json",
        )
        _load_comments.clear()
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Monthly → weekly spend conversion ────────────────────────────────────────
def _monthly_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Pro-rate wide monthly spend to weekly using day-count allocation."""
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


# ── Key parsing helpers ───────────────────────────────────────────────────────
def _parse_key(key: str) -> dict:
    """'STATE_CD=AL | CHANNEL_CD=DIGITAL | H_TACTIC=LSM'  →  dict"""
    result: dict = {}
    for seg in str(key).split("|"):
        seg = seg.strip()
        if "=" in seg:
            col, _, val = seg.partition("=")
            result[col.strip()] = val.strip()
    return result


def _grain_level(parsed: dict) -> int:
    """0 = state only, 1 = +channel, 2 = +H_tactic, 3 = +detail_tactic"""
    if "DETAIL_TACTIC" in parsed: return 3
    if "H_TACTIC"      in parsed: return 2
    if "CHANNEL_CD"    in parsed: return 1
    return 0


# ── Core scorer (one coefficient row) ────────────────────────────────────────
def _score_coeff_row(
    coeff: pd.Series,
    spend_row: pd.Series,
    iso_year: int,
    iso_week: int,
) -> dict:
    """
    Score a single coefficient row against spend inputs.
    Returns a dict with prediction, CI, contributions.
    Formula (matches Excel Output_Data exactly, validated ✅):
      Intercept + Σ(coef × MinMax(spend)) + time_index_contrib
      + time_index_sq_contrib + W_{week}_coef
    """
    def scale(val: float, col_name: str) -> float:
        mn  = coeff.get(f"{col_name}__MinMax_Min",   0)
        rng = coeff.get(f"{col_name}__MinMax_Range", 1)
        mn  = 0.0 if pd.isna(mn)  else float(mn)
        rng = 1.0 if pd.isna(rng) else float(rng)
        return 0.0 if rng == 0 else (val - mn) / rng

    intercept  = float(coeff.get("Intercept", 0) or 0)
    prediction = intercept
    contrib: dict = {}

    # Tactic contributions
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

    # time_index  (+1 offset confirmed against Output_Data: W9/2026 → 114)
    time_index    = (iso_year - 2024) * 52 + iso_week + 1
    time_index_sq = time_index ** 2

    ti_c_raw = coeff.get("time_index", np.nan)
    ti_contrib = 0.0
    if not pd.isna(ti_c_raw):
        ti_contrib = float(ti_c_raw) * scale(time_index, "time_index")
    prediction += ti_contrib

    ti_sq_c_raw = coeff.get("time_index_sq", np.nan)
    ti_sq_contrib = 0.0
    if not pd.isna(ti_sq_c_raw):
        ti_sq_contrib = float(ti_sq_c_raw) * scale(time_index_sq, "time_index_sq")
    prediction += ti_sq_contrib

    # Weekly dummy (W_1 is the baseline — coefficient is 0 by convention)
    w_contrib = float(coeff.get(f"W_{iso_week}", 0) or 0) if iso_week > 1 else 0.0
    prediction += w_contrib

    if np.isnan(prediction):
        prediction = 0.0

    sigma    = float(coeff.get("Sigma", 0) or 0)
    lower_ci = max(0.0, prediction - 1.96 * sigma)
    upper_ci = prediction + 1.96 * sigma

    return {
        "Predicted APPS":               max(0, int(round(prediction))),
        "Predicted APPS Raw":           round(prediction, 6),
        "95% Confidence Lower Limit":   int(round(lower_ci)),
        "95% Confidence Upper Limit":   int(round(upper_ci)),
        "time_index":                   time_index,
        "time_index_sq":                time_index_sq,
        **contrib,
        "time_index_contrib":           round(ti_contrib,    6),
        "time_index_sq_contrib":        round(ti_sq_contrib, 6),
        "weekly_dummy_contrib":         round(w_contrib,     6),
        "Intercept":                    round(intercept,     6),
        "Sigma":                        round(sigma,         6),
    }


# ── Main prediction engine ────────────────────────────────────────────────────
def run_predictions(input_df: pd.DataFrame, coeff_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (State, ISO Week) in the input, find ALL coefficient rows that
    match that state — at every grain level (state / channel / H_tactic /
    detail_tactic) — score each one, and return a hierarchical table that
    mirrors the Output_Data sheet format exactly.
    """
    results = []

    df = input_df.copy()
    df["Date"]    = pd.to_datetime(df["Date"])
    df["ISO_YEAR"] = df["Date"].apply(lambda d: d.isocalendar()[0])
    df["ISO_WEEK"] = df["Date"].apply(lambda d: d.isocalendar()[1])
    df["Month"]   = df["Date"].dt.month

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

        # All coefficient rows for this state (any grain)
        state_coeffs = coeff_df[
            coeff_df["Key"].astype(str).str.startswith(f"STATE_CD={state}")
        ]

        if state_coeffs.empty:
            results.append({
                "State": state, "ISO_Year": iso_year, "ISO_Week": iso_week,
                "Month": month,
                **{c: row[c] for c in spend_cols},
                "Channel": None, "H_Tactic": None,
                "Detail_Tactic": None, "Product": None,
                "Predicted APPS": None,
                "Predicted APPS Raw": None,
                "95% Confidence Lower Limit": None,
                "95% Confidence Upper Limit": None,
                "Run_Status": "SKIPPED",
                "_grain": -1, "_ch": "", "_ht": "", "_dt": "",
                "Model_Key": f"STATE_CD={state}",
                "Model_Status": "No coefficient found",
            })
            continue

        for _, coeff in state_coeffs.iterrows():
            key    = str(coeff["Key"])
            parsed = _parse_key(key)

            # Strict state match (avoids partial string collisions e.g. AL vs ALA)
            if parsed.get("STATE_CD", "") != state:
                continue

            grain        = _grain_level(parsed)
            channel      = parsed.get("CHANNEL_CD",    None)
            h_tactic     = parsed.get("H_TACTIC",      None)
            detail_tactic= parsed.get("DETAIL_TACTIC", None)

            scored = _score_coeff_row(coeff, row, iso_year, iso_week)

            results.append({
                "State": state, "ISO_Year": iso_year, "ISO_Week": iso_week,
                "Month": month,
                **{c: row[c] for c in spend_cols},
                "Channel":       channel,
                "H_Tactic":      h_tactic,
                "Detail_Tactic": detail_tactic,
                "Product":       None,
                **scored,
                "Run_Status": "SUCCESS",
                "_grain": grain,
                "_ch":    channel      or "",
                "_ht":    h_tactic     or "",
                "_dt":    detail_tactic or "",
                "Model_Key":    key,
                "Model_Status": "OK",
            })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out.sort_values(
        ["State", "ISO_Year", "ISO_Week", "_grain", "_ch", "_ht", "_dt"],
        ascending=True,
        na_position="first",
    ).drop(columns=["_grain", "_ch", "_ht", "_dt"]).reset_index(drop=True)
    return out


# ── Excel export helper ───────────────────────────────────────────────────────
def to_excel_bytes(results_df: pd.DataFrame, input_df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Predictions", index=False)
        input_df.to_excel(writer,   sheet_name="Input_Data",  index=False)
    return buf.getvalue()


# ── Scenario runner ───────────────────────────────────────────────────────────
def run_scenario(spend_df: pd.DataFrame, coeff_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a spend dataframe. Returns (results_df, monthly_df)."""
    weekly_df  = _monthly_to_weekly(spend_df.copy())
    results_df = run_predictions(weekly_df, coeff_df)

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
        _key_rates = coeff_df[["Key", "APPROVAL_RATE", "ORIGINATION_RATE"]].copy()
        _rates = results_df[["Model_Key"]].merge(
            _key_rates, left_on="Model_Key", right_on="Key", how="left"
        )
        _raw_clipped = results_df["Predicted APPS Raw"].clip(lower=0)
        results_df["APPROVAL_Total"]     = (_raw_clipped * _rates["APPROVAL_RATE"].fillna(0).values).fillna(0)
        results_df["ORIGINATIONS_Total"] = (_raw_clipped * _rates["ORIGINATION_RATE"].fillna(0).values).fillna(0)

    _rollup_input = results_df.copy()
    _rollup_input["Key"] = _rollup_input["Model_Key"]
    _monthly_pred = roll_up_weekly_forecast_to_monthly(_rollup_input)

    _baseline_rollup = baseline_df.drop(columns=["Predicted APPS Raw"], errors="ignore").copy()
    _baseline_rollup["Key"] = _baseline_rollup["Model_Key"]
    _monthly_base = roll_up_weekly_forecast_to_monthly(_baseline_rollup)

    if not _monthly_pred.empty and not _monthly_base.empty:
        _merge_keys = ["Key", "State", "Calendar_Year", "Calendar_Month", "Channel", "H_Tactic", "Detail_Tactic"]
        _merge_keys = [k for k in _merge_keys if k in _monthly_pred.columns and k in _monthly_base.columns]
        _base_slim = (
            _monthly_base[_merge_keys + ["Allocated_Predicted_APPS"]]
            .rename(columns={"Allocated_Predicted_APPS": "_Baseline_raw"})
        )
        _monthly_pred = _monthly_pred.merge(_base_slim, on=_merge_keys, how="left")
        _monthly_pred["Baseline APPS"] = _monthly_pred[["Allocated_Predicted_APPS", "_Baseline_raw"]].min(axis=1).clip(lower=0)
        _monthly_pred["Incremental APPS"] = (
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


# ── Shared display helpers ────────────────────────────────────────────────────
_MONTH_NAME = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

def _apply_grain_filter(df: pd.DataFrame, col: str, selection: list) -> pd.DataFrame:
    if not selection:
        return df
    mask = pd.Series(False, index=df.index)
    if "--Default--" in selection:
        mask |= df[col].isna()
    others = [v for v in selection if v != "--Default--"]
    if others:
        mask |= df[col].isin(others)
    return df[mask]


def _full_month_filter(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows belonging to months fully covered by the forecast."""
    if monthly_df is None or monthly_df.empty:
        return monthly_df
    _cov = (
        monthly_df
        .groupby(["State", "Calendar_Year", "Calendar_Month"], as_index=False)["Allocated_Days"]
        .max()
    )
    _cov["_days_in_month"] = _cov.apply(
        lambda r: _calendar.monthrange(int(r["Calendar_Year"]), int(r["Calendar_Month"]))[1],
        axis=1,
    )
    _full = _cov[_cov["Allocated_Days"] >= _cov["_days_in_month"]][
        ["State", "Calendar_Year", "Calendar_Month"]
    ]
    return monthly_df.merge(_full, on=["State", "Calendar_Year", "Calendar_Month"], how="inner")


# ── Session state init ────────────────────────────────────────────────────────
if "coeff_df"           not in st.session_state: st.session_state.coeff_df           = None
if "coeff_source"       not in st.session_state: st.session_state.coeff_source        = None
if "product_factors_df" not in st.session_state: st.session_state.product_factors_df  = None
if "spaces_errors"      not in st.session_state: st.session_state.spaces_errors       = {}

_SCENARIO_NAMES = ["Baseline", "Scenario 1", "Scenario 2", "Scenario 3"]

def _blank_scenario(name: str) -> dict:
    return {
        "name":            name,
        "upload_df":       None,
        "results_df":      None,
        "monthly_df":      None,
        "input_snap":      None,
        "upload_version":  0,
        "last_input_name": None,
        "spend_source":    None,
    }

if "scenarios" not in st.session_state:
    st.session_state.scenarios = [_blank_scenario(n) for n in _SCENARIO_NAMES]

# ── Auto-load from DO Spaces (runs once per session when no file is loaded) ───
if st.session_state.coeff_df is None:
    _spaces_model, _err = _load_df_from_spaces(
        "SPACES_MODEL_FILE", "modelcoeff_and_prodfactors.csv"
    )
    if _spaces_model is not None:
        st.session_state.coeff_df            = _load_coeff_df(_spaces_model)
        st.session_state.product_factors_df  = _load_product_factors(_spaces_model)
        st.session_state.coeff_source        = "spaces"
        st.session_state.spaces_errors.pop("model", None)
    elif _err:
        st.session_state.spaces_errors["model"] = _err

_sc0 = st.session_state.scenarios[0]
if _sc0["upload_df"] is None:
    _spaces_spend, _err = _load_df_from_spaces("SPACES_SPEND_FILE", "FutureSpend.csv")
    if _spaces_spend is not None:
        try:
            _sc0["upload_df"]      = _normalise_upload(_spaces_spend)
            _sc0["spend_source"]   = "spaces"
            _sc0["upload_version"] += 1
            st.session_state.spaces_errors.pop("spend", None)
        except Exception as e:
            st.session_state.spaces_errors["spend"] = f"FutureSpend.csv parsed but normalise failed: {e}"
    elif _err:
        st.session_state.spaces_errors["spend"] = _err


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Model file uploader
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    _m_loaded    = st.session_state.coeff_df is not None
    _m_key_count = (
        len(st.session_state.coeff_df["Key"].dropna())
        if _m_loaded and "Key" in st.session_state.coeff_df.columns else 0
    )
    _m_header = f"⚙️ Model — ✅ {_m_key_count} keys" if _m_loaded else "⚙️ Model File — ⚠️ not loaded"

    with st.expander(_m_header, expanded=not _m_loaded):
        if _m_loaded:
            _src_label = "Spaces" if st.session_state.coeff_source == "spaces" else "upload"
            st.caption(f"Source: {_src_label}. Upload below to override.")
        else:
            st.caption("Upload modelcoeff_and_prodfactors.csv from the model pipeline.")
        _model_file = st.file_uploader(
            "Model file",
            type=["csv"],
            key="model_uploader",
            label_visibility="collapsed",
        )
        if _model_file:
            try:
                _mf_raw   = pd.read_csv(_model_file)
                _mf_coeff = _load_coeff_df(_mf_raw)
                st.session_state.coeff_df           = _mf_coeff
                st.session_state.product_factors_df = _load_product_factors(_mf_raw)
                st.session_state.coeff_source       = "upload"
                st.success(f"✅ {len(_mf_coeff)} keys loaded")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("## 🗂 Scenarios")

    # ── Baseline status + file upload ─────────────────────────────────────────
    _sb_sc0 = st.session_state.scenarios[0]
    with st.expander("Baseline", expanded=False):
        if _sb_sc0["results_df"] is not None:
            _sb_total = (
                int(_sb_sc0["monthly_df"]["Allocated_Predicted_APPS_Rounded"].sum())
                if _sb_sc0["monthly_df"] is not None
                and "Allocated_Predicted_APPS_Rounded" in _sb_sc0["monthly_df"].columns
                else None
            )
            st.success(f"✅ Predicted — {_sb_total:,} APPS" if _sb_total is not None else "✅ Predictions ready")
        elif _sb_sc0["upload_df"] is not None:
            st.info(f"Spend loaded ({len(_sb_sc0['upload_df'])} rows) — edit table and run below")
        else:
            st.info("Upload a file or fill the table in the main area below")
        st.caption("Upload spend data (CSV or Excel) to populate the table:")
        _sb_input = st.file_uploader(
            "Spend file", type=["csv", "xlsx"],
            key="input_uploader", label_visibility="collapsed",
        )
        if _sb_input is not None and _sb_input.name != _sb_sc0["last_input_name"]:
            try:
                _sb_raw = (
                    pd.read_csv(_sb_input)
                    if _sb_input.name.endswith(".csv")
                    else pd.read_excel(_sb_input)
                )
                _sb_parsed = _normalise_upload(_sb_raw)
                _sb_sc0["upload_df"]       = _sb_parsed
                _sb_sc0["last_input_name"] = _sb_input.name
                _sb_sc0["upload_version"] += 1
                st.success(f"✅ {len(_sb_parsed)} rows loaded")
            except Exception as _sb_e:
                st.error(str(_sb_e))

    # ── Scenarios 1–3 ─────────────────────────────────────────────────────────
    for _si in range(1, 4):
        _sc = st.session_state.scenarios[_si]
        _name_key = f"sc_name_{_si}"
        if _name_key not in st.session_state:
            st.session_state[_name_key] = _sc["name"]

        _sc_label = st.session_state[_name_key] or f"Scenario {_si}"
        _sc_icon  = "✅ " if _sc["results_df"] is not None else ("📂 " if _sc["upload_df"] is not None else "")

        with st.expander(
            f"{_sc_icon}{_sc_label}",
            expanded=(_sc["upload_df"] is not None and _sc["results_df"] is None),
        ):
            _new_name = st.text_input("Name", key=_name_key, placeholder=f"Scenario {_si}")
            _sc["name"] = _new_name or f"Scenario {_si}"

            st.caption("Spend CSV or Excel (same format as baseline)")
            _sc_up = st.file_uploader(
                "Spend file",
                type=["csv", "xlsx"],
                key=f"sc_upload_{_si}",
                label_visibility="collapsed",
            )
            if _sc_up is not None and _sc_up.name != _sc["last_input_name"]:
                try:
                    _sc_raw = (
                        pd.read_csv(_sc_up)
                        if _sc_up.name.endswith(".csv")
                        else pd.read_excel(_sc_up)
                    )
                    _sc["upload_df"]       = _normalise_upload(_sc_raw)
                    _sc["last_input_name"] = _sc_up.name
                    _sc["upload_version"] += 1
                    st.success(f"✅ {len(_sc['upload_df'])} rows loaded")
                except Exception as _sc_e:
                    st.error(str(_sc_e))

            if _sc["upload_df"] is not None:
                if _sc["results_df"] is not None:
                    _sc_total = (
                        int(_sc["monthly_df"]["Allocated_Predicted_APPS_Rounded"].sum())
                        if _sc["monthly_df"] is not None
                        and "Allocated_Predicted_APPS_Rounded" in _sc["monthly_df"].columns
                        else None
                    )
                    if _sc_total is not None:
                        st.caption(f"Last run: {_sc_total:,} APPS")
                if st.button(f"▶ Run {_sc['name']}", key=f"sc_run_{_si}", use_container_width=True):
                    if st.session_state.coeff_df is None:
                        st.error("Upload model file first.")
                    else:
                        with st.spinner(f"Running {_sc['name']}…"):
                            _sc_res, _sc_mon = run_scenario(_sc["upload_df"], st.session_state.coeff_df)
                        _sc["results_df"] = _sc_res
                        _sc["monthly_df"] = _sc_mon
                        _sc["input_snap"] = _sc["upload_df"].copy()
                        st.rerun()
            else:
                st.caption("Upload spend data to enable.")

    st.markdown("---")
    with st.expander("🔧 Spaces diagnostics"):
        region = os.environ.get("SPACES_REGION", "").lower().strip()
        bucket = os.environ.get("SPACES_BUCKET", "")
        st.markdown(f"**Region:** `{region or '(not set)'}`")
        st.markdown(f"**Bucket:** `{bucket or '(not set)'}`")
        st.markdown(f"**SPACES_KEY set:** `{'yes' if os.environ.get('SPACES_KEY') else 'no'}`")
        st.markdown(f"**SPACES_SECRET set:** `{'yes' if os.environ.get('SPACES_SECRET') else 'no'}`")
        st.markdown(f"**SPACES_MODEL_FILE:** `{os.environ.get('SPACES_MODEL_FILE', '(default)')}`")
        st.markdown(f"**SPACES_SPEND_FILE:** `{os.environ.get('SPACES_SPEND_FILE', '(default)')}`")
        if st.session_state.spaces_errors:
            for k, msg in st.session_state.spaces_errors.items():
                st.error(f"{k}: {msg}")
        else:
            st.success("No Spaces errors recorded")

    st.markdown(
        "<small style='color:var(--text-color);opacity:0.5'>"
        "Oracle v1.0<br>SigmaAIAnalytics.com</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.1rem;margin-bottom:0;"
    "color:var(--text-color)'>🎯 Baseline</h1>"
    "<p style='color:var(--text-color);opacity:0.55;margin-top:0.1rem'>"
    "Application Calculator — manual spend input → predicted APPs</p>",
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Baseline Spend Data Input
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📋 Baseline Spend Data Input</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='note-box'>Enter future monthly spend data manually below. "
    "Or upload a batch file from the Baseline dropdown under the Scenarios section on the left.</div>",
    unsafe_allow_html=True,
)

# ── Template CSV (built once, used below the table) ──────────────────────────
_template_df = pd.DataFrame(columns=_REQUIRED_COLS)
_template_csv = _template_df.to_csv(index=False).encode("utf-8")

# Default 5 rows
default_rows = pd.DataFrame(
    {
        "Date":           [date.today()] * 5,
        "State":          ["AL"] * 5,
        "DSP ($)":        [0.0] * 5,
        "LeadGen ($)":    [0.0] * 5,
        "Paid Search ($)":[0.0] * 5,
        "Paid Social ($)":[0.0] * 5,
        "Prescreen ($)":  [0.0] * 5,
        "Referrals ($)":  [0.0] * 5,
        "Sweepstakes ($)":[0.0] * 5,
    }
)

column_config = {
    "Date":  st.column_config.DateColumn("Date", required=True),
    "State": st.column_config.SelectboxColumn("State", options=STATE_OPTIONS, required=True),
    **{
        col: st.column_config.NumberColumn(col, min_value=0.0, format="%.2f", default=0.0)
        for col in SPEND_COLUMNS
    },
}

_sc0 = st.session_state.scenarios[0]
_editor_data = (
    _sc0["upload_df"]
    if _sc0["upload_df"] is not None
    else default_rows
)

edited_df = st.data_editor(
    _editor_data,
    column_config=column_config,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key=f"spend_editor_{_sc0['upload_version']}",
)

st.markdown("<br>", unsafe_allow_html=True)

_btn_col, _tmpl_col = st.columns([2, 1])
run_clicked = _btn_col.button("▶ Run Predictions", type="primary", use_container_width=True)
_tmpl_col.download_button(
    "⬇ Download template",
    data=_template_csv,
    file_name="spend_template.csv",
    mime="text/csv",
    use_container_width=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION — triggered on button click
# ══════════════════════════════════════════════════════════════════════════════
if run_clicked:
    # ── Validation ────────────────────────────────────────────────────────────
    if st.session_state.coeff_df is None:
        st.error("⚠️ Please upload modelcoeff_and_prodfactors.csv in the sidebar first.")
        st.stop()

    valid_rows = edited_df.dropna(subset=["Date", "State"])
    valid_rows = valid_rows[valid_rows["State"].astype(str).str.strip() != ""]

    if valid_rows.empty:
        st.error("⚠️ Input table must have at least one row with a valid Date and State.")
        st.stop()

    with st.spinner("Running predictions…"):
        _results, _monthly = run_scenario(valid_rows, st.session_state.coeff_df)

    _sc0 = st.session_state.scenarios[0]
    _sc0["results_df"] = _results
    _sc0["input_snap"] = valid_rows.copy()
    _sc0["monthly_df"] = _monthly


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Output
# ══════════════════════════════════════════════════════════════════════════════
_active_scens = [
    sc for sc in st.session_state.scenarios
    if sc["monthly_df"] is not None and not sc["monthly_df"].empty
]

if _active_scens:
    st.divider()
    st.markdown("<div class='section-header'>📊 Predictions</div>", unsafe_allow_html=True)

    # ── Coverage issues expander (baseline only) ──────────────────────────────
    _sc0 = st.session_state.scenarios[0]
    if _sc0["results_df"] is not None:
        _issue_rows = []
        _fail_rows  = _sc0["results_df"][_sc0["results_df"]["Model_Status"] != "OK"]
        _real_fails = _fail_rows[
            ~_fail_rows["State"].astype(str).str.strip().str.lower().isin(("nan", "none", ""))
        ]
        if not _real_fails.empty:
            for _state, _grp in _real_fails.groupby("State"):
                _issue_rows.append({"State": _state, "Issue": f"No model found — {len(_grp)} week(s) skipped"})
        if st.session_state.coeff_df is not None and "Key" in st.session_state.coeff_df.columns:
            _input_states = set(_sc0["results_df"]["State"].astype(str).unique())
            _model_states = set(
                st.session_state.coeff_df["Key"].dropna()
                .apply(lambda k: _parse_key(str(k)).get("STATE_CD", "")).unique()
            ) - {""}
            for _s in sorted(_model_states - _input_states):
                _issue_rows.append({"State": _s, "Issue": "Has a model but no spend data provided"})
        if _issue_rows:
            with st.expander(f"⚠️ {len(_issue_rows)} coverage issue(s)", expanded=False):
                st.dataframe(pd.DataFrame(_issue_rows), use_container_width=True, hide_index=True)

    # ── Full-month filter per scenario ────────────────────────────────────────
    _full_monthlys: dict[str, pd.DataFrame] = {}
    for _asc in _active_scens:
        _fm = _full_month_filter(_asc["monthly_df"])
        if _fm is not None and not _fm.empty:
            _full_monthlys[_asc["name"]] = _fm

    if _full_monthlys:
        _all_monthly = pd.concat(list(_full_monthlys.values()), ignore_index=True)

        # ── Unified filter bar ────────────────────────────────────────────────
        _mf1, _mf2, _mf3, _mf4, _mf5 = st.columns(5)

        _sel_m_st = _mf1.multiselect(
            "Filter by State",
            sorted(_all_monthly["State"].dropna().unique().tolist()),
            key="monthly_filter_state",
            placeholder="All states",
        )
        _m_mo_base   = _all_monthly if not _sel_m_st else _all_monthly[_all_monthly["State"].isin(_sel_m_st)]
        _m_mo_nums   = sorted(_m_mo_base["Calendar_Month"].dropna().unique().astype(int).tolist())
        _m_mo_labels = [_MONTH_NAME.get(m, str(m)) for m in _m_mo_nums]
        _m_mo_map    = dict(zip(_m_mo_labels, _m_mo_nums))
        _sel_m_mo = _mf2.multiselect(
            "Filter by Month",
            _m_mo_labels,
            key="monthly_filter_month",
            placeholder="All months",
        )
        _ch_opts = ["--Default--"] + (
            sorted(_all_monthly["Channel"].dropna().unique().tolist()) if "Channel" in _all_monthly.columns else []
        )
        _ht_opts = ["--Default--"] + (
            sorted(_all_monthly["H_Tactic"].dropna().unique().tolist()) if "H_Tactic" in _all_monthly.columns else []
        )
        _dt_opts = ["--Default--"] + (
            sorted(_all_monthly["Detail_Tactic"].dropna().unique().tolist()) if "Detail_Tactic" in _all_monthly.columns else []
        )
        _sel_m_ch = _mf3.multiselect("Channel",       _ch_opts, key="monthly_filter_channel",       placeholder="All")
        _sel_m_ht = _mf4.multiselect("H_Tactic",      _ht_opts, key="monthly_filter_h_tactic",      placeholder="All")
        _sel_m_dt = _mf5.multiselect("Detail_Tactic", _dt_opts, key="monthly_filter_detail_tactic", placeholder="All")

        _pf_data  = st.session_state.product_factors_df
        _prod_opts = (
            sorted(_pf_data["PRODUCT_FUNDED"].dropna().unique().tolist())
            if _pf_data is not None and not _pf_data.empty else []
        )
        if _prod_opts:
            _pf_col, _ = st.columns([2, 3])
            _sel_prod = _pf_col.multiselect("Filter by Product", _prod_opts, key="monthly_filter_product", placeholder="All products")
        else:
            _sel_prod = []

        # ── APPS View (hardcoded to All; UI selector removed) ─────────────────
        _view = "All"

        _apps_col_map = {"All": "Allocated_Predicted_APPS_Rounded", "Baseline": "Baseline_APPS_Rounded",    "Incremental": "Incremental_APPS_Rounded"}
        _appr_col_map = {"All": "Allocated_Approved_Rounded",       "Baseline": "Baseline_Approved_Rounded", "Incremental": "Incremental_Approved_Rounded"}
        _orig_col_map = {"All": "Allocated_Originations_Rounded",   "Baseline": "Baseline_Originations_Rounded", "Incremental": "Incremental_Originations_Rounded"}
        _selected_apps_col = _apps_col_map[_view]
        _approval_col      = _appr_col_map[_view]
        _origination_col   = _orig_col_map[_view]
        _all_agg_cols      = list(_apps_col_map.values()) + list(_appr_col_map.values()) + list(_orig_col_map.values())
        _orig_rounded_cols = list(_orig_col_map.values())

        # ── Apply filters + aggregate per scenario ────────────────────────────
        _scene_agg: dict[str, pd.DataFrame] = {}
        for _asc in _active_scens:
            _mdf = _full_monthlys.get(_asc["name"])
            if _mdf is None:
                continue
            _mdf = _mdf.copy()
            if _sel_m_st: _mdf = _mdf[_mdf["State"].isin(_sel_m_st)]
            if _sel_m_mo: _mdf = _mdf[_mdf["Calendar_Month"].isin([_m_mo_map[m] for m in _sel_m_mo])]
            _mdf = _apply_grain_filter(_mdf, "Channel",       _sel_m_ch)
            _mdf = _apply_grain_filter(_mdf, "H_Tactic",      _sel_m_ht)
            _mdf = _apply_grain_filter(_mdf, "Detail_Tactic", _sel_m_dt)
            if _sel_prod and _pf_data is not None and not _pf_data.empty:
                _pf_sel = (
                    _pf_data[_pf_data["PRODUCT_FUNDED"].isin(_sel_prod)]
                    .groupby("Key", as_index=False)["APPLICATION_SHARE"].sum()
                )
                _mdf = _mdf.merge(_pf_sel, on="Key", how="left")
                _share = _mdf["APPLICATION_SHARE"].fillna(0)
                for _oc in [c for c in _orig_rounded_cols if c in _mdf.columns]:
                    _mdf[_oc] = (_mdf[_oc].astype(float) * _share).round().astype("Int64")
                _mdf = _mdf.drop(columns=["APPLICATION_SHARE"])
            _agg_cols_present = [c for c in _all_agg_cols if c in _mdf.columns]
            _grain_keys = [c for c in ["Channel", "H_Tactic", "Detail_Tactic"] if c in _mdf.columns]
            _agg = _mdf.groupby(["State", "Calendar_Year", "Calendar_Month"] + _grain_keys, as_index=False)[_agg_cols_present].sum()
            _agg["Period"] = (
                _agg["Calendar_Month"].astype(int).map(_MONTH_NAME)
                + " " + _agg["Calendar_Year"].astype(int).astype(str)
            )
            _scene_agg[_asc["name"]] = _agg

        # ── Baseline display ──────────────────────────────────────────────────
        _sc0_name = st.session_state.scenarios[0]["name"]
        m_display = _scene_agg.get(_sc0_name)

        if m_display is None or m_display.empty:
            st.info("No rows match the selected filters.")
        else:
            # ── Blended rates (approval and origination) ──────────────────────
            _appr_apps_sum = int(m_display["Allocated_Predicted_APPS_Rounded"].sum()) if "Allocated_Predicted_APPS_Rounded" in m_display.columns else 0
            _has_appr_data = _approval_col    in m_display.columns and _appr_apps_sum > 0
            _appr_sum      = m_display[_approval_col].sum() if _approval_col in m_display.columns else 0
            _has_orig_data = _origination_col in m_display.columns and _appr_sum > 0
            _blended_appr_rate = _appr_sum / _appr_apps_sum if _has_appr_data else 0.0
            _blended_orig_rate = m_display[_origination_col].sum() / _appr_sum if _has_orig_data else 0.0

            _display_apps_col = _selected_apps_col if _selected_apps_col in m_display.columns else None
            _has_approved     = _approval_col    in m_display.columns
            _has_originated   = _origination_col in m_display.columns

            _apps_total = int(m_display[_display_apps_col].sum()) if _display_apps_col else 0
            _appr_total = int(m_display[_approval_col].sum())     if _has_approved     else None
            _orig_total = int(m_display[_origination_col].sum())  if _has_originated   else None

            # ── Total spend from input_snap filtered by State + Month only ────
            _snap = st.session_state.scenarios[0].get("input_snap")
            _spend_total = None
            if _snap is not None and not _snap.empty:
                _sp = _snap.copy()
                _sp["_date"] = pd.to_datetime(_sp["Date"], errors="coerce")
                _sp["_month_num"] = _sp["_date"].dt.month
                if _sel_m_st:
                    _sp = _sp[_sp["State"].isin(_sel_m_st)]
                if _sel_m_mo:
                    _sp = _sp[_sp["_month_num"].isin([_m_mo_map[m] for m in _sel_m_mo])]
                _spend_total = _sp[SPEND_COLUMNS].sum().sum()

            def _fmt_spend(v: float) -> str:
                if v >= 1_000_000:
                    return f"${v / 1_000_000:.1f}MM"
                if v >= 1_000:
                    return f"${v / 1_000:.1f}K"
                return f"${v:.0f}"

            _n_mc  = 1 + (1 if _appr_total is not None else 0) + (1 if _orig_total is not None else 0) + (1 if _spend_total is not None else 0)
            _mcols = st.columns(_n_mc)
            _mcols[0].metric("Predicted Applications", f"{_apps_total:,}")
            if _appr_total is not None:
                _mcols[1].metric("Likely Approvals", f"{_appr_total:,}")
                if _has_appr_data:
                    _mcols[1].markdown(
                        f"<div style='font-size:0.75rem;color:var(--text-color);opacity:0.6;margin-top:0.15rem'>"
                        f"Approval Rate: <strong>{_blended_appr_rate * 100:.0f}%</strong></div>",
                        unsafe_allow_html=True,
                    )
            if _orig_total is not None:
                _orig_idx = 1 + (1 if _appr_total is not None else 0)
                _mcols[_orig_idx].metric("Likely Funded", f"{_orig_total:,}")
                if _has_orig_data:
                    _mcols[_orig_idx].markdown(
                        f"<div style='font-size:0.75rem;color:var(--text-color);opacity:0.6;margin-top:0.15rem'>"
                        f"Funding Rate: <strong>{_blended_orig_rate * 100:.0f}%</strong></div>",
                        unsafe_allow_html=True,
                    )
            if _spend_total is not None:
                _spend_idx = 1 + (1 if _appr_total is not None else 0) + (1 if _orig_total is not None else 0)
                _grain_active = any([_sel_m_ch, _sel_m_ht, _sel_m_dt])
                _mcols[_spend_idx].metric("Total Spend (@ State/Month only)", "N/A" if _grain_active else _fmt_spend(_spend_total))
                if _grain_active:
                    _cpf_label = "Not Calculated"
                elif _orig_total and _orig_total > 0:
                    _cpf_label = _fmt_spend(_spend_total / _orig_total)
                else:
                    _cpf_label = "—"
                _mcols[_spend_idx].markdown(
                    f"<div style='font-size:0.75rem;color:var(--text-color);opacity:0.6;margin-top:0.15rem'>"
                    f"CPF (per State/Month): <strong>{_cpf_label}</strong></div>",
                    unsafe_allow_html=True,
                )

            _grain_cols = [c for c in ["Channel", "H_Tactic", "Detail_Tactic"] if c in m_display.columns]
            _monthly_primary_cols = ["State", "Period"] + _grain_cols
            _col_rename = {}
            if _display_apps_col:
                _monthly_primary_cols.append(_display_apps_col)
                _col_rename[_display_apps_col] = "Predicted Applications"
            if _has_approved:
                _monthly_primary_cols.append(_approval_col)
                _col_rename[_approval_col] = "Likely Approvals"
            if _has_originated:
                _monthly_primary_cols.append(_origination_col)
                _col_rename[_origination_col] = "Likely Funded"
            _monthly_fmt   = {c: "{:,}" for c in _monthly_primary_cols if c not in ["State", "Period"] + _grain_cols}
            _display_slice = m_display[_monthly_primary_cols].rename(columns=_col_rename)
            _display_fmt   = {_col_rename.get(c, c): fmt for c, fmt in _monthly_fmt.items()}

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                _display_slice.style.format(_display_fmt, na_rep=""),
                use_container_width=True,
                height=min(400, 45 + len(m_display) * 35),
                hide_index=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇ Download Monthly as CSV",
            data=(m_display.to_csv(index=False).encode("utf-8") if m_display is not None and not m_display.empty else b""),
            file_name=f"monthly_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Comments
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("<div class='section-header'>💬 Comments</div>", unsafe_allow_html=True)

_spaces_configured = bool(
    os.environ.get("SPACES_KEY") and
    os.environ.get("SPACES_SECRET") and
    os.environ.get("SPACES_BUCKET")
)

if not _spaces_configured:
    st.info("Comments require Spaces to be configured.")
else:
    _all_comments = _load_comments()
    _open_comments     = [c for c in _all_comments if not c.get("resolved", False)]
    _resolved_comments = [c for c in _all_comments if c.get("resolved", False)]

    # ── Open comments ─────────────────────────────────────────────────────────
    if _open_comments:
        for _c in _open_comments:
            _cc1, _cc2 = st.columns([11, 1])
            with _cc1:
                st.markdown(
                    f"**{_c['author']}** "
                    f"<span style='color:var(--text-color);opacity:0.45;font-size:0.82rem'>"
                    f"{_c['timestamp'][:16].replace('T', ' ')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(_c["text"])
            with _cc2:
                if st.button("✓", key=f"resolve_{_c['id']}", help="Mark resolved"):
                    for _x in _all_comments:
                        if _x["id"] == _c["id"]:
                            _x["resolved"] = True
                    _ok, _err = _save_comments(_all_comments)
                    if _ok:
                        st.rerun()
                    else:
                        st.error(f"Could not save: {_err}")
    else:
        st.markdown(
            "<div style='color:var(--text-color);opacity:0.5;font-size:0.9rem'>"
            "No open comments.</div>",
            unsafe_allow_html=True,
        )

    # ── Resolved comments (collapsed) ────────────────────────────────────────
    if _resolved_comments:
        with st.expander(f"Resolved ({len(_resolved_comments)})"):
            for _c in _resolved_comments:
                st.markdown(
                    f"~~**{_c['author']}**~~ "
                    f"<span style='color:var(--text-color);opacity:0.4;font-size:0.82rem'>"
                    f"{_c['timestamp'][:16].replace('T', ' ')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"~~{_c['text']}~~")

    # ── New comment form ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("comment_form", clear_on_submit=True):
        _name = st.text_input("Your name")
        _text = st.text_area("Comment", height=100)
        _submitted = st.form_submit_button("Submit comment")
        if _submitted:
            if not _name.strip() or not _text.strip():
                st.warning("Please enter both your name and a comment.")
            else:
                _new = {
                    "id":        str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "author":    _name.strip(),
                    "text":      _text.strip(),
                    "resolved":  False,
                    "page":      "Predictions",
                }
                _all_comments.append(_new)
                _ok, _err = _save_comments(_all_comments)
                if _ok:
                    st.success("Comment saved.")
                    st.rerun()
                else:
                    st.error(f"Could not save comment: {_err}")
