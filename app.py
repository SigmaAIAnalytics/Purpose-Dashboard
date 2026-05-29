from __future__ import annotations

import calendar as _calendar
import json
import os
import uuid
from datetime import date, datetime
from io import BytesIO
from typing import Any

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.client import Config as _BotoConfig
from build_state_division_models import roll_up_weekly_forecast_to_monthly, spread_monthly_spend_to_weekly

st.set_page_config(
    page_title="Forecast — Purpose Dashboard",
    page_icon="🔮",
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

def _safe_col(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name)).strip("_")


_PF_COLS = ["PRODUCT_FUNDED", "APPLICATION_SHARE", "APPROVAL_RATE", "ORIGINATION_RATE"]


def _split_model_file(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split modelcoeff_and_prodfactors into (coeff_df, product_factors_df)."""
    coeff = (
        df.drop(columns=[c for c in _PF_COLS if c in df.columns])
        .drop_duplicates(subset=["Key"])
        .reset_index(drop=True)
    )
    pf_present = [c for c in ["Key"] + _PF_COLS if c in df.columns]
    product_factors = (
        df[pf_present]
        .dropna(subset=["PRODUCT_FUNDED"])
        .reset_index(drop=True)
    )
    return coeff, product_factors


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


# ── Session state init ────────────────────────────────────────────────────────
if "results_df"        not in st.session_state: st.session_state.results_df        = None
if "monthly_df"        not in st.session_state: st.session_state.monthly_df        = None
if "input_snap"        not in st.session_state: st.session_state.input_snap        = None
if "coeff_df"          not in st.session_state: st.session_state.coeff_df          = None
if "coeff_source"      not in st.session_state: st.session_state.coeff_source      = None
if "product_factors_df"not in st.session_state: st.session_state.product_factors_df= None
if "upload_df"         not in st.session_state: st.session_state.upload_df         = None
if "upload_version"    not in st.session_state: st.session_state.upload_version    = 0
if "last_input_name"   not in st.session_state: st.session_state.last_input_name   = None
if "spend_source"      not in st.session_state: st.session_state.spend_source      = None
if "spaces_errors"     not in st.session_state: st.session_state.spaces_errors     = {}

# ── Auto-load from DO Spaces (runs once per session when no file is loaded) ───
if st.session_state.coeff_df is None:
    _spaces_model, _err = _load_df_from_spaces(
        "SPACES_MODEL_FILE", "modelcoeff_and_prodfactors.csv"
    )
    if _spaces_model is not None:
        _coeff, _pf = _split_model_file(_spaces_model)
        st.session_state.coeff_df          = _coeff
        st.session_state.product_factors_df = _pf
        st.session_state.coeff_source      = "spaces"
        st.session_state.spaces_errors.pop("model", None)
    elif _err:
        st.session_state.spaces_errors["model"] = _err

if st.session_state.upload_df is None:
    _spaces_spend, _err = _load_df_from_spaces("SPACES_SPEND_FILE", "FutureSpend.csv")
    if _spaces_spend is not None:
        try:
            st.session_state.upload_df    = _normalise_upload(_spaces_spend)
            st.session_state.spend_source = "spaces"
            st.session_state.upload_version += 1
            st.session_state.spaces_errors.pop("spend", None)
        except Exception as e:
            st.session_state.spaces_errors["spend"] = f"FutureSpend.csv parsed but normalise failed: {e}"
    elif _err:
        st.session_state.spaces_errors["spend"] = _err


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Model file uploader
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Model File")

    if st.session_state.coeff_source == "spaces":
        _coeff_keys = (
            st.session_state.coeff_df["Key"].dropna().tolist()
            if "Key" in st.session_state.coeff_df.columns else []
        )
        _pf_products = (
            st.session_state.product_factors_df["PRODUCT_FUNDED"].dropna().unique().tolist()
            if st.session_state.product_factors_df is not None else []
        )
        st.success(
            f"✅ Auto-loaded from Spaces — {len(_coeff_keys)} model keys, "
            f"{len(_pf_products)} product(s)"
        )
        with st.expander("Override with a local file"):
            _ov_model = st.file_uploader(
                "Upload modelcoeff_and_prodfactors.csv",
                type=["csv"],
                key="model_uploader",
            )
            if _ov_model:
                try:
                    _ov_df = pd.read_csv(_ov_model)
                    _ov_coeff, _ov_pf = _split_model_file(_ov_df)
                    st.session_state.coeff_df          = _ov_coeff
                    st.session_state.product_factors_df = _ov_pf
                    st.session_state.coeff_source      = "upload"
                    st.success(
                        f"✅ Overridden — {len(_ov_coeff)} keys, "
                        f"{len(_ov_pf['PRODUCT_FUNDED'].dropna().unique())} product(s)"
                    )
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
    else:
        st.markdown("Upload `modelcoeff_and_prodfactors.csv` generated by the model pipeline.")
        model_file = st.file_uploader(
            "Upload modelcoeff_and_prodfactors.csv",
            type=["csv"],
            key="model_uploader",
        )
        if model_file:
            try:
                _raw = pd.read_csv(model_file)
                _coeff, _pf = _split_model_file(_raw)
                st.session_state.coeff_df          = _coeff
                st.session_state.product_factors_df = _pf
                st.session_state.coeff_source      = "upload"
                _keys = _coeff["Key"].dropna().tolist() if "Key" in _coeff.columns else []
                _prods = _pf["PRODUCT_FUNDED"].dropna().unique().tolist()
                st.success(
                    f"✅ Loaded — {len(_keys)} model keys, "
                    f"{len(_prods)} product(s): {', '.join(sorted(_prods))}"
                )
            except Exception as e:
                st.error(f"Failed to read model file: {e}")
        else:
            if st.session_state.coeff_df is not None:
                _keys = (
                    st.session_state.coeff_df["Key"].dropna().tolist()
                    if "Key" in st.session_state.coeff_df.columns else []
                )
                _prods = (
                    st.session_state.product_factors_df["PRODUCT_FUNDED"].dropna().unique().tolist()
                    if st.session_state.product_factors_df is not None else []
                )
                st.success(
                    f"✅ Loaded — {len(_keys)} model keys, "
                    f"{len(_prods)} product(s)"
                )
            else:
                st.info("No file uploaded yet.")

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
        "Purpose Predictor v1.0<br>Replicates Excel Output_Data scoring logic</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.1rem;margin-bottom:0;"
    "color:var(--text-color)'>Purpose Predictor</h1>"
    "<p style='color:var(--text-color);opacity:0.55;margin-top:0.1rem'>"
    "Application Calculator — manual spend input → predicted APPs</p>",
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Spend Data Input
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📋 Spend Data Input</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='note-box'>Enter one row per date / state combination. "
    "Multiple rows for the same state + week are <strong>summed automatically</strong> before prediction.</div>",
    unsafe_allow_html=True,
)

# ── Template download ─────────────────────────────────────────────────────────
_template_df = pd.DataFrame(columns=_REQUIRED_COLS)
_template_csv = _template_df.to_csv(index=False).encode("utf-8")

# ── Upload widget ─────────────────────────────────────────────────────────────
_up_col, _dl_col = st.columns([3, 1])
with _up_col:
    input_file = st.file_uploader(
        "Upload spend data (CSV or Excel) — optional, or fill the table manually below",
        type=["csv", "xlsx"],
        key="input_uploader",
        label_visibility="visible",
    )
with _dl_col:
    st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    st.download_button(
        "⬇ Download template",
        data=_template_csv,
        file_name="spend_template.csv",
        mime="text/csv",
    )

if input_file is not None and input_file.name != st.session_state.last_input_name:
    try:
        raw = (
            pd.read_csv(input_file)
            if input_file.name.endswith(".csv")
            else pd.read_excel(input_file)
        )
        parsed = _normalise_upload(raw)
        st.session_state.upload_df       = parsed
        st.session_state.last_input_name = input_file.name
        st.session_state.upload_version += 1
        st.success(f"✅ Loaded {len(parsed)} row(s) from **{input_file.name}**")
    except Exception as e:
        st.error(f"Could not parse upload: {e}")

st.markdown("<br>", unsafe_allow_html=True)

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

_editor_data = (
    st.session_state.upload_df
    if st.session_state.upload_df is not None
    else default_rows
)

edited_df = st.data_editor(
    _editor_data,
    column_config=column_config,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key=f"spend_editor_{st.session_state.upload_version}",
)

st.markdown("<br>", unsafe_allow_html=True)

run_clicked = st.button("▶ Run Predictions", type="primary")


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

    # ── Run ───────────────────────────────────────────────────────────────────
    with st.spinner("Running predictions…"):
        weekly_df  = _monthly_to_weekly(valid_rows.copy())
        results_df = run_predictions(weekly_df, st.session_state.coeff_df)

        # Baseline: same weeks/states with all spend zeroed
        zero_df = weekly_df.copy()
        for _c in SPEND_COLUMNS:
            zero_df[_c] = 0.0
        baseline_df = run_predictions(zero_df, st.session_state.coeff_df)

        baseline_lookup = (
            baseline_df[["State", "ISO_Year", "ISO_Week", "Model_Key", "Predicted APPS"]]
            .rename(columns={"Predicted APPS": "Baseline APPS"})
        )
        results_df = results_df.merge(
            baseline_lookup,
            on=["State", "ISO_Year", "ISO_Week", "Model_Key"],
            how="left",
        )
        results_df["Baseline APPS"] = results_df[["Predicted APPS", "Baseline APPS"]].min(axis=1)
        results_df["Incremental APPS"] = (
            results_df["Predicted APPS"] - results_df["Baseline APPS"].fillna(0)
        ).clip(lower=0).round().astype("Int64")

        # Product allocation (only if factors file is loaded)
        if st.session_state.product_factors_df is not None:
            pf = st.session_state.product_factors_df.copy()
            pf["PRODUCT_FUNDED"] = pf["PRODUCT_FUNDED"].astype(str)

            # Per-product Applications and Originations (equal: PRODUCT_FUNDED records originated loans)
            for product, grp in pf.groupby("PRODUCT_FUNDED", dropna=False):
                pkey    = _safe_col(product)
                factors = results_df[["Model_Key"]].merge(
                    grp[["Key", "APPLICATION_SHARE"]],
                    left_on="Model_Key", right_on="Key", how="left",
                )
                alloc = results_df["Predicted APPS Raw"].clip(lower=0) * factors["APPLICATION_SHARE"].fillna(0).values
                results_df[f"APPLICATIONS_{pkey}"] = alloc.fillna(0).round().astype(int)
                results_df[f"ORIGINATIONS_{pkey}"] = alloc.fillna(0).round().astype(int)

            # Key-level Approved and Originated — rates are not meaningful per product
            _key_rates = pf.drop_duplicates("Key")[["Key", "APPROVAL_RATE", "ORIGINATION_RATE"]]
            _rates = results_df[["Model_Key"]].merge(
                _key_rates, left_on="Model_Key", right_on="Key", how="left"
            )
            _raw_clipped = results_df["Predicted APPS Raw"].clip(lower=0)
            # *_Total (float) columns flow into the monthly rollup for accurate pro-rating
            results_df["APPROVAL_Total"]         = (_raw_clipped * _rates["APPROVAL_RATE"].fillna(0).values).fillna(0)
            results_df["ORIGINATION_Total"]      = (_raw_clipped * _rates["ORIGINATION_RATE"].fillna(0).values).fillna(0)
            results_df["Allocated_Approved"]     = results_df["APPROVAL_Total"].round().astype(int)
            results_df["Allocated_Originations"] = results_df["ORIGINATION_Total"].round().astype(int)

    st.session_state.results_df = results_df
    st.session_state.input_snap = valid_rows.copy()

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
        _monthly_pred["Baseline_APPS_Rounded"]     = _monthly_pred["Baseline APPS"].round().astype("Int64")
        _monthly_pred["Incremental_APPS_Rounded"]  = _monthly_pred["Incremental APPS"].round().astype("Int64")
        _monthly_pred = _monthly_pred.drop(columns=["_Baseline_raw"])

    st.session_state.monthly_df = _monthly_pred


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Output
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    input_snap = st.session_state.input_snap

    ok_rows   = results_df[results_df["Model_Status"] == "OK"]
    fail_rows = results_df[results_df["Model_Status"] != "OK"]

    st.divider()
    st.markdown("<div class='section-header'>📊 Predictions</div>", unsafe_allow_html=True)

    # ── Warnings for missing coefficients (skip NaN/blank states) ───────────────
    if not fail_rows.empty:
        for _, r in fail_rows.iterrows():
            if str(r["State"]).strip().lower() in ("nan", "none", ""):
                continue
            st.warning(
                f"⚠️ No coefficient found for state **{r['State']}** "
                f"(Week {r['ISO_Week']}) — skipped."
            )

    _MONTH_NAME = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    if not ok_rows.empty:
        st.success(
            f"✅ {len(ok_rows)} prediction row(s) across "
            f"{ok_rows[['State','ISO_Year','ISO_Week']].drop_duplicates().shape[0]} "
            "state-week combination(s)"
        )

        # ── Output filters ────────────────────────────────────────────────────
        _ff1, _ff2, _ff3, _ff4, _ff5 = st.columns(5)

        # State (independent)
        _sel_st = _ff1.multiselect(
            "Filter by State",
            sorted(ok_rows["State"].dropna().unique().tolist()),
            key="filter_state", placeholder="All states",
        )

        # Month (scoped to State)
        _mo_base   = ok_rows if not _sel_st else ok_rows[ok_rows["State"].isin(_sel_st)]
        _mo_nums   = sorted(_mo_base["Month"].dropna().unique().astype(int).tolist())
        _mo_labels = [_MONTH_NAME.get(m, str(m)) for m in _mo_nums]
        _mo_map    = dict(zip(_mo_labels, _mo_nums))
        _sel_mo    = _ff2.multiselect(
            "Filter by Month", _mo_labels,
            key="filter_month", placeholder="All months",
        )

        # Channel (scoped to State + Month)
        _ch_base = _mo_base if not _sel_mo else _mo_base[_mo_base["Month"].isin([_mo_map[m] for m in _sel_mo])]
        _sel_ch  = _ff3.multiselect(
            "Filter by Channel",
            sorted(_ch_base["Channel"].dropna().unique().tolist()),
            key="filter_channel", placeholder="All channels",
        )

        # H_Tactic (scoped to Channel)
        _ht_base = _ch_base if not _sel_ch else _ch_base[_ch_base["Channel"].isin(_sel_ch)]
        _sel_ht  = _ff4.multiselect(
            "Filter by H_Tactic",
            sorted(_ht_base["H_Tactic"].dropna().unique().tolist()),
            key="filter_h_tactic", placeholder="All",
        )

        # Detail_Tactic (scoped to H_Tactic)
        _dt_base = _ht_base if not _sel_ht else _ht_base[_ht_base["H_Tactic"].isin(_sel_ht)]
        _sel_dt  = _ff5.multiselect(
            "Filter by Detail_Tactic",
            sorted(_dt_base["Detail_Tactic"].dropna().unique().tolist()),
            key="filter_detail_tactic", placeholder="All",
        )

        # Product (only shown if product factors file is loaded)
        _sel_prod = []
        if st.session_state.product_factors_df is not None:
            _prod_col, _ = st.columns([1, 4])
            _sel_prod = _prod_col.multiselect(
                "Filter by Product Funded",
                sorted(st.session_state.product_factors_df["PRODUCT_FUNDED"].dropna().astype(str).unique().tolist()),
                key="filter_product", placeholder="All products",
            )

        # Apply row filters — empty list = no restriction
        display_df = results_df.copy()
        if _sel_st: display_df = display_df[display_df["State"].isin(_sel_st)]
        if _sel_mo: display_df = display_df[display_df["Month"].isin([_mo_map[m] for m in _sel_mo])]
        if _sel_ch: display_df = display_df[display_df["Channel"].isin(_sel_ch)]
        if _sel_ht: display_df = display_df[display_df["H_Tactic"].isin(_sel_ht)]
        if _sel_dt: display_df = display_df[display_df["Detail_Tactic"].isin(_sel_dt)]

        # ── Primary output table ──────────────────────────────────────────────
        primary_cols = [
            "State", "ISO_Year", "ISO_Week", "Month",
            *SPEND_COLUMNS,
            "Channel", "H_Tactic", "Detail_Tactic", "Product",
            "Predicted APPS", "Baseline APPS", "Incremental APPS",
        ]
        _prod_format: dict = {}
        if st.session_state.product_factors_df is not None:
            if not _sel_prod:
                primary_cols += ["Allocated_Approved", "Allocated_Originations"]
                _prod_format  = {"Allocated_Approved": "{:,}", "Allocated_Originations": "{:,}"}
            elif len(_sel_prod) == 1:
                _pkey = _safe_col(_sel_prod[0])
                # Approvals are key-level (same for all products); Applications = Originations per product
                primary_cols += [f"APPLICATIONS_{_pkey}", f"ORIGINATIONS_{_pkey}", "Allocated_Approved"]
                _prod_format  = {
                    f"APPLICATIONS_{_pkey}": "{:,}",
                    f"ORIGINATIONS_{_pkey}": "{:,}",
                    "Allocated_Approved":    "{:,}",
                }
            else:
                primary_cols += ["Allocated_Approved", "Allocated_Originations"]
                _prod_format  = {"Allocated_Approved": "{:,}", "Allocated_Originations": "{:,}"}
        primary_cols = [c for c in primary_cols if c in results_df.columns]

        if display_df.empty:
            st.info("No rows match the selected filters.")
        else:
            st.dataframe(
                display_df[primary_cols].style.format(
                    {
                        **{c: "{:,.2f}" for c in SPEND_COLUMNS if c in display_df.columns},
                        "Predicted APPS":               "{:,}",
                        "Baseline APPS":                "{:,}",
                        "Incremental APPS":             "{:,}",
                        "95% Confidence Lower Limit":   "{:,}",
                        "95% Confidence Upper Limit":   "{:,}",
                        **_prod_format,
                    },
                    na_rep="",
                ),
                use_container_width=True,
                height=min(400, 45 + len(display_df) * 35),
                hide_index=True,
            )

    # ── Download buttons ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    c1, c2, _ = st.columns([1, 1, 4])

    # Full export includes both primary + detail columns
    export_cols = [
        "State", "ISO_Year", "ISO_Week", "Month",
        *SPEND_COLUMNS,
        "Channel", "H_Tactic", "Detail_Tactic", "Product",
        "Predicted APPS", "Baseline APPS", "Incremental APPS",
        "Allocated_Approved", "Allocated_Originations", "APPROVAL_Total", "Predicted APPS Raw",
        "95% Confidence Lower Limit", "95% Confidence Upper Limit",
        "time_index", "time_index_sq",
        "DSP_contrib", "LeadGen_contrib", "Paid_Search_contrib",
        "Paid_Social_contrib", "Prescreen_contrib", "Referrals_contrib",
        "Sweepstakes_contrib", "time_index_contrib", "time_index_sq_contrib",
        "weekly_dummy_contrib", "Intercept", "Sigma", "Model_Key", "Model_Status",
    ]
    export_cols = [c for c in export_cols if c in results_df.columns]
    export_df   = results_df[export_cols]

    with c1:
        st.download_button(
            label="⬇ Download as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{ts}.csv",
            mime="text/csv",
        )

    with c2:
        excel_bytes = to_excel_bytes(export_df, input_snap)
        st.download_button(
            label="⬇ Download as Excel",
            data=excel_bytes,
            file_name=f"predictions_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── Monthly Rollup ────────────────────────────────────────────────────────
    monthly_df = st.session_state.monthly_df
    if monthly_df is not None and not monthly_df.empty:
        # Keep only months where every day is covered by the forecast
        _coverage = (
            monthly_df.groupby(["State", "Calendar_Year", "Calendar_Month"], as_index=False)["Allocated_Days"]
            .max()
        )
        _coverage["_days_in_month"] = _coverage.apply(
            lambda r: _calendar.monthrange(int(r["Calendar_Year"]), int(r["Calendar_Month"]))[1],
            axis=1,
        )
        _full_months = _coverage[_coverage["Allocated_Days"] >= _coverage["_days_in_month"]][
            ["State", "Calendar_Year", "Calendar_Month"]
        ]
        monthly_df = monthly_df.merge(_full_months, on=["State", "Calendar_Year", "Calendar_Month"], how="inner")

    if monthly_df is not None and not monthly_df.empty:
        st.divider()
        st.markdown("<div class='section-header'>📅 Monthly Rollup</div>", unsafe_allow_html=True)

        _mf1, _mf2, _mf3, _mf4, _mf5 = st.columns(5)

        # State (independent)
        _sel_m_st = _mf1.multiselect(
            "Filter by State",
            sorted(monthly_df["State"].dropna().unique().tolist()),
            key="monthly_filter_state",
            placeholder="All states",
        )

        # Month (scoped to State)
        _m_mo_base  = monthly_df if not _sel_m_st else monthly_df[monthly_df["State"].isin(_sel_m_st)]
        _m_mo_nums  = sorted(_m_mo_base["Calendar_Month"].dropna().unique().astype(int).tolist())
        _m_mo_labels = [_MONTH_NAME.get(m, str(m)) for m in _m_mo_nums]
        _m_mo_map    = dict(zip(_m_mo_labels, _m_mo_nums))
        _sel_m_mo = _mf2.multiselect(
            "Filter by Month",
            _m_mo_labels,
            key="monthly_filter_month",
            placeholder="All months",
        )

        # Channel (scoped to State + Month)
        _m_ch_base = _m_mo_base if not _sel_m_mo else _m_mo_base[_m_mo_base["Calendar_Month"].isin([_m_mo_map[m] for m in _sel_m_mo])]
        _m_ch_opts = sorted(_m_ch_base["Channel"].dropna().unique().tolist()) if "Channel" in monthly_df.columns else []
        _sel_m_ch = _mf3.multiselect(
            "Filter by Channel",
            _m_ch_opts,
            key="monthly_filter_channel",
            placeholder="All channels",
        )

        # H_Tactic (scoped to Channel)
        _m_ht_base = _m_ch_base if not _sel_m_ch else _m_ch_base[_m_ch_base["Channel"].isin(_sel_m_ch)]
        _m_ht_opts = sorted(_m_ht_base["H_Tactic"].dropna().unique().tolist()) if "H_Tactic" in monthly_df.columns else []
        _sel_m_ht = _mf4.multiselect(
            "Filter by H_Tactic",
            _m_ht_opts,
            key="monthly_filter_h_tactic",
            placeholder="All",
        )

        # Detail_Tactic (scoped to H_Tactic)
        _m_dt_base = _m_ht_base if not _sel_m_ht else _m_ht_base[_m_ht_base["H_Tactic"].isin(_sel_m_ht)]
        _m_dt_opts = sorted(_m_dt_base["Detail_Tactic"].dropna().unique().tolist()) if "Detail_Tactic" in monthly_df.columns else []
        _sel_m_dt = _mf5.multiselect(
            "Filter by Detail_Tactic",
            _m_dt_opts,
            key="monthly_filter_detail_tactic",
            placeholder="All",
        )

        # Product filter (column selector — not a row filter)
        _sel_m_prod = []
        if st.session_state.product_factors_df is not None:
            _m_prod_col, _ = st.columns([1, 4])
            _sel_m_prod = _m_prod_col.multiselect(
                "Filter by Product Funded",
                sorted(st.session_state.product_factors_df["PRODUCT_FUNDED"].dropna().astype(str).unique().tolist()),
                key="monthly_filter_product", placeholder="All products",
            )

        m_display = monthly_df.copy()
        if _sel_m_st: m_display = m_display[m_display["State"].isin(_sel_m_st)]
        if _sel_m_mo: m_display = m_display[m_display["Calendar_Month"].isin([_m_mo_map[m] for m in _sel_m_mo])]
        if _sel_m_ch: m_display = m_display[m_display["Channel"].isin(_sel_m_ch)]
        if _sel_m_ht: m_display = m_display[m_display["H_Tactic"].isin(_sel_m_ht)]
        if _sel_m_dt: m_display = m_display[m_display["Detail_Tactic"].isin(_sel_m_dt)]

        monthly_primary_cols = [
            "State", "Calendar_Year", "Calendar_Month",
            "Channel", "H_Tactic", "Detail_Tactic",
            "Allocated_Predicted_APPS_Rounded",
            "Baseline_APPS_Rounded",
            "Incremental_APPS_Rounded",
        ]
        monthly_fmt = {
            "Allocated_Predicted_APPS_Rounded":  "{:,}",
            "Baseline_APPS_Rounded":             "{:,}",
            "Incremental_APPS_Rounded":          "{:,}",
        }

        if st.session_state.product_factors_df is not None:
            if len(_sel_m_prod) == 1:
                _m_pkey = _safe_col(_sel_m_prod[0])
                _m_apps_col  = f"APPLICATIONS_{_m_pkey}"
                _m_orig_col  = f"ORIGINATIONS_{_m_pkey}"
                # Round on-the-fly — rollup stores these as floats
                if _m_apps_col in m_display.columns:
                    m_display[f"{_m_apps_col}_Rounded"] = m_display[_m_apps_col].round().astype("Int64")
                    m_display[f"{_m_orig_col}_Rounded"] = m_display[_m_orig_col].round().astype("Int64")
                    monthly_primary_cols += [f"{_m_apps_col}_Rounded", f"{_m_orig_col}_Rounded"]
                    monthly_fmt[f"{_m_apps_col}_Rounded"] = "{:,}"
                    monthly_fmt[f"{_m_orig_col}_Rounded"] = "{:,}"
                monthly_primary_cols += ["Allocated_Approved_Rounded"]
                monthly_fmt["Allocated_Approved_Rounded"] = "{:,}"
            else:
                monthly_primary_cols += ["Allocated_Approved_Rounded", "Allocated_Originations_Rounded"]
                monthly_fmt["Allocated_Approved_Rounded"]     = "{:,}"
                monthly_fmt["Allocated_Originations_Rounded"] = "{:,}"

        monthly_primary_cols = [c for c in monthly_primary_cols if c in m_display.columns]

        if m_display.empty:
            st.info("No rows match the selected filters.")
        else:
            st.dataframe(
                m_display[monthly_primary_cols].style.format(monthly_fmt, na_rep=""),
                use_container_width=True,
                height=min(400, 45 + len(m_display) * 35),
                hide_index=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇ Download Monthly as CSV",
            data=m_display.to_csv(index=False).encode("utf-8"),
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
