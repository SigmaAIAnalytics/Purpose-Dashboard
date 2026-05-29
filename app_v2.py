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

pd.set_option("styler.render.max_elements", 1_000_000)
from botocore.client import Config as _BotoConfig
from build_state_division_models import roll_up_weekly_forecast_to_monthly, spread_monthly_spend_to_weekly

st.set_page_config(
    page_title="Forecast — Purpose Dashboard",
    page_icon="🧿",
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


# ── Session state init ────────────────────────────────────────────────────────
if "results_df"          not in st.session_state: st.session_state.results_df          = None
if "monthly_df"          not in st.session_state: st.session_state.monthly_df          = None
if "input_snap"          not in st.session_state: st.session_state.input_snap          = None
if "coeff_df"            not in st.session_state: st.session_state.coeff_df            = None
if "coeff_source"        not in st.session_state: st.session_state.coeff_source        = None
if "product_factors_df"  not in st.session_state: st.session_state.product_factors_df  = None
if "upload_df"           not in st.session_state: st.session_state.upload_df           = None
if "upload_version"      not in st.session_state: st.session_state.upload_version      = 0
if "last_input_name"     not in st.session_state: st.session_state.last_input_name     = None
if "spend_source"        not in st.session_state: st.session_state.spend_source        = None
if "spaces_errors"       not in st.session_state: st.session_state.spaces_errors       = {}

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
        st.success(f"✅ Auto-loaded from Spaces — {len(_coeff_keys)} model keys")
        with st.expander("Override with a local file"):
            _ov_model = st.file_uploader(
                "Upload modelcoeff_and_prodfactors.csv",
                type=["csv"],
                key="model_uploader",
            )
            if _ov_model:
                try:
                    _ov_raw = pd.read_csv(_ov_model)
                    _ov_coeff = _load_coeff_df(_ov_raw)
                    st.session_state.coeff_df           = _ov_coeff
                    st.session_state.product_factors_df = _load_product_factors(_ov_raw)
                    st.session_state.coeff_source       = "upload"
                    st.success(f"✅ Overridden — {len(_ov_coeff)} keys")
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
                _coeff = _load_coeff_df(_raw)
                st.session_state.coeff_df           = _coeff
                st.session_state.product_factors_df = _load_product_factors(_raw)
                st.session_state.coeff_source       = "upload"
                _keys = _coeff["Key"].dropna().tolist() if "Key" in _coeff.columns else []
                st.success(f"✅ Loaded — {len(_keys)} model keys")
            except Exception as e:
                st.error(f"Failed to read model file: {e}")
        else:
            if st.session_state.coeff_df is not None:
                _keys = (
                    st.session_state.coeff_df["Key"].dropna().tolist()
                    if "Key" in st.session_state.coeff_df.columns else []
                )
                st.success(f"✅ Loaded — {len(_keys)} model keys")
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
        "Oracle v1.0<br>Replicates Excel Output_Data scoring logic</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.1rem;margin-bottom:0;"
    "color:var(--text-color)'>🧿 Oracle</h1>"
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

        # Key-level approvals and originations (available when model file includes rates)
        _cdf = st.session_state.coeff_df
        if "APPROVAL_RATE" in _cdf.columns and "ORIGINATION_RATE" in _cdf.columns:
            _key_rates = _cdf[["Key", "APPROVAL_RATE", "ORIGINATION_RATE"]].copy()
            _rates = results_df[["Model_Key"]].merge(
                _key_rates, left_on="Model_Key", right_on="Key", how="left"
            )
            _raw_clipped = results_df["Predicted APPS Raw"].clip(lower=0)
            # Column prefixes must match roll_up_weekly_forecast_to_monthly() pattern:
            # APPROVAL_* and ORIGINATIONS_* (with S) are prorated; sum → Allocated_Approved/Originations_Rounded
            results_df["APPROVAL_Total"]     = (_raw_clipped * _rates["APPROVAL_RATE"].fillna(0).values).fillna(0)
            results_df["ORIGINATIONS_Total"] = (_raw_clipped * _rates["ORIGINATION_RATE"].fillna(0).values).fillna(0)

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
        _pred_denom = _monthly_pred["Allocated_Predicted_APPS"].replace(0, np.nan)
        if "Allocated_Approved" in _monthly_pred.columns:
            _appr_rate = (_monthly_pred["Allocated_Approved"] / _pred_denom).fillna(0)
            _monthly_pred["Baseline_Approved_Rounded"]     = (_monthly_pred["Baseline APPS"]     * _appr_rate).round().astype("Int64")
            _monthly_pred["Incremental_Approved_Rounded"]  = (_monthly_pred["Incremental APPS"]  * _appr_rate).round().astype("Int64")
        if "Allocated_Originations" in _monthly_pred.columns:
            _orig_rate = (_monthly_pred["Allocated_Originations"] / _pred_denom).fillna(0)
            _monthly_pred["Baseline_Originations_Rounded"]    = (_monthly_pred["Baseline APPS"]    * _orig_rate).round().astype("Int64")
            _monthly_pred["Incremental_Originations_Rounded"] = (_monthly_pred["Incremental APPS"] * _orig_rate).round().astype("Int64")
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

        _mf1, _mf2, _mf3, _mf4, _mf5 = st.columns(5)

        # State — multiselect (does not affect model grain)
        _sel_m_st = _mf1.multiselect(
            "Filter by State",
            sorted(monthly_df["State"].dropna().unique().tolist()),
            key="monthly_filter_state",
            placeholder="All states",
        )

        # Month — multiselect, scoped to selected states
        _m_mo_base   = monthly_df if not _sel_m_st else monthly_df[monthly_df["State"].isin(_sel_m_st)]
        _m_mo_nums   = sorted(_m_mo_base["Calendar_Month"].dropna().unique().astype(int).tolist())
        _m_mo_labels = [_MONTH_NAME.get(m, str(m)) for m in _m_mo_nums]
        _m_mo_map    = dict(zip(_m_mo_labels, _m_mo_nums))
        _sel_m_mo = _mf2.multiselect(
            "Filter by Month",
            _m_mo_labels,
            key="monthly_filter_month",
            placeholder="All months",
        )

        # Channel / H_Tactic / Detail_Tactic — multiselects; "--Default--" represents null (state-only grain).
        _ch_opts = ["--Default--"] + (
            sorted(monthly_df["Channel"].dropna().unique().tolist())
            if "Channel" in monthly_df.columns else []
        )
        _ht_opts = ["--Default--"] + (
            sorted(monthly_df["H_Tactic"].dropna().unique().tolist())
            if "H_Tactic" in monthly_df.columns else []
        )
        _dt_opts = ["--Default--"] + (
            sorted(monthly_df["Detail_Tactic"].dropna().unique().tolist())
            if "Detail_Tactic" in monthly_df.columns else []
        )

        _sel_m_ch = _mf3.multiselect("Channel",       _ch_opts, default=["--Default--"], key="monthly_filter_channel",       placeholder="All")
        _sel_m_ht = _mf4.multiselect("H_Tactic",      _ht_opts, default=["--Default--"], key="monthly_filter_h_tactic",      placeholder="All")
        _sel_m_dt = _mf5.multiselect("Detail_Tactic", _dt_opts, default=["--Default--"], key="monthly_filter_detail_tactic", placeholder="All")

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

        # Product filter
        _pf_data = st.session_state.product_factors_df
        _prod_opts = (
            sorted(_pf_data["PRODUCT_FUNDED"].dropna().unique().tolist())
            if _pf_data is not None and not _pf_data.empty else []
        )
        if _prod_opts:
            _pf_col, _ = st.columns([2, 3])
            _sel_prod = _pf_col.multiselect(
                "Filter by Product",
                _prod_opts,
                key="monthly_filter_product",
                placeholder="All products",
            )
        else:
            _sel_prod = []

        # Apply filters
        m_display = monthly_df.copy()
        if _sel_m_st: m_display = m_display[m_display["State"].isin(_sel_m_st)]
        if _sel_m_mo: m_display = m_display[m_display["Calendar_Month"].isin([_m_mo_map[m] for m in _sel_m_mo])]
        m_display = _apply_grain_filter(m_display, "Channel",       _sel_m_ch)
        m_display = _apply_grain_filter(m_display, "H_Tactic",      _sel_m_ht)
        m_display = _apply_grain_filter(m_display, "Detail_Tactic", _sel_m_dt)

        # Scale origination columns by APPLICATION_SHARE for selected product
        _orig_rounded_cols = [
            "Allocated_Originations_Rounded",
            "Baseline_Originations_Rounded",
            "Incremental_Originations_Rounded",
        ]
        if _sel_prod and _pf_data is not None and not _pf_data.empty:
            _pf_sel = (
                _pf_data[_pf_data["PRODUCT_FUNDED"].isin(_sel_prod)]
                .groupby("Key", as_index=False)["APPLICATION_SHARE"]
                .sum()
            )
            m_display = m_display.merge(_pf_sel, on="Key", how="left")
            _share = m_display["APPLICATION_SHARE"].fillna(0)
            for _oc in [c for c in _orig_rounded_cols if c in m_display.columns]:
                m_display[_oc] = (m_display[_oc].astype(float) * _share).round().astype("Int64")
            m_display = m_display.drop(columns=["APPLICATION_SHARE"])

        # ── APPS view selector ────────────────────────────────────────────────────
        _view_col, _ = st.columns([2, 3])
        _view = _view_col.radio(
            "APPS View",
            ["All", "Baseline", "Incremental"],
            horizontal=True,
            key="monthly_apps_view",
        )
        _apps_col_map = {
            "All":         "Allocated_Predicted_APPS_Rounded",
            "Baseline":    "Baseline_APPS_Rounded",
            "Incremental": "Incremental_APPS_Rounded",
        }
        _selected_apps_col = _apps_col_map[_view]

        # All 3 APPS cols are aggregated; only the selected one is shown
        _all_apps_cols = [
            "Allocated_Predicted_APPS_Rounded",
            "Baseline_APPS_Rounded",
            "Incremental_APPS_Rounded",
        ]
        _appr_col_map = {
            "All":         "Allocated_Approved_Rounded",
            "Baseline":    "Baseline_Approved_Rounded",
            "Incremental": "Incremental_Approved_Rounded",
        }
        _orig_col_map = {
            "All":         "Allocated_Originations_Rounded",
            "Baseline":    "Baseline_Originations_Rounded",
            "Incremental": "Incremental_Originations_Rounded",
        }
        _approval_col    = _appr_col_map[_view]
        _origination_col = _orig_col_map[_view]
        agg_cols = _all_apps_cols + list(_appr_col_map.values()) + list(_orig_col_map.values())

        # Aggregate filtered rows to one row per State × Calendar_Year × Calendar_Month
        agg_cols_present = [c for c in agg_cols if c in m_display.columns]
        m_display = (
            m_display
            .groupby(["State", "Calendar_Year", "Calendar_Month"], as_index=False)[agg_cols_present]
            .sum()
        )

        # Combine year + month into a single "Mon YYYY" label column
        m_display["Period"] = (
            m_display["Calendar_Month"].astype(int).map(_MONTH_NAME)
            + " "
            + m_display["Calendar_Year"].astype(int).astype(str)
        )

        # Display only the selected APPS col plus approvals/originations if available
        _display_apps_col  = _selected_apps_col if _selected_apps_col in agg_cols_present else None
        _has_approved      = _approval_col in agg_cols_present
        _has_originated    = _origination_col in agg_cols_present
        monthly_primary_cols = ["State", "Period"]
        if _display_apps_col:
            monthly_primary_cols.append(_display_apps_col)
        if _has_approved:
            monthly_primary_cols.append(_approval_col)
        if _has_originated:
            monthly_primary_cols.append(_origination_col)
        monthly_fmt = {c: "{:,}" for c in monthly_primary_cols if c not in ["State", "Period"]}

        if m_display.empty:
            st.info("No rows match the selected filters.")
        else:
            # ── Summary metrics ───────────────────────────────────────────────
            _apps_total  = int(m_display[_display_apps_col].sum()) if _display_apps_col else 0
            _appr_total  = int(m_display[_approval_col].sum())    if _has_approved else None
            _orig_total  = int(m_display[_origination_col].sum()) if _has_originated else None

            _n_metric_cols = 1 + (1 if _appr_total is not None else 0) + (1 if _orig_total is not None else 0)
            _mcols = st.columns(_n_metric_cols)
            _mcols[0].metric("Predicted Applications", f"{_apps_total:,}")
            if _appr_total is not None:
                _mcols[1].metric("Likely Approvals", f"{_appr_total:,}")
            if _orig_total is not None:
                _mcols[1 + (1 if _appr_total is not None else 0)].metric("Likely Originations", f"{_orig_total:,}")

            _col_rename = {}
            if _display_apps_col:
                _col_rename[_display_apps_col] = "Predicted Applications"
            if _has_approved:
                _col_rename[_approval_col] = "Likely Approvals"
            if _has_originated:
                _col_rename[_origination_col] = "Likely Originations"
            _display_slice = m_display[monthly_primary_cols].rename(columns=_col_rename)
            _display_fmt   = {_col_rename.get(c, c): fmt for c, fmt in monthly_fmt.items()}

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
