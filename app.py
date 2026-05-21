from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from build_state_division_models import spread_monthly_spend_to_weekly

st.set_page_config(
    page_title="Purpose Predictor — Application Calculator",
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
        "raw_prediction":               round(prediction, 6),
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
                "State": state, "ISO Year": iso_year, "ISO Week": iso_week,
                "Month": month,
                **{c: row[c] for c in spend_cols},
                "Channel": None, "H_Tactic": None,
                "Detail_Tactic": None, "Product": None,
                "Predicted APPS": None,
                "raw_prediction": None,
                "95% Confidence Lower Limit": None,
                "95% Confidence Upper Limit": None,
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
                "State": state, "ISO Year": iso_year, "ISO Week": iso_week,
                "Month": month,
                **{c: row[c] for c in spend_cols},
                "Channel":       channel,
                "H_Tactic":      h_tactic,
                "Detail_Tactic": detail_tactic,
                "Product":       None,
                **scored,
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
        ["State", "ISO Year", "ISO Week", "_grain", "_ch", "_ht", "_dt"],
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
if "input_snap"        not in st.session_state: st.session_state.input_snap        = None
if "coeff_df"          not in st.session_state: st.session_state.coeff_df          = None
if "product_factors_df"not in st.session_state: st.session_state.product_factors_df= None
if "upload_df"         not in st.session_state: st.session_state.upload_df         = None
if "upload_version"    not in st.session_state: st.session_state.upload_version    = 0
if "last_input_name"   not in st.session_state: st.session_state.last_input_name   = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Coefficient file uploader
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Model Coefficients")
    st.markdown("Upload the `MODEL_Coefficients` sheet from the Excel workbook.")

    coeff_file = st.file_uploader(
        "Upload Model Coefficients",
        type=["csv", "xlsx"],
        key="coeff_uploader",
    )

    if coeff_file:
        try:
            if coeff_file.name.endswith(".csv"):
                coeff_df = pd.read_csv(coeff_file)
            else:
                xl = pd.ExcelFile(coeff_file)
                sheet = (
                    "MODEL_Coefficients"
                    if "MODEL_Coefficients" in xl.sheet_names
                    else xl.sheet_names[0]
                )
                coeff_df = xl.parse(sheet)

            st.session_state.coeff_df = coeff_df
            keys = coeff_df["Key"].dropna().tolist() if "Key" in coeff_df.columns else []
            st.success(f"✅ Coefficients loaded — {len(keys)} model keys found")

            with st.expander("Available state keys"):
                st.write(keys)

        except Exception as e:
            st.error(f"Failed to read coefficient file: {e}")

    else:
        st.info("No file uploaded yet.")

    st.markdown("---")
    st.markdown("## 📦 Product Factors")
    st.markdown("Upload the `product_factors.csv` produced alongside the coefficients file.")

    pf_file = st.file_uploader(
        "Upload Product Factors",
        type=["csv"],
        key="pf_uploader",
    )

    if pf_file:
        try:
            pf_df = pd.read_csv(pf_file)
            required_pf = {"Key", "PRODUCT_FUNDED", "APPLICATION_SHARE", "APPROVAL_RATE", "ORIGINATION_RATE"}
            missing_pf  = required_pf - set(pf_df.columns)
            if missing_pf:
                st.error(f"Missing columns: {', '.join(sorted(missing_pf))}")
            else:
                st.session_state.product_factors_df = pf_df
                products = pf_df["PRODUCT_FUNDED"].dropna().unique().tolist()
                st.success(f"✅ Product factors loaded — {len(products)} product(s): {', '.join(sorted(products))}")
        except Exception as e:
            st.error(f"Failed to read product factors file: {e}")
    else:
        st.info("No file uploaded yet.")

    st.markdown("---")
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

# ── Column aliases for flexible upload parsing ────────────────────────────────
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
        st.error("⚠️ Please upload a coefficient file in the sidebar first.")
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
            baseline_df[["State", "ISO Year", "ISO Week", "Model_Key", "Predicted APPS"]]
            .rename(columns={"Predicted APPS": "Baseline APPS"})
        )
        results_df = results_df.merge(
            baseline_lookup,
            on=["State", "ISO Year", "ISO Week", "Model_Key"],
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
            results_df["Allocated_Approved"]     = 0.0
            results_df["Allocated_Originations"] = 0.0
            for product, grp in pf.groupby("PRODUCT_FUNDED", dropna=False):
                pkey    = _safe_col(product)
                factors = results_df[["Model_Key"]].merge(
                    grp[["Key", "APPLICATION_SHARE", "APPROVAL_RATE", "ORIGINATION_RATE"]],
                    left_on="Model_Key", right_on="Key", how="left",
                )
                apps   = results_df["raw_prediction"].clip(lower=0) * factors["APPLICATION_SHARE"].fillna(0).values
                approv = apps * factors["APPROVAL_RATE"].fillna(0).values
                orig   = apps * factors["ORIGINATION_RATE"].fillna(0).values
                results_df[f"Applications_{pkey}"] = apps.round().astype(int)
                results_df[f"Approvals_{pkey}"]    = approv.round().astype(int)
                results_df[f"Originations_{pkey}"] = orig.round().astype(int)
                results_df["Allocated_Approved"]     += approv.fillna(0)
                results_df["Allocated_Originations"] += orig.fillna(0)
            results_df["Allocated_Approved"]     = results_df["Allocated_Approved"].round().astype(int)
            results_df["Allocated_Originations"] = results_df["Allocated_Originations"].round().astype(int)

    st.session_state.results_df = results_df
    st.session_state.input_snap = valid_rows.copy()


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

    # ── Warnings for missing coefficients ─────────────────────────────────────
    if not fail_rows.empty:
        for _, r in fail_rows.iterrows():
            st.warning(
                f"⚠️ No coefficient found for state **{r['State']}** "
                f"(Week {r['ISO Week']}) — skipped."
            )

    if not ok_rows.empty:
        st.success(
            f"✅ {len(ok_rows)} prediction row(s) across "
            f"{ok_rows[['State','ISO Year','ISO Week']].drop_duplicates().shape[0]} "
            "state-week combination(s)"
        )

        # ── Output filters ────────────────────────────────────────────────────
        _ff1, _ff2, _ff3, _ff4, _ff5 = st.columns(5)

        _MONTH_NAME = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

        # State (independent)
        _st_opts = ["All"] + sorted(ok_rows["State"].dropna().unique().tolist())
        _sel_st  = _ff1.selectbox("Filter by State", _st_opts, key="filter_state")

        # Month (scoped to State)
        _mo_base  = ok_rows if _sel_st == "All" else ok_rows[ok_rows["State"] == _sel_st]
        _mo_nums  = sorted(_mo_base["Month"].dropna().unique().astype(int).tolist())
        _mo_labels= [_MONTH_NAME.get(m, str(m)) for m in _mo_nums]
        _mo_map   = dict(zip(_mo_labels, _mo_nums))
        _sel_mo   = _ff2.selectbox("Filter by Month", ["All"] + _mo_labels, key="filter_month")

        # Channel (scoped to State + Month)
        _ch_base = _mo_base if _sel_mo == "All" else _mo_base[_mo_base["Month"] == _mo_map[_sel_mo]]
        _ch_opts = ["All"] + sorted(_ch_base["Channel"].dropna().unique().tolist())
        _sel_ch  = _ff3.selectbox("Filter by Channel", _ch_opts, key="filter_channel")

        # H_Tactic (scoped to Channel)
        _ht_base = _ch_base if _sel_ch == "All" else _ch_base[_ch_base["Channel"] == _sel_ch]
        _ht_opts = ["All"] + sorted(_ht_base["H_Tactic"].dropna().unique().tolist())
        _sel_ht  = _ff4.selectbox("Filter by H_Tactic", _ht_opts, key="filter_h_tactic")

        # Detail_Tactic (scoped to H_Tactic)
        _dt_base = _ht_base if _sel_ht == "All" else _ht_base[_ht_base["H_Tactic"] == _sel_ht]
        _dt_opts = ["All"] + sorted(_dt_base["Detail_Tactic"].dropna().unique().tolist())
        _sel_dt  = _ff5.selectbox("Filter by Detail_Tactic", _dt_opts, key="filter_detail_tactic")

        # Product (only shown if product factors file is loaded)
        _sel_prod = "All"
        if st.session_state.product_factors_df is not None:
            _prod_opts = ["All"] + sorted(
                st.session_state.product_factors_df["PRODUCT_FUNDED"].dropna().astype(str).unique().tolist()
            )
            _prod_col, _ = st.columns([1, 4])
            _sel_prod = _prod_col.selectbox("Filter by Product", _prod_opts, key="filter_product")

        # Apply row filters
        display_df = results_df.copy()
        if _sel_st != "All":
            display_df = display_df[display_df["State"] == _sel_st]
        if _sel_mo != "All":
            display_df = display_df[display_df["Month"] == _mo_map[_sel_mo]]
        if _sel_ch != "All":
            display_df = display_df[display_df["Channel"] == _sel_ch]
        if _sel_ht != "All":
            display_df = display_df[display_df["H_Tactic"] == _sel_ht]
        if _sel_dt != "All":
            display_df = display_df[display_df["Detail_Tactic"] == _sel_dt]

        # ── Primary output table ──────────────────────────────────────────────
        primary_cols = [
            "State", "ISO Year", "ISO Week", "Month",
            *SPEND_COLUMNS,
            "Channel", "H_Tactic", "Detail_Tactic", "Product",
            "Predicted APPS", "Baseline APPS", "Incremental APPS",
        ]
        _prod_format: dict = {}
        if st.session_state.product_factors_df is not None:
            if _sel_prod == "All":
                primary_cols += ["Allocated_Approved", "Allocated_Originations"]
                _prod_format  = {"Allocated_Approved": "{:,}", "Allocated_Originations": "{:,}"}
            else:
                _pkey = _safe_col(_sel_prod)
                primary_cols += [f"Applications_{_pkey}", f"Approvals_{_pkey}", f"Originations_{_pkey}"]
                _prod_format  = {
                    f"Applications_{_pkey}": "{:,}",
                    f"Approvals_{_pkey}":    "{:,}",
                    f"Originations_{_pkey}": "{:,}",
                }
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
        "State", "ISO Year", "ISO Week", "Month",
        *SPEND_COLUMNS,
        "Channel", "H_Tactic", "Detail_Tactic", "Product",
        "Predicted APPS", "Baseline APPS", "Incremental APPS",
        "Allocated_Approved", "Allocated_Originations", "raw_prediction",
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
