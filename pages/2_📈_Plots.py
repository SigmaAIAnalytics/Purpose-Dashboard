"""Page 2: Historical actuals vs. forecast line chart."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_df_from_spaces, render_comments_section

st.set_page_config(
    page_title="Plots — Purpose Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS (matches app.py) ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, .stApp, .stMarkdown, .stText,
    p, li, td, th, input, textarea, select {
        font-family: 'DM Sans', sans-serif !important;
    }
    h1, h2, h3, .section-header {
        font-family: 'DM Serif Display', serif !important;
    }
    .stApp { background-color: var(--background-color); }
    .block-container { padding-top: 2rem; }
    p, li, td, th,
    .stMarkdown p, .stMarkdown li,
    [data-testid="stWidgetLabel"] p {
        color: var(--text-color) !important;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: var(--text-color) !important;
    }
    .section-header {
        font-size: 1.35rem;
        color: var(--text-color) !important;
        margin-bottom: 0.25rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(148, 163, 184, 0.4);
    }
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
    details summary,
    details summary p,
    .streamlit-expanderHeader p {
        color: var(--text-color) !important;
    }
    [data-testid="stDataFrame"] * {
        color: var(--text-color) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "hist_df"     not in st.session_state: st.session_state.hist_df     = None
if "hist_source" not in st.session_state: st.session_state.hist_source = None
if "hist_error"  not in st.session_state: st.session_state.hist_error  = None

# ── Auto-load from Spaces ─────────────────────────────────────────────────────
if st.session_state.hist_df is None:
    _df, _err = load_df_from_spaces("SPACES_HIST_FILE", "historical_forecast.csv")
    if _df is not None:
        st.session_state.hist_df     = _df
        st.session_state.hist_source = "spaces"
    elif _err:
        st.session_state.hist_error  = _err

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Data File")

    if st.session_state.hist_source == "spaces":
        st.success(f"✅ Auto-loaded from Spaces — {len(st.session_state.hist_df):,} rows")
        with st.expander("Override with a local file"):
            _ov = st.file_uploader(
                "Upload historical_forecast.csv",
                type=["csv"],
                key="hist_uploader",
            )
            if _ov:
                try:
                    st.session_state.hist_df     = pd.read_csv(_ov)
                    st.session_state.hist_source = "upload"
                    st.success(f"✅ Overridden — {len(st.session_state.hist_df):,} rows")
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
    else:
        st.markdown(
            "Upload `historical_forecast.csv` produced by `build_historical_forecast.py`, "
            "or set the `SPACES_HIST_FILE` env var to auto-load from Spaces."
        )
        _up = st.file_uploader(
            "Upload historical_forecast.csv",
            type=["csv"],
            key="hist_uploader",
        )
        if _up:
            try:
                st.session_state.hist_df     = pd.read_csv(_up)
                st.session_state.hist_source = "upload"
                st.success(f"✅ Loaded — {len(st.session_state.hist_df):,} rows")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
        elif st.session_state.hist_df is not None:
            st.success(f"✅ Loaded — {len(st.session_state.hist_df):,} rows")
        else:
            if st.session_state.hist_error:
                st.error(f"Spaces error: {st.session_state.hist_error}")
            else:
                st.info("No file loaded.")

    st.markdown("---")
    with st.expander("🔧 Spaces diagnostics"):
        region = os.environ.get("SPACES_REGION", "").lower().strip()
        bucket = os.environ.get("SPACES_BUCKET", "")
        st.markdown(f"**Region:** `{region or '(not set)'}`")
        st.markdown(f"**Bucket:** `{bucket or '(not set)'}`")
        st.markdown(f"**SPACES_KEY set:** `{'yes' if os.environ.get('SPACES_KEY') else 'no'}`")
        st.markdown(f"**SPACES_SECRET set:** `{'yes' if os.environ.get('SPACES_SECRET') else 'no'}`")
        st.markdown(f"**SPACES_HIST_FILE:** `{os.environ.get('SPACES_HIST_FILE', '(default: historical_forecast.csv)')}`")

    st.markdown(
        "<small style='color:var(--text-color);opacity:0.5'>"
        "Purpose Predictor v1.0<br>Historical actuals + model forecast</small>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.1rem;margin-bottom:0;"
    "color:var(--text-color)'>Historical & Forecast</h1>"
    "<p style='color:var(--text-color);opacity:0.55;margin-top:0.1rem'>"
    "Actuals vs. predictions by state, channel, and product</p>",
    unsafe_allow_html=True,
)
st.divider()

if st.session_state.hist_df is None:
    st.info("Upload `historical_forecast.csv` using the sidebar to get started.")
    render_comments_section("Historical and Forecast")
    st.stop()

# ── Coerce types ──────────────────────────────────────────────────────────────
df = st.session_state.hist_df.copy()

for col in ["ISO_Year", "ISO_Week"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["ISO_Year", "ISO_Week"])
df["ISO_Year"] = df["ISO_Year"].astype(int)
df["ISO_Week"] = df["ISO_Week"].astype(int)

for col in ["Applications", "Approvals", "Originations"]:
    df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)

_OVERALL_COLS = {"Channel", "H_Tactic", "Detail_Tactic"}

for col in ["State", "Channel", "H_Tactic", "Detail_Tactic", "Product_Funded", "Type"]:
    if col in df.columns:
        # Normalise to clean strings; convert any residual nulls in the
        # dimension columns to "Overall" (build script does this too, but
        # guard here as well in case of manually-uploaded files)
        df[col] = df[col].fillna("Overall" if col in _OVERALL_COLS else "")
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"nan": "Overall" if col in _OVERALL_COLS else "",
                      "None": "Overall" if col in _OVERALL_COLS else ""}
            )
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
_BLANKS = {""}

def _opts(series: pd.Series) -> list[str]:
    return sorted(v for v in series.dropna().unique() if str(v) not in _BLANKS)

def _default_idx(opts: list[str], preferred: str = "Overall") -> int:
    return opts.index(preferred) if preferred in opts else 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Filters + chart
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='section-header'>📈 Applications · Approvals · Originations — Actuals + Forecast</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Cascading filters ─────────────────────────────────────────────────────────
_ff1, _ff2, _ff3, _ff4, _ff5 = st.columns(5)

_st_opts = ["All"] + _opts(df["State"])
_sel_st  = _ff1.selectbox("State", _st_opts, key="hf_state")

_ch_base = df if _sel_st == "All" else df[df["State"] == _sel_st]
_ch_opts = ["All"] + _opts(_ch_base["Channel"])
_sel_ch  = _ff2.selectbox("Channel", _ch_opts, index=_default_idx(_ch_opts), key="hf_channel")

_ht_base = _ch_base if _sel_ch == "All" else _ch_base[_ch_base["Channel"] == _sel_ch]
_ht_opts = ["All"] + _opts(_ht_base["H_Tactic"])
_sel_ht  = _ff3.selectbox("H_Tactic", _ht_opts, index=_default_idx(_ht_opts), key="hf_h_tactic")

_dt_base = _ht_base if _sel_ht == "All" else _ht_base[_ht_base["H_Tactic"] == _sel_ht]
_dt_opts = ["All"] + _opts(_dt_base["Detail_Tactic"])
_sel_dt  = _ff4.selectbox("Detail_Tactic", _dt_opts, index=_default_idx(_dt_opts), key="hf_detail_tactic")

_pf_base = _dt_base if _sel_dt == "All" else _dt_base[_dt_base["Detail_Tactic"] == _sel_dt]
_sel_pf  = _ff5.selectbox("Product Funded", ["All"] + _opts(_pf_base["Product_Funded"]), key="hf_product")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if _sel_st != "All": filtered = filtered[filtered["State"]          == _sel_st]
if _sel_ch != "All": filtered = filtered[filtered["Channel"]        == _sel_ch]
if _sel_ht != "All": filtered = filtered[filtered["H_Tactic"]       == _sel_ht]
if _sel_dt != "All": filtered = filtered[filtered["Detail_Tactic"]  == _sel_dt]
if _sel_pf != "All": filtered = filtered[filtered["Product_Funded"] == _sel_pf]

# ── Aggregate to Year-Month-Type ──────────────────────────────────────────────
if filtered.empty:
    st.info("No rows match the selected filters.")
else:
    chart_df = (
        filtered
        .groupby(["ISO_Year", "ISO_Week", "Type"], as_index=False)
        [["Applications", "Approvals", "Originations"]]
        .sum()
    )
    chart_df["Period"] = chart_df.apply(
        lambda r: f"W{int(r['ISO_Week'])} {int(r['ISO_Year'])}",
        axis=1,
    )
    chart_df["_sort"] = chart_df["ISO_Year"] * 100 + chart_df["ISO_Week"]
    chart_df = chart_df.sort_values("_sort").reset_index(drop=True)

    actual_df   = chart_df[chart_df["Type"] == "Actual"].reset_index(drop=True)
    forecast_df = chart_df[chart_df["Type"] == "Forecast"].reset_index(drop=True)

    # ── Plotly chart ──────────────────────────────────────────────────────────
    METRIC_COLOR = {
        "Applications": "#0369a1",
        "Approvals":    "#0f766e",
        "Originations": "#7c3aed",
    }

    fig = go.Figure()

    for metric, color in METRIC_COLOR.items():
        has_actual   = not actual_df.empty   and actual_df[metric].sum()   > 0
        has_forecast = not forecast_df.empty and forecast_df[metric].sum() > 0

        if not has_actual and not has_forecast:
            continue

        # Actual trace — solid line
        if has_actual:
            fig.add_trace(go.Scatter(
                x=actual_df["Period"],
                y=actual_df[metric],
                name=f"Actual {metric}",
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                hovertemplate=f"<b>%{{x}}</b><br>Actual {metric}: %{{y:,.0f}}<extra></extra>",
            ))

        # Forecast trace — dotted, bridged from last actual point
        if has_forecast:
            if has_actual:
                bridge_x = [actual_df["Period"].iloc[-1]] + forecast_df["Period"].tolist()
                bridge_y = [float(actual_df[metric].iloc[-1])] + forecast_df[metric].tolist()
            else:
                bridge_x = forecast_df["Period"].tolist()
                bridge_y = forecast_df[metric].tolist()

            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=bridge_y,
                name=f"Forecast {metric}",
                mode="lines+markers",
                line=dict(color=color, width=2.5, dash="dot"),
                marker=dict(size=5, symbol="circle", color=color,
                            line=dict(color="#ffffff", width=1)),
                hovertemplate=f"<b>%{{x}}</b><br>Forecast {metric}: %{{y:,.0f}}<extra></extra>",
            ))

    # Vertical separator at actual/forecast boundary
    if not actual_df.empty and not forecast_df.empty:
        _boundary = actual_df["Period"].iloc[-1]
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=_boundary, x1=_boundary, y0=0, y1=1,
            line=dict(dash="dot", color="#94a3b8", width=1),
        )
        fig.add_annotation(
            xref="x", yref="paper",
            x=_boundary, y=1.03,
            text="◀ Actual   Forecast ▶",
            showarrow=False,
            font=dict(color="#64748b", size=10),
            xanchor="center",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial", size=12, color="#94a3b8"),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.15)",
            linecolor="#94a3b8",
            tickangle=-35,
            tickfont=dict(color="#94a3b8", size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.15)",
            linecolor="#94a3b8",
            rangemode="tozero",
            tickfont=dict(color="#94a3b8", size=11),
            title="Count",
            title_font=dict(color="#94a3b8", size=12),
            tickformat=",",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
            font=dict(color="#94a3b8", size=11),
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="left",   x=0,
        ),
        height=460,
        margin=dict(l=60, r=20, t=80, b=70),
        hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9", font_size=12),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Underlying data table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View underlying data"):
        _metrics_present = [m for m in ["Applications", "Approvals", "Originations"]
                            if chart_df[m].sum() > 0]
        tbl = (
            chart_df[["Period", "Type", "_sort"] + _metrics_present]
            .sort_values(["_sort", "Type"])
            .drop(columns=["_sort"])
        )
        fmt = {m: "{:,.0f}" for m in _metrics_present}
        st.dataframe(
            tbl.style.format(fmt, na_rep=""),
            use_container_width=True,
            hide_index=True,
        )


# ── Comments ──────────────────────────────────────────────────────────────────
render_comments_section("Historical and Forecast")
