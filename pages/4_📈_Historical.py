"""Page 1: Historical Spend & Applications."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_df_from_spaces

st.set_page_config(
    page_title="Historical — Oracle",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
    html, body, .stApp, .stMarkdown, .stText,
    p, li, td, th, input, textarea, select { font-family: 'DM Sans', sans-serif !important; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
    .stApp { background-color: var(--background-color); }
    .block-container { padding-top: 2rem; }
    p, li, td, th, .stMarkdown p, .stMarkdown li,
    [data-testid="stWidgetLabel"] p { color: var(--text-color) !important; }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label { color: var(--text-color) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
_SPEND_TACTICS = ["DSP", "LeadGen", "Paid Search", "Paid Social", "Prescreen", "Referrals", "Sweepstakes"]
_APP_METRICS   = ["APPLICATIONS", "APPROVED", "ORIGINATIONS"]

_TACTIC_COLORS = {
    "DSP":          "#0ea5e9",
    "LeadGen":      "#f59e0b",
    "Paid Search":  "#10b981",
    "Paid Social":  "#f43f5e",
    "Prescreen":    "#8b5cf6",
    "Referrals":    "#14b8a6",
    "Sweepstakes":  "#fb923c",
}
_METRIC_COLORS = {
    "APPLICATIONS": "#e2e8f0",
    "APPROVED":     "#38bdf8",
    "ORIGINATIONS": "#4ade80",
}
_METRIC_LABELS = {
    "APPLICATIONS": "Applications",
    "APPROVED":     "Approved",
    "ORIGINATIONS": "Funded",
}

_DATA_PATH = Path(__file__).parent.parent / "historical_spend.csv"

_MONTH_ORDER = {
    f"{y}{m}": y * 100 + i
    for y in range(2020, 2030)
    for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
    )
}


def _process(df: pd.DataFrame) -> pd.DataFrame:
    for col in _SPEND_TACTICS + _APP_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["_sort_key"] = df["Month_Year"].map(_MONTH_ORDER).fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    return _process(pd.read_csv(_DATA_PATH))


@st.cache_data(show_spinner=False)
def _load_bytes(data: bytes, ext: str) -> pd.DataFrame:
    import io
    raw = pd.read_excel(io.BytesIO(data)) if ext in ("xlsx", "xls") else pd.read_csv(io.BytesIO(data))
    return _process(raw)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.2rem;margin-bottom:0;"
    "color:var(--text-color)'>📈 Historical</h1>"
    "<p style='color:var(--text-color);opacity:0.55;font-size:1rem;margin-top:0.2rem'>"
    "Marketing spend and application outcomes by month</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar: upload only ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data")
    uploaded = st.file_uploader(
        "Upload historical spend file",
        type=["csv", "xlsx", "xls"],
        key="hist_upload",
        help="CSV or Excel with Month_Year, STATE_CD, spend and metric columns",
    )

# ── Load ───────────────────────────────────────────────────────────────────────
if uploaded is not None:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    df = _load_bytes(uploaded.read(), ext)
elif _DATA_PATH.exists():
    df = _load()
else:
    _spaces_df, _spaces_err = load_df_from_spaces("SPACES_HIST_FILE", "historical_spend.csv")
    if _spaces_df is not None:
        df = _process(_spaces_df)
    else:
        st.error(
            "No historical data loaded. "
            + (f"Spaces error: {_spaces_err} — " if _spaces_err else "")
            + "Upload a file using the sidebar or place `historical_spend.csv` in the project folder."
        )
        st.stop()

# ── Inline filters ─────────────────────────────────────────────────────────────
_fc1, _fc2, _fc3, _fc4, _fc5 = st.columns(5)

def _ms(col_ctx, label: str, col: str) -> list:
    opts = sorted(df[col].dropna().unique())
    return col_ctx.multiselect(label, opts, default=[], key=f"hist_{col}", placeholder="All")

sel_state   = _ms(_fc1, "State",         "STATE_CD")
sel_channel = _ms(_fc2, "Channel",       "CHANNEL_CD")
sel_htactic = _ms(_fc3, "H Tactic",      "H_TACTIC")
sel_dtactic = _ms(_fc4, "Detail Tactic", "DETAIL_TACTIC")
sel_product = _ms(_fc5, "Product",       "PRODUCT_FUNDED")

# ── Aggregate spend (dedup to one row per STATE × Month, then sum across states) ──────
spend_base = df[df["STATE_CD"].isin(sel_state)] if sel_state else df
spend_agg = (
    spend_base.drop_duplicates(subset=["Month_Year", "STATE_CD", "_sort_key"])
    .groupby(["Month_Year", "_sort_key"], as_index=False)[_SPEND_TACTICS]
    .sum()
    .sort_values("_sort_key")
)

# ── Aggregate apps (filtered by all dimensions) ────────────────────────────────
filtered = df.copy()
if sel_state:
    filtered = filtered[filtered["STATE_CD"].isin(sel_state)]
if sel_channel:
    filtered = filtered[filtered["CHANNEL_CD"].isin(sel_channel)]
if sel_htactic:
    filtered = filtered[filtered["H_TACTIC"].isin(sel_htactic)]
if sel_dtactic:
    filtered = filtered[filtered["DETAIL_TACTIC"].isin(sel_dtactic)]
if sel_product:
    filtered = filtered[filtered["PRODUCT_FUNDED"].isin(sel_product)]

apps_agg = (
    filtered.groupby(["Month_Year", "_sort_key"], as_index=False)[_APP_METRICS]
    .sum()
    .sort_values("_sort_key")
)

# Merge on Month_Year for aligned x axis
chart_df = spend_agg.merge(apps_agg, on=["Month_Year", "_sort_key"], how="left").sort_values("_sort_key")

chart_df["_total_spend"] = chart_df[_SPEND_TACTICS].sum(axis=1)

# ── Y-axis ranges locked to state selection (ignore sub-filters) ───────────────
_state_apps = (
    (df[df["STATE_CD"].isin(sel_state)] if sel_state else df)
    .groupby(["Month_Year", "_sort_key"])[_APP_METRICS]
    .sum()
)
_y_metric_max = max(_state_apps.max().max() * 1.1, 1)
_y_spend_max  = max(
    spend_agg[_SPEND_TACTICS].sum(axis=1).max() * 1.1, 1
)

x_labels = chart_df["Month_Year"].tolist()

# ── Build chart ────────────────────────────────────────────────────────────────
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Stacked bars — spend by tactic (primary/left axis; renders underneath secondary layer)
for tactic in _SPEND_TACTICS:
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=chart_df[tactic],
            name=tactic,
            marker_color=_TACTIC_COLORS[tactic],
            legendgroup="spend",
            legendgrouptitle_text="Spend",
            hovertemplate=f"<b>{tactic}</b>: $%{{y:,.1f}}<extra></extra>",
        ),
        secondary_y=False,
    )

# Invisible trace for total spend — appears at bottom of unified hover
fig.add_trace(
    go.Scatter(
        x=x_labels,
        y=chart_df["_total_spend"],
        name="Total Spend",
        mode="markers",
        marker=dict(opacity=0, size=0),
        showlegend=False,
        hovertemplate="<b>Total Spend</b>: $%{y:,.1f}<extra></extra>",
    ),
    secondary_y=False,
)

# Lines — application metrics (secondary axis; renders on top of bars)
for metric in _APP_METRICS:
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=chart_df[metric],
            name=_METRIC_LABELS[metric],
            mode="lines+markers",
            line=dict(color=_METRIC_COLORS[metric], width=2),
            marker=dict(size=5),
            legendgroup="metrics",
            legendgrouptitle_text="Metrics",
            hovertemplate=f"<b>{_METRIC_LABELS[metric]}</b>: %{{y:,.0f}}<extra></extra>",
        ),
        secondary_y=True,
    )

fig.update_layout(
    barmode="stack",
    height=540,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.18,
        xanchor="left",
        x=0,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    margin=dict(t=20, b=140, l=60, r=60),
    xaxis=dict(
        tickangle=-45,
        showgrid=False,
        tickfont=dict(size=10),
    ),
    font=dict(family="DM Sans"),
    hovermode="x unified",
)
fig.update_yaxes(
    title_text="Spend ($)",
    secondary_y=False,
    side="left",
    range=[0, _y_spend_max],
    showgrid=False,
    zeroline=False,
)
fig.update_yaxes(
    title_text="Applications / Approved / Funded",
    secondary_y=True,
    side="right",
    range=[0, _y_metric_max],
    showgrid=True,
    gridcolor="rgba(148,163,184,0.15)",
    zeroline=False,
)

st.plotly_chart(fig, use_container_width=True)
