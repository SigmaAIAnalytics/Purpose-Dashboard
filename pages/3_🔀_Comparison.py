"""Page 3: Scenario Comparison — multi-scenario APPS / Approvals / Originations."""
from __future__ import annotations

import calendar as _calendar
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Scenario Comparison — Oracle",
    page_icon="🔀",
    layout="wide",
)

# ── CSS (matches app_v2_multi.py) ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
    html, body, .stApp, .stMarkdown, .stText,
    p, li, td, th, input, textarea, select { font-family: 'DM Sans', sans-serif !important; }
    h1, h2, h3, .section-header { font-family: 'DM Serif Display', serif !important; }
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

# ── Helpers ────────────────────────────────────────────────────────────────────
_MONTH_NAME = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

_SCENARIO_COLORS = ["#0ea5e9", "#f59e0b", "#10b981", "#f43f5e"]


def _full_month_filter(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows belonging to months fully covered by the forecast."""
    if monthly_df is None or monthly_df.empty:
        return monthly_df
    _cov = monthly_df.groupby(
        ["State", "Calendar_Year", "Calendar_Month"], as_index=False
    )["Allocated_Days"].max()
    _cov["_days_in_month"] = _cov.apply(
        lambda r: _calendar.monthrange(int(r["Calendar_Year"]), int(r["Calendar_Month"]))[1],
        axis=1,
    )
    _full = _cov[_cov["Allocated_Days"] >= _cov["_days_in_month"]][
        ["State", "Calendar_Year", "Calendar_Month"]
    ]
    return monthly_df.merge(_full, on=["State", "Calendar_Year", "Calendar_Month"], how="inner")


def _apply_grain_filter(df: pd.DataFrame, col: str, selected: list) -> pd.DataFrame:
    if col not in df.columns or not selected:
        return df
    if "--Default--" in selected:
        mask = df[col].isna()
        others = [v for v in selected if v != "--Default--"]
        if others:
            mask = mask | df[col].isin(others)
        return df[mask]
    return df[df[col].isin(selected)]


# ── Guard: session state must exist ───────────────────────────────────────────
if "scenarios" not in st.session_state:
    st.info("Open the Oracle page first to initialise the app.")
    st.stop()

_active = [sc for sc in st.session_state.scenarios if sc.get("results_df") is not None]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.1rem;margin-bottom:0;"
    "color:var(--text-color)'>🔀 Scenario Comparison</h1>"
    "<p style='color:var(--text-color);opacity:0.55;margin-top:0.1rem'>"
    "Compare Applications, Approvals and Originations across scenarios</p>",
    unsafe_allow_html=True,
)
st.divider()

if not _active:
    st.info("No scenarios have been run yet. Go to Oracle and run at least the Baseline.")
    st.stop()

# ── Build combined frame for filter option discovery ───────────────────────────
_all_monthly = pd.concat(
    [_full_month_filter(sc["monthly_df"]) for sc in _active if sc.get("monthly_df") is not None],
    ignore_index=True,
)
if "Period" not in _all_monthly.columns and "Calendar_Month" in _all_monthly.columns:
    _all_monthly["Period"] = (
        _all_monthly["Calendar_Month"].astype(int).map(_MONTH_NAME)
        + " " + _all_monthly["Calendar_Year"].astype(int).astype(str)
    )

_state_opts = sorted(_all_monthly["State"].dropna().unique().tolist()) if "State" in _all_monthly.columns else []

_period_sort_key: dict[str, int] = {}
if "Calendar_Year" in _all_monthly.columns and "Calendar_Month" in _all_monthly.columns:
    for _, _r in _all_monthly[["Period", "Calendar_Year", "Calendar_Month"]].drop_duplicates().iterrows():
        _period_sort_key[_r["Period"]] = int(_r["Calendar_Year"]) * 100 + int(_r["Calendar_Month"])
_month_opts = sorted(_period_sort_key.keys(), key=lambda p: _period_sort_key[p])

_ch_opts = ["--Default--"] + (sorted(_all_monthly["Channel"].dropna().unique().tolist())       if "Channel"       in _all_monthly.columns else [])
_ht_opts = ["--Default--"] + (sorted(_all_monthly["H_Tactic"].dropna().unique().tolist())      if "H_Tactic"      in _all_monthly.columns else [])
_dt_opts = ["--Default--"] + (sorted(_all_monthly["Detail_Tactic"].dropna().unique().tolist()) if "Detail_Tactic" in _all_monthly.columns else [])

# ── Filter bar ─────────────────────────────────────────────────────────────────
_f1, _f2, _f3, _f4, _f5 = st.columns(5)
_sel_state = _f1.multiselect("State",         _state_opts, key="cmp_state",  placeholder="All")
_sel_month = _f2.multiselect("Month",         _month_opts, key="cmp_month",  placeholder="All")
_sel_ch    = _f3.multiselect("Channel",       _ch_opts, default=["--Default--"], key="cmp_channel")
_sel_ht    = _f4.multiselect("H_Tactic",      _ht_opts, default=["--Default--"], key="cmp_h_tactic")
_sel_dt    = _f5.multiselect("Detail_Tactic", _dt_opts, default=["--Default--"], key="cmp_detail_tactic")

_pf_data   = st.session_state.get("product_factors_df")
_prod_opts = (sorted(_pf_data["PRODUCT_FUNDED"].dropna().unique().tolist())
              if _pf_data is not None and not _pf_data.empty else [])

_sel_prod: list = []
if _prod_opts:
    _pf_col, _, _vw_col = st.columns([2, 2, 2])
    _sel_prod   = _pf_col.multiselect("Filter by Product", _prod_opts, key="cmp_product", placeholder="All products")
    _view_label = _vw_col.radio("APPS View", ["All", "No-Spend", "Incremental"], horizontal=True, key="cmp_view")
else:
    _, _vw_col  = st.columns([4, 2])
    _view_label = _vw_col.radio("APPS View", ["All", "No-Spend", "Incremental"], horizontal=True, key="cmp_view")

_view = "Baseline" if _view_label == "No-Spend" else _view_label

_apps_col_map = {"All": "Allocated_Predicted_APPS_Rounded", "Baseline": "Baseline_APPS_Rounded",    "Incremental": "Incremental_APPS_Rounded"}
_appr_col_map = {"All": "Allocated_Approved_Rounded",       "Baseline": "Baseline_Approved_Rounded", "Incremental": "Incremental_Approved_Rounded"}
_orig_col_map = {"All": "Allocated_Originations_Rounded",   "Baseline": "Baseline_Originations_Rounded", "Incremental": "Incremental_Originations_Rounded"}
_all_agg_cols = list(set(list(_apps_col_map.values()) + list(_appr_col_map.values()) + list(_orig_col_map.values())))

_selected_apps_col = _apps_col_map[_view]
_approval_col      = _appr_col_map[_view]
_origination_col   = _orig_col_map[_view]

# ── Filter + aggregate per scenario ───────────────────────────────────────────
_scene_agg: dict[str, pd.DataFrame] = {}

for _sc in _active:
    _mdf = _full_month_filter(_sc.get("monthly_df"))
    if _mdf is None or _mdf.empty:
        continue
    _mdf = _mdf.copy()
    if "Period" not in _mdf.columns and "Calendar_Month" in _mdf.columns:
        _mdf["Period"] = (
            _mdf["Calendar_Month"].astype(int).map(_MONTH_NAME)
            + " " + _mdf["Calendar_Year"].astype(int).astype(str)
        )
    if _sel_state:
        _mdf = _mdf[_mdf["State"].isin(_sel_state)]
    if _sel_month:
        _mdf = _mdf[_mdf["Period"].isin(_sel_month)]
    _mdf = _apply_grain_filter(_mdf, "Channel",       _sel_ch)
    _mdf = _apply_grain_filter(_mdf, "H_Tactic",      _sel_ht)
    _mdf = _apply_grain_filter(_mdf, "Detail_Tactic", _sel_dt)

    if _sel_prod and _pf_data is not None and not _pf_data.empty and "Key" in _mdf.columns:
        _pf_sel  = _pf_data[_pf_data["PRODUCT_FUNDED"].isin(_sel_prod)]
        _key_shr = _pf_sel.groupby("Key")["APPLICATION_SHARE"].sum().reset_index()
        _key_shr.columns = ["Key", "_ps"]
        _mdf = _mdf.merge(_key_shr, on="Key", how="inner")
        for _or in ["Allocated_Originations_Rounded", "Baseline_Originations_Rounded", "Incremental_Originations_Rounded"]:
            if _or in _mdf.columns:
                _mdf[_or] = (_mdf[_or].astype(float) * _mdf["_ps"]).round().astype("Int64")
        _mdf = _mdf.drop(columns=["_ps"])

    if _mdf.empty:
        continue

    _agg_cols_present = [c for c in _all_agg_cols if c in _mdf.columns]
    _agg = _mdf.groupby(["State", "Calendar_Year", "Calendar_Month"], as_index=False)[_agg_cols_present].sum()
    _agg["Period"] = (
        _agg["Calendar_Month"].astype(int).map(_MONTH_NAME)
        + " " + _agg["Calendar_Year"].astype(int).astype(str)
    )
    _scene_agg[_sc["name"]] = _agg

if not _scene_agg:
    st.info("No data matches the selected filters.")
    st.stop()

# ── Read rate overrides from session state and apply BEFORE rendering cards ────
# Pattern: read session state → apply to data → render cards → render inputs
_rate_meta: dict[str, dict] = {}

for _i, _sc in enumerate(_active):
    _agg = _scene_agg.get(_sc["name"])
    if _agg is None or _agg.empty:
        continue
    _apps_sum = int(_agg["Allocated_Predicted_APPS_Rounded"].sum()) if "Allocated_Predicted_APPS_Rounded" in _agg.columns else 0
    if _apps_sum == 0:
        continue

    _model_appr = _agg[_approval_col].sum()    / _apps_sum if _approval_col    in _agg.columns else 0.0
    _model_orig = _agg[_origination_col].sum() / _apps_sum if _origination_col in _agg.columns else 0.0

    _appr_rate = st.session_state.get(f"cmp_appr_rate_{_i}", round(_model_appr * 100)) / 100.0
    _orig_rate = st.session_state.get(f"cmp_orig_rate_{_i}", round(_model_orig * 100)) / 100.0

    for _ar, _appr_r in [
        ("Allocated_Predicted_APPS_Rounded", "Allocated_Approved_Rounded"),
        ("Baseline_APPS_Rounded",            "Baseline_Approved_Rounded"),
        ("Incremental_APPS_Rounded",         "Incremental_Approved_Rounded"),
    ]:
        if _ar in _agg.columns and _appr_r in _agg.columns:
            _agg[_appr_r] = (_agg[_ar].astype(float) * _appr_rate).round().astype("Int64")

    for _ar, _orig_r in [
        ("Allocated_Predicted_APPS_Rounded", "Allocated_Originations_Rounded"),
        ("Baseline_APPS_Rounded",            "Baseline_Originations_Rounded"),
        ("Incremental_APPS_Rounded",         "Incremental_Originations_Rounded"),
    ]:
        if _ar in _agg.columns and _orig_r in _agg.columns:
            _agg[_orig_r] = (_agg[_ar].astype(float) * _orig_rate).round().astype("Int64")

    _scene_agg[_sc["name"]] = _agg
    _rate_meta[_sc["name"]] = {
        "model_appr": _model_appr,
        "model_orig": _model_orig,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Metric cards (compact, one column per active scenario)
# ══════════════════════════════════════════════════════════════════════════════
_mc_cols = st.columns(len(_active))
for _ci, _sc in enumerate(_active):
    _agg   = _scene_agg.get(_sc["name"])
    _color = _SCENARIO_COLORS[_ci % len(_SCENARIO_COLORS)]
    with _mc_cols[_ci]:
        st.markdown(
            f"<div style='font-weight:600;font-size:0.95rem;margin-bottom:0.3rem;"
            f"color:{_color}'>{_sc['name']}</div>",
            unsafe_allow_html=True,
        )
        if _agg is None or _agg.empty:
            st.caption("— no data for current filters")
        else:
            _apps_t = int(_agg[_selected_apps_col].sum()) if _selected_apps_col in _agg.columns else None
            _appr_t = int(_agg[_approval_col].sum())      if _approval_col       in _agg.columns else None
            _orig_t = int(_agg[_origination_col].sum())   if _origination_col    in _agg.columns else None
            if _apps_t is not None: st.metric("Predicted Applications", f"{_apps_t:,}")
            if _appr_t is not None: st.metric("Likely Approvals",       f"{_appr_t:,}")
            if _orig_t is not None: st.metric("Likely Originations",    f"{_orig_t:,}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Rate override row
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("**Rate Overrides**")
_rc_cols = st.columns(len(_active))
for _ci, _sc in enumerate(_active):
    _rd = _rate_meta.get(_sc["name"])
    with _rc_cols[_ci]:
        if _rd is None:
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:600'>{_sc['name']}</div>",
                unsafe_allow_html=True,
            )
            st.caption("—")
        else:
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:600'>{_sc['name']}</div>"
                f"<div style='font-size:0.7rem;color:gray;margin-bottom:0.3rem'>"
                f"Model defaults — Appr: {round(_rd['model_appr'] * 100):.0f}% &nbsp;|&nbsp; "
                f"Orig: {round(_rd['model_orig'] * 100):.0f}%</div>",
                unsafe_allow_html=True,
            )
            st.number_input(
                "Approval Rate (%)", min_value=0.0, max_value=100.0,
                value=float(round(_rd["model_appr"] * 100)), step=1.0, format="%.0f",
                key=f"cmp_appr_rate_{_ci}",
            )
            st.number_input(
                "Origination Rate (%)", min_value=0.0, max_value=100.0,
                value=float(round(_rd["model_orig"] * 100)), step=1.0, format="%.0f",
                key=f"cmp_orig_rate_{_ci}",
            )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Charts — one per metric, shared Y scale, title on Y axis
# ══════════════════════════════════════════════════════════════════════════════

# Compute global Y max across all metrics and scenarios so all charts share the same scale
_ymax_vals = []
for _sc in _active:
    _agg = _scene_agg.get(_sc["name"])
    if _agg is None or _agg.empty:
        continue
    for _col in [_selected_apps_col, _approval_col, _origination_col]:
        if _col in _agg.columns:
            _period_max = _agg.groupby(["Calendar_Year", "Calendar_Month"])[_col].sum().max()
            if pd.notna(_period_max):
                _ymax_vals.append(float(_period_max))
_global_ymax = max(_ymax_vals) * 1.2 if _ymax_vals else None

# ── Monthly total spend per scenario (for secondary axis on Applications chart) ─
_spend_series: dict[str, pd.DataFrame] = {}
for _sc in _active:
    _snap = _sc.get("input_snap")
    if _snap is None or _snap.empty:
        continue
    _sp = _snap.copy()
    _sp["_date"] = pd.to_datetime(_sp["Date"], errors="coerce")
    _sp = _sp.dropna(subset=["_date"])
    _sp["Calendar_Year"]  = _sp["_date"].dt.year
    _sp["Calendar_Month"] = _sp["_date"].dt.month
    if _sel_state and "State" in _sp.columns:
        _sp = _sp[_sp["State"].isin(_sel_state)]
    _spend_cols = [c for c in _sp.columns
                   if c not in ("Date", "State", "_date", "Calendar_Year", "Calendar_Month")
                   and pd.api.types.is_numeric_dtype(_sp[c])]
    if not _spend_cols:
        continue
    _sp["_total"] = _sp[_spend_cols].sum(axis=1)
    _ms = _sp.groupby(["Calendar_Year", "Calendar_Month"])["_total"].sum().reset_index()
    _ms["Period"] = _ms["Calendar_Month"].astype(int).map(_MONTH_NAME) + " " + _ms["Calendar_Year"].astype(int).astype(str)
    _ms["_sort"] = _ms["Calendar_Year"].astype(int) * 100 + _ms["Calendar_Month"].astype(int)
    _ms = _ms.sort_values("_sort")
    if _sel_month:
        _ms = _ms[_ms["Period"].isin(_sel_month)]
    _spend_series[_sc["name"]] = _ms[["Period", "_total"]]


def _make_chart(
    title: str,
    col: str,
    y_max: float | None,
    spend_series: dict | None = None,
) -> go.Figure:
    fig = go.Figure()
    for _i, _sc in enumerate(_active):
        _agg = _scene_agg.get(_sc["name"])
        if _agg is None or _agg.empty or col not in _agg.columns:
            continue
        _ts = (
            _agg.groupby(["Calendar_Year", "Calendar_Month", "Period"])[col]
            .sum().reset_index()
        )
        _ts["_sort"] = _ts["Calendar_Year"].astype(int) * 100 + _ts["Calendar_Month"].astype(int)
        _ts = _ts.sort_values("_sort")
        _color = _SCENARIO_COLORS[_i % len(_SCENARIO_COLORS)]

        # Merge spend into customdata if available
        _sp = spend_series.get(_sc["name"]) if spend_series else None
        if _sp is not None and not _sp.empty:
            _ts = _ts.merge(_sp[["Period", "_total"]], on="Period", how="left")
            _customdata  = _ts["_total"].fillna(0).values
            _texttemplate = "%{y:,.0f}<br>$%{customdata:,.0f}"
        else:
            _customdata  = None
            _texttemplate = "%{y:,.0f}"

        fig.add_trace(go.Bar(
            x=_ts["Period"],
            y=_ts[col],
            name=_sc["name"],
            marker_color=_color,
            customdata=_customdata,
            texttemplate=_texttemplate,
            textposition="outside",
            textfont=dict(size=9, color="#94a3b8"),
            hovertemplate=(
                f"<b>%{{x}}</b><br>{_sc['name']}: %{{y:,.0f}}"
                + ("<br>Spend: $%{customdata:,.0f}" if _customdata is not None else "")
                + "<extra></extra>"
            ),
        ))

    _yaxis = dict(
        gridcolor="rgba(148,163,184,0.15)",
        linecolor="#94a3b8",
        tickfont=dict(color="#94a3b8", size=11),
        tickformat=",",
        title=dict(
            text=title,
            font=dict(family="DM Serif Display, serif", size=13, color="#94a3b8"),
            standoff=12,
        ),
    )
    if y_max is not None:
        _yaxis["range"] = [0, y_max]
    else:
        _yaxis["rangemode"] = "tozero"

    fig.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.12,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial", size=12, color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)", linecolor="#94a3b8",
                   tickangle=-35, tickfont=dict(color="#94a3b8", size=10)),
        yaxis=_yaxis,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(148,163,184,0.3)", borderwidth=1,
                    font=dict(color="#94a3b8", size=11), orientation="h",
                    yanchor="bottom", y=1.05, xanchor="left", x=0),
        height=320,
        margin=dict(l=90, r=20, t=40, b=70),
        hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9", font_size=12),
    )
    return fig


st.plotly_chart(_make_chart("Predicted Applications", _selected_apps_col, _global_ymax, _spend_series), use_container_width=True)
st.plotly_chart(_make_chart("Likely Approvals",       _approval_col,       _global_ymax), use_container_width=True)
st.plotly_chart(_make_chart("Likely Originations",    _origination_col,    _global_ymax), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Full comparison table + download
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("Full comparison table"):
    _parts = []
    for _sc in _active:
        _agg = _scene_agg.get(_sc["name"])
        if _agg is None or _agg.empty:
            continue
        _use_cols = [c for c in [_selected_apps_col, _approval_col, _origination_col] if c in _agg.columns]
        _ts = (
            _agg.groupby(["State", "Calendar_Year", "Calendar_Month", "Period"])[_use_cols]
            .sum().reset_index()
        )
        _ts["_sort"] = _ts["Calendar_Year"].astype(int) * 100 + _ts["Calendar_Month"].astype(int)
        _ts = _ts.sort_values(["State", "_sort"])[["State", "Period"] + _use_cols].rename(columns={
            _selected_apps_col: f"{_sc['name']} — Applications",
            _approval_col:      f"{_sc['name']} — Approvals",
            _origination_col:   f"{_sc['name']} — Originations",
        })
        _parts.append(_ts.set_index(["State", "Period"]))

    if _parts:
        _wide     = pd.concat(_parts, axis=1).reset_index()
        _num_cols = [c for c in _wide.columns if c not in ("State", "Period")]
        st.dataframe(
            _wide.style.format({c: "{:,.0f}" for c in _num_cols}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇ Download comparison CSV",
            data=_wide.to_csv(index=False).encode("utf-8"),
            file_name="scenario_comparison.csv",
            mime="text/csv",
        )
