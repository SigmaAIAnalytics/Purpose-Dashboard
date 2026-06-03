from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Oracle — Purpose Dashboard",
    page_icon="🧿",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, .stApp, .stMarkdown, .stText,
    p, li, td, th, input, textarea, select {
        font-family: 'DM Sans', sans-serif !important;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
    }

    .stApp { background-color: var(--background-color); }
    .block-container { padding-top: 2rem; }

    p, li, td, th,
    .stMarkdown p, .stMarkdown li,
    [data-testid="stWidgetLabel"] p {
        color: var(--text-color) !important;
    }

    .step-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.75rem;
    }
    .step-number {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        color: #38bdf8;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .step-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.05rem;
        color: var(--text-color);
        margin-bottom: 0.25rem;
    }
    .step-body {
        font-size: 0.875rem;
        color: var(--text-color);
        opacity: 0.75;
        line-height: 1.5;
    }

    .info-card {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: var(--text-color);
    }

    .col-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.78rem;
        font-family: monospace;
        color: var(--text-color);
        margin: 0.15rem 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.4rem;margin-bottom:0;"
    "color:var(--text-color)'>🧿 Oracle</h1>"
    "<p style='color:var(--text-color);opacity:0.55;font-size:1.05rem;margin-top:0.2rem'>"
    "Marketing Mix Modelling — spend forecasting for loan applications</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── What is Oracle ────────────────────────────────────────────────────────────
st.markdown("## What is Oracle?")
st.markdown(
    "Oracle is a forecasting tool built on a Marketing Mix Model (MMM). "
    "Enter planned marketing spend by state and month and Oracle predicts the number of "
    "loan **Applications**, likely **Approvals**, and likely **Originations** that will result. "
    "You can run up to four spend scenarios side-by-side and compare outcomes."
)

st.divider()

# ── How to use ────────────────────────────────────────────────────────────────
st.markdown("## How to Use")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        <div class='step-card'>
            <div class='step-number'>1</div>
            <div class='step-title'>Load the model file</div>
            <div class='step-body'>
                On the <strong>Baseline</strong> page, open the <em>Model</em> expander in the
                sidebar. Upload <code>modelcoeff_and_prodfactors.csv</code> from the model
                pipeline — or it will load automatically if DigitalOcean Spaces is configured.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='step-card'>
            <div class='step-number'>2</div>
            <div class='step-title'>Enter baseline spend</div>
            <div class='step-body'>
                On the <strong>Baseline</strong> page, type monthly spend directly into the
                table or upload a CSV / Excel file. Each row is one state × month combination.
                Download the template if you need the correct column format.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class='step-card'>
            <div class='step-number'>3</div>
            <div class='step-title'>Run predictions</div>
            <div class='step-body'>
                Click <strong>▶ Run Predictions</strong>. Oracle converts your monthly spend
                to weekly, scores every coefficient row for each state, and rolls the results
                back up to monthly. Predicted Applications, Approvals, and Originations are
                shown in a filterable table below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='step-card'>
            <div class='step-number'>4</div>
            <div class='step-title'>Compare scenarios</div>
            <div class='step-body'>
                Upload up to three additional spend files using the <em>Scenario 1–3</em>
                expanders in the sidebar, then run each one. Open the
                <strong>Scenario Comparison</strong> page to see side-by-side bar charts
                and a full comparison table across all active scenarios.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Spend file format ─────────────────────────────────────────────────────────
st.markdown("## Spend File Format")
st.markdown(
    "Your CSV or Excel spend file must contain the following columns. "
    "Column names are case-insensitive and common aliases are accepted automatically."
)

st.markdown(
    """
    <div class='info-card'>
        <strong>Required columns</strong><br><br>
        <span class='col-pill'>Date</span>
        <span class='col-pill'>State</span>
        <span class='col-pill'>DSP ($)</span>
        <span class='col-pill'>LeadGen ($)</span>
        <span class='col-pill'>Paid Search ($)</span>
        <span class='col-pill'>Paid Social ($)</span>
        <span class='col-pill'>Prescreen ($)</span>
        <span class='col-pill'>Referrals ($)</span>
        <span class='col-pill'>Sweepstakes ($)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

_fmt_col, _note_col = st.columns([1, 1])
with _fmt_col:
    st.markdown("**Column notes**")
    st.markdown(
        "- **Date** — first day of the month, e.g. `2026-07-01`\n"
        "- **State** — two-letter code, e.g. `TX`, `CA`\n"
        "- **Spend columns** — total $ spend for that tactic in that month for that state\n"
        "- Leave any tactic at `0` if no spend is planned"
    )
with _note_col:
    st.markdown("**Supported states**")
    st.markdown(
        "AL, CA, CO, DE, FL, IA, ID, IN, KS, KY, LA, MI, MO, MS, "
        "NV, OH, OK, RI, SC, TN, TX, UT, WI, WY"
    )

st.divider()

# ── Pages overview ────────────────────────────────────────────────────────────
st.markdown("## Pages")

_p1, _p2, _p3 = st.columns(3)
with _p1:
    st.markdown("**🏠 Home** *(this page)*")
    st.markdown("Overview and instructions.")
with _p2:
    st.markdown("**📊 Baseline**")
    st.markdown(
        "Enter spend, load the model, run predictions. "
        "Also manages all scenario files and runs."
    )
with _p3:
    st.markdown("**🔀 Scenario Comparison**")
    st.markdown(
        "Side-by-side bar charts and a full table comparing "
        "Applications, Approvals, and Originations across all run scenarios."
    )

st.markdown(
    "<br><small style='color:var(--text-color);opacity:0.4'>"
    "Oracle v1.0 · SigmaAIAnalytics.com</small>",
    unsafe_allow_html=True,
)
