"""
build_historical_forecast.py

Produces a combined historical + forecast CSV by stacking:
  - Last N months of actuals from a history file at (State, Channel,
    H_Tactic, Detail_Tactic, Product_Funded) grain
  - M months of predictions scored from FutureSpend.csv + model
    coefficients, with product allocation applied

The output file (historical_forecast.csv by default) is designed to be
uploaded to DO Spaces so the dashboard's second page can chart it.

Usage:
    python build_historical_forecast.py \\
        --history        history.csv \\
        --future-spend   FutureSpend.csv \\
        --coefficients   model_coefficients_consolidated.csv \\
        --product-factors product_factors_consolidated.csv \\
        [--output        historical_forecast.csv] \\
        [--model-type    OLS] \\
        [--feature-run   weekly] \\
        [--history-months 6]

When ready to package into build_state_division_models.py, the three
main functions (extract_actuals, prepare_weekly_spend, score_forecast)
can be moved there and this file becomes a thin CLI wrapper.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from build_state_division_models import (
    YEAR_COL,
    WEEK_COL,
    STATE_COL,
    _parse_key_string,
    apply_product_allocation_to_forecast,
    format_predicted_apps,
    load_tabular_input,
    safe_name,
    score_spend_with_coefficients,
    spread_monthly_spend_to_weekly,
)

# ── Column mappings for the monthly-wide dashboard spend format ───────────────
_SPEND_COL_MAP: dict[str, str] = {
    "DSP ($)":          "DSP",
    "LeadGen ($)":      "LeadGen",
    "Paid Search ($)":  "Paid Search",
    "Paid Social ($)":  "Paid Social",
    "Prescreen ($)":    "Prescreen",
    "Referrals ($)":    "Referrals",
    "Sweepstakes ($)":  "Sweepstakes",
}
_SPEND_COLS = list(_SPEND_COL_MAP.values())
_MONTHLY_TACTICS = ["Prescreen"]

# ── Output column order ───────────────────────────────────────────────────────
_OUTPUT_COLS = [
    "Type", "State", "ISO_Year", "ISO_Week",
    "Channel", "H_Tactic", "Detail_Tactic", "Product_Funded",
    "Applications", "Approvals", "Originations",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso_week_start(iso_year: int, iso_week: int) -> date:
    return date.fromisocalendar(int(iso_year), int(iso_week), 1)


def _months_ago(n: int) -> tuple[int, int]:
    """Return (year, month) that is n months before the current month."""
    today = date.today()
    m = today.month - n
    y = today.year
    while m <= 0:
        m += 12
        y -= 1
    return y, m


def _ym_ge(row_year: int, row_month: int, cutoff_year: int, cutoff_month: int) -> bool:
    return (row_year, row_month) >= (cutoff_year, cutoff_month)


# ── Step 1: Extract actuals ───────────────────────────────────────────────────

def extract_actuals(history_path: str, n_months: int) -> pd.DataFrame:
    """Load history file and return last n_months of actuals at monthly grain.

    Groups by (State, Channel, H_Tactic, Detail_Tactic, Product_Funded,
    Year, Month) and sums Applications, Approvals, Originations.
    """
    raw = load_tabular_input(history_path, "History")

    required = {YEAR_COL, WEEK_COL, STATE_COL, "APPLICATIONS"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"History file missing columns: {sorted(missing)}")

    df = raw.copy()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[WEEK_COL] = pd.to_numeric(df[WEEK_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, WEEK_COL, STATE_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[WEEK_COL] = df[WEEK_COL].astype(int)

    for metric in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        else:
            df[metric] = 0.0

    # Map ISO week → calendar month
    df["_year"]  = df.apply(lambda r: _iso_week_start(r[YEAR_COL], r[WEEK_COL]).year,  axis=1)
    df["_month"] = df.apply(lambda r: _iso_week_start(r[YEAR_COL], r[WEEK_COL]).month, axis=1)

    # Filter to last n_months
    cutoff_y, cutoff_m = _months_ago(n_months)
    df = df[df.apply(lambda r: _ym_ge(r["_year"], r["_month"], cutoff_y, cutoff_m), axis=1)]

    if df.empty:
        print(f"  Warning: no actuals found within the last {n_months} months.")
        return pd.DataFrame()

    # Optional dimension columns
    dim_map = {
        "CHANNEL_CD":    "Channel",
        "H_TACTIC":      "H_Tactic",
        "DETAIL_TACTIC": "Detail_Tactic",
        "PRODUCT_FUNDED":"Product_Funded",
    }
    for src, dst in dim_map.items():
        df[dst] = df[src].astype(str) if src in df.columns else None

    df["State"] = df[STATE_COL].astype(str)

    group_cols = ["State", YEAR_COL, WEEK_COL, "Channel", "H_Tactic", "Detail_Tactic", "Product_Funded"]
    agg = (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            Applications=("APPLICATIONS", "sum"),
            Approvals=("APPROVED",       "sum"),
            Originations=("ORIGINATIONS", "sum"),
        )
        .rename(columns={YEAR_COL: "ISO_Year", WEEK_COL: "ISO_Week"})
    )
    agg["Type"] = "Actual"
    return agg


# ── Step 2: Prepare weekly spend from FutureSpend.csv ────────────────────────

def prepare_weekly_spend(future_spend_path: str, spend_format: str = "monthly") -> pd.DataFrame:
    """Load FutureSpend.csv and return weekly wide spend DataFrame.

    Supports two formats:
      'monthly' (default) — Date, State, DSP ($), ...  (dashboard upload format)
      'weekly'            — ISO_YEAR, ISO_WEEK, STATE_CD, DSP, ...
    """
    raw = pd.read_csv(future_spend_path)

    if spend_format == "weekly":
        required = {YEAR_COL, WEEK_COL, STATE_COL}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Weekly spend file missing columns: {sorted(missing)}")
        return raw

    # Monthly wide: rename columns, melt, spread, pivot back to wide
    raw = raw.rename(columns={"State": "STATE_CD", "Date": "BUSINESS_DATE"})
    raw = raw.rename(columns={k: v for k, v in _SPEND_COL_MAP.items() if k in raw.columns})
    raw["BUSINESS_DATE"] = pd.to_datetime(raw["BUSINESS_DATE"])

    spend_cols_present = [c for c in _SPEND_COLS if c in raw.columns]
    long = raw.melt(
        id_vars=["BUSINESS_DATE", "STATE_CD"],
        value_vars=spend_cols_present,
        var_name="DETAIL_TACTIC",
        value_name="TOTAL_COST",
    )
    long["TOTAL_COST"] = pd.to_numeric(long["TOTAL_COST"], errors="coerce").fillna(0.0)

    weekly_long = spread_monthly_spend_to_weekly(long, monthly_tactics=_MONTHLY_TACTICS)
    weekly_long = weekly_long.dropna(subset=["ISO_WEEK"]).copy()
    weekly_long["ISO_WEEK"] = weekly_long["ISO_WEEK"].astype(int)
    weekly_long["ISO_YEAR"] = weekly_long["ISO_YEAR"].astype(int)

    wide = (
        weekly_long.groupby([STATE_COL, YEAR_COL, WEEK_COL, "DETAIL_TACTIC"])["TOTAL_COST"]
        .sum()
        .unstack("DETAIL_TACTIC")
        .reset_index()
    )
    for col in spend_cols_present:
        if col not in wide.columns:
            wide[col] = 0.0
    wide[spend_cols_present] = wide[spend_cols_present].fillna(0.0)
    return wide


# ── Step 3: Score and roll up to monthly ─────────────────────────────────────

def score_forecast(
    spend_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    product_factors_df: pd.DataFrame,
    model_type: str = "OLS",
    feature_run: str = "weekly",
    key_contains: Optional[str] = None,
) -> pd.DataFrame:
    """Score weekly spend, apply product allocation, return weekly rows."""
    if key_contains:
        n_before = len(coeff_df)
        coeff_df = coeff_df[coeff_df["Key"].astype(str).str.contains(key_contains, na=False)].copy()
        print(f"  KEY_CONTAINS='{key_contains}': kept {len(coeff_df)}/{n_before} coefficient rows")

    preds = score_spend_with_coefficients(spend_df, coeff_df, model_type, feature_run)
    if preds.empty:
        print(f"  Warning: no predictions produced for model_type={model_type}, "
              f"feature_run={feature_run}. Check that these match values in the "
              f"coefficients file.")
        return pd.DataFrame()

    # Warn if the same state has keys at different grain depths — summing them
    # in _add_state_rollup would inflate totals.  Set KEY_CONTAINS to fix this.
    if not key_contains:
        preds["_depth"] = preds["Key"].astype(str).str.count(r"\|")
        mixed = preds.groupby(STATE_COL)["_depth"].nunique()
        mixed_states = mixed[mixed > 1]
        if not mixed_states.empty:
            print(
                f"  ⚠️  Multi-grain coefficients detected for "
                f"{len(mixed_states)} state(s): {mixed_states.index.tolist()[:5]}...\n"
                f"     Set KEY_CONTAINS (e.g. 'CHANNEL_CD') to isolate one grain level "
                f"and avoid double-counting in the rollup."
            )
        preds = preds.drop(columns=["_depth"])

    # Parse Key → dimension columns required by roll_up_weekly_forecast_to_monthly
    parsed = preds["Key"].apply(_parse_key_string)
    preds["State"]         = parsed.apply(lambda d: d.get("STATE_CD"))
    preds["Channel"]       = parsed.apply(lambda d: d.get("CHANNEL_CD"))
    preds["H_Tactic"]      = parsed.apply(lambda d: d.get("H_TACTIC"))
    preds["Detail_Tactic"] = parsed.apply(lambda d: d.get("DETAIL_TACTIC"))
    preds["Product"]       = None
    preds["Scope"]         = None
    preds["Run_Status"]    = "SUCCESS"

    # Rename to the column names expected by downstream helpers
    preds = preds.rename(columns={
        YEAR_COL:              "ISO_Year",
        WEEK_COL:              "ISO_Week",
        "Predicted_APPS_Raw":  "Predicted APPS Raw",
        "Predicted_APPS":      "Predicted APPS",
    })

    # Remove coarser-grain rows before rollup so channel-level and tactic-level
    # predictions for the same channel/week aren't both counted.
    preds = _dedup_forecast_grain(preds)

    # Apply product allocation (adds APPLICATIONS_{label} etc. at weekly grain)
    if not product_factors_df.empty:
        preds = apply_product_allocation_to_forecast(preds, product_factors_df)

    id_cols = [c for c in ["State", "ISO_Year", "ISO_Week", "Channel", "H_Tactic", "Detail_Tactic"] if c in preds.columns]

    # Reshape to long format: one row per (weekly grain × product)
    if not product_factors_df.empty:
        product_names = sorted(product_factors_df["PRODUCT_FUNDED"].dropna().astype(str).unique())
        frames = []
        for prod_name in product_names:
            label     = safe_name(prod_name)
            apps_col  = f"APPLICATIONS_{label}"
            appr_col  = f"APPROVAL_{label}"
            orig_col  = f"ORIGINATIONS_{label}"
            if apps_col not in preds.columns:
                continue
            sub = preds[id_cols].copy()
            sub["Product_Funded"] = prod_name
            sub["Applications"]   = preds[apps_col].fillna(0).apply(format_predicted_apps)
            sub["Approvals"]      = preds.get(appr_col,  pd.Series(0, index=preds.index)).fillna(0).apply(format_predicted_apps)
            sub["Originations"]   = preds.get(orig_col,  pd.Series(0, index=preds.index)).fillna(0).apply(format_predicted_apps)
            frames.append(sub)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        result = preds[id_cols].copy()
        result["Product_Funded"] = None
        result["Applications"]   = preds["Predicted APPS"].apply(format_predicted_apps)
        result["Approvals"]      = pd.Series(0, index=preds.index)
        result["Originations"]   = pd.Series(0, index=preds.index)

    if result.empty:
        return pd.DataFrame()

    result["Type"] = "Forecast"
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def _dedup_forecast_grain(preds: pd.DataFrame) -> pd.DataFrame:
    """Remove coarser-grain rows where finer-grain rows exist for the same path.

    If (State, Channel, week) has both channel-only rows (H_Tactic=null) and
    tactic-level rows (H_Tactic specified), the channel-only rows are dropped.
    Same logic applied for Detail_Tactic within each (State, Channel, H_Tactic).

    This prevents _add_state_rollup from double-counting when the coefficient
    file contains predictions at multiple grain levels for the same channel.
    """
    if preds.empty:
        return preds

    preds = preds.copy()
    n_before = len(preds)

    # Drop null-H_Tactic rows where tactic-level rows exist for same channel/week
    has_htactic = preds.groupby(
        ["State", "Channel", "ISO_Year", "ISO_Week"], dropna=False
    )["H_Tactic"].transform(lambda s: s.notna().any())
    preds = preds[~(preds["H_Tactic"].isna() & has_htactic)].copy()

    # Drop null-Detail_Tactic rows where detail-level rows exist for same tactic/week
    has_detail = preds.groupby(
        ["State", "Channel", "H_Tactic", "ISO_Year", "ISO_Week"], dropna=False
    )["Detail_Tactic"].transform(lambda s: s.notna().any())
    preds = preds[~(preds["Detail_Tactic"].isna() & has_detail)]

    dropped = n_before - len(preds)
    if dropped:
        print(f"  Cross-grain dedup: removed {dropped} coarser-grain rows "
              f"(kept finest grain per State × Channel × week)")
    return preds


def _add_state_rollup(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For each Type (Actual / Forecast), add a state-level aggregate row where
    Channel, H_Tactic, Detail_Tactic, and Product_Funded are null.

    These are the rows the dashboard shows when the user selects 'Overall' in
    the dimension dropdowns.

    For Forecast: if null-Channel rows already exist (depth-1 state-level model
    predictions), use THEM as the authoritative rollup.  Summing depth-4 tactic
    predictions over-counts for the ~23 states that have both a state-level and
    tactic-level coefficient key.
    For Actual: rollup is always computed by summing the detail rows (no depth-1
    actuals exist).
    """
    frames = []
    for type_val, grp in combined.groupby("Type"):
        null_ch = (
            grp["Channel"].isna() |
            grp["Channel"].astype(str).str.strip().isin({"None", "nan", ""})
        )
        detail_rows = grp[~null_ch]

        if detail_rows.empty:
            # Only state-level source rows — keep as-is, no rollup needed
            frames.append(grp)
            continue

        detail_rows = detail_rows.copy()
        for col in ["Applications", "Approvals", "Originations"]:
            if col in detail_rows.columns:
                detail_rows[col] = pd.to_numeric(detail_rows[col], errors="coerce").fillna(0)

        existing_null_ch = grp[null_ch]

        if type_val == "Forecast" and not existing_null_ch.empty:
            # Use depth-1 state-level model predictions as the rollup; they are
            # already the correct state total and must not be replaced by the
            # sum of tactic-level predictions.
            existing_null_ch = existing_null_ch.copy()
            for col in ["Applications", "Approvals", "Originations"]:
                if col in existing_null_ch.columns:
                    existing_null_ch[col] = pd.to_numeric(existing_null_ch[col], errors="coerce").fillna(0)
            rollup = (
                existing_null_ch
                .groupby(["State", "ISO_Year", "ISO_Week"], as_index=False)
                [["Applications", "Approvals", "Originations"]]
                .sum()
            )
        else:
            rollup = (
                detail_rows
                .groupby(["State", "ISO_Year", "ISO_Week"], as_index=False)
                [["Applications", "Approvals", "Originations"]]
                .sum()
            )

        rollup["Type"]           = type_val
        rollup["Channel"]        = None
        rollup["H_Tactic"]       = None
        rollup["Detail_Tactic"]  = None
        rollup["Product_Funded"] = None

        frames.append(detail_rows)
        frames.append(rollup)

    return pd.concat(frames, ignore_index=True) if frames else combined


def build_historical_forecast(
    history_path: str,
    future_spend_path: str,
    coefficients_path: str,
    product_factors_path: Optional[str] = None,
    output_path: str = "historical_forecast.csv",
    model_type: str = "OLS",
    feature_run: str = "weekly",
    history_months: int = 6,
    spend_format: str = "monthly",
    key_contains: Optional[str] = None,
) -> pd.DataFrame:
    """Build and save historical_forecast.csv.

    Returns the combined DataFrame so callers can inspect or upload it
    without re-reading the file.
    """
    print(f"Loading coefficients from {coefficients_path} ...")
    coeff_df = pd.read_csv(coefficients_path)
    available = coeff_df[["Model_Type", "Feature_Run"]].drop_duplicates().to_dict("records")
    print(f"  Available model variants: {available}")

    product_factors_df = pd.DataFrame()
    if product_factors_path and Path(product_factors_path).exists():
        print(f"Loading product factors from {product_factors_path} ...")
        product_factors_df = pd.read_csv(product_factors_path)
        products = product_factors_df["PRODUCT_FUNDED"].dropna().unique().tolist()
        print(f"  Products: {products}")

    print(f"\nExtracting last {history_months} months of actuals from {history_path} ...")
    actuals = extract_actuals(history_path, history_months)
    print(f"  → {len(actuals)} actuals rows")

    print(f"\nPreparing weekly spend from {future_spend_path} (format: {spend_format}) ...")
    spend_df = prepare_weekly_spend(future_spend_path, spend_format)
    print(f"  → {len(spend_df)} weekly spend rows across {spend_df[STATE_COL].nunique()} states")

    print(f"\nScoring forecast (model_type={model_type}, feature_run={feature_run}"
          + (f", key_contains='{key_contains}'" if key_contains else "") + ") ...")
    forecast = score_forecast(spend_df, coeff_df, product_factors_df, model_type, feature_run,
                              key_contains=key_contains)
    print(f"  → {len(forecast)} forecast rows")

    if actuals.empty and forecast.empty:
        print("\nNothing to write — both actuals and forecast are empty.")
        return pd.DataFrame()

    combined = pd.concat([actuals, forecast], ignore_index=True)
    combined = _add_state_rollup(combined)
    present  = [c for c in _OUTPUT_COLS if c in combined.columns]
    combined = (
        combined[present]
        .sort_values(["State", "Type", "ISO_Year", "ISO_Week"])
        .reset_index(drop=True)
    )

    # Write "Overall" explicitly for null/blank dimension columns so the chart
    # page dropdowns can always filter to them without relying on coercion.
    for col in ["Channel", "H_Tactic", "Detail_Tactic"]:
        if col in combined.columns:
            combined[col] = (
                combined[col]
                .fillna("Overall")
                .astype(str)
                .str.strip()
                .replace({"nan": "Overall", "None": "Overall", "": "Overall"})
            )

    combined.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}  ({len(combined)} rows)")
    return combined


# ── Configuration — edit these paths before running ──────────────────────────

HISTORY_PATH          = "history.csv"
FUTURE_SPEND_PATH     = "FutureSpend.csv"
COEFFICIENTS_PATH     = "model output/model_coefficients_consolidated.csv"
PRODUCT_FACTORS_PATH  = "model output/product_factors_consolidated.csv"   # set to None to skip
OUTPUT_PATH           = "historical_forecast.csv"

MODEL_TYPE            = "OLS"      # "OLS" or "NNLS"
FEATURE_RUN           = "weekly"   # "weekly", "f_dummy", or "fourier"
HISTORY_MONTHS        = 6          # how many months of actuals to include
SPEND_FORMAT          = "monthly"  # "monthly" = dashboard format (Date, State, DSP ($) ...)
                                   # "weekly"  = ISO_YEAR, ISO_WEEK, STATE_CD, DSP ...

# Optional: filter coefficient Keys to a specific grain level.
# If the coefficient file has both state-level keys ("STATE_CD=TX") and
# channel-level keys ("STATE_CD=TX|CHANNEL_CD=Paid Search"), scoring both
# would double-count.  Set KEY_CONTAINS to keep only the grain you want, e.g.:
#   KEY_CONTAINS = "CHANNEL_CD"   → channel-level only
#   KEY_CONTAINS = "H_TACTIC"     → tactic-level only
#   KEY_CONTAINS = None           → use all keys (may inflate if multi-grain)
KEY_CONTAINS          = None

# ── Run ───────────────────────────────────────────────────────────────────────

combined_df = build_historical_forecast(
    history_path=HISTORY_PATH,
    future_spend_path=FUTURE_SPEND_PATH,
    coefficients_path=COEFFICIENTS_PATH,
    product_factors_path=PRODUCT_FACTORS_PATH,
    output_path=OUTPUT_PATH,
    model_type=MODEL_TYPE,
    feature_run=FEATURE_RUN,
    history_months=HISTORY_MONTHS,
    spend_format=SPEND_FORMAT,
    key_contains=KEY_CONTAINS,
)

combined_df
