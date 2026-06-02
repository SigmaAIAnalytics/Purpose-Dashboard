# Adding Channel / H_Tactic / Detail_Tactic to the Spend Input File

## Simple Changes (ingestion)

- **`_UPLOAD_ALIASES`** — add aliases for the new column names
- **`_normalise_upload`** — accept them as optional columns (not required, so existing files don't break)
- **`_monthly_to_weekly`** — pass the grain columns through to the weekly output so they aren't dropped during the melt/spread

## The Hard Change — `run_predictions` Scoring Logic

This is where it gets significant. Currently the model works like this:
- Input spend is at **State × Month** level (total per tactic)
- Every coefficient row for that state (at every grain — channel, H_tactic, detail_tactic) gets scored against the **same total tactic spend**
- The coefficients at each grain level were fitted against total spend, so this is by design

If Channel / H_Tactic / Detail_Tactic are added to the input, you need to decide what those columns *mean*:

| Interpretation | Implication |
|---|---|
| **Breakdown** — spend is split across channels (e.g. DIGITAL gets $X of total DSP) | Scoring logic changes: each coefficient row only gets its matching slice of spend, not the total |
| **Override** — each grain row has its own independent spend figure | Simpler but requires the input file to have one row per grain combination |

The breakdown interpretation is a meaningful change to the scoring model and would need to be validated against the original model fitting assumptions.

## Upside for the Spend Metric

If grain columns are included in the input file, calculating a filtered spend metric becomes trivial — filter `input_snap` by the same State / Month / Channel / H_Tactic / Detail_Tactic selections and sum directly. No `TACTIC_MAP` mapping required.

## Recommendation

Before making any file format changes, confirm with whoever built the model coefficients whether the coefficients assume **total spend** or **grain-level spend** as their input. That answer determines whether this is a small change or a significant one.
