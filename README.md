# Purpose Dashboard

A Streamlit application for predicting loan applications from planned marketing spend, built for Sigma AI Analytics.

## What it does

- Accepts monthly spend inputs by state (manually or via file upload)
- Converts monthly spend to weekly using pro-rata day allocation before scoring
- Scores predictions against a pre-trained marketing mix model at multiple grain levels (state, channel, H_Tactic, Detail_Tactic)
- Produces a baseline forecast (zero spend) and incremental APPS (lift from spend)
- Optionally allocates predicted applications to Approvals and Originations by product using historical conversion rates

## Getting started

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running locally

```bash
streamlit run app.py
```

### Input files

Three files can be uploaded via the sidebar:

| File | Required | Description |
|------|----------|-------------|
| Model Coefficients | Yes | CSV or Excel file containing the `MODEL_Coefficients` sheet from the model output |
| Product Factors | No | `product_factors.csv` produced alongside the coefficients file — enables Approvals and Originations columns |
| Spend Data | No | CSV or Excel with columns: `Date`, `State`, and one column per tactic (see template download) |

### Spend data format

The spend input table expects one row per date/state combination with monthly spend figures. Columns:

```
Date | State | DSP ($) | LeadGen ($) | Paid Search ($) | Paid Social ($) | Prescreen ($) | Referrals ($) | Sweepstakes ($)
```

A template CSV can be downloaded from the app. Uploaded files are matched flexibly — column names are case-insensitive and the `($)` suffix is optional.

## Output

The predictions table includes:

- **Predicted APPS** — model output for the entered spend
- **Baseline APPS** — model output with all spend set to zero (intercept + time trend only)
- **Incremental APPS** — lift attributable to the entered spend (always ≥ 0)
- **Allocated Approvals / Originations** — available when the product factors file is uploaded

Results can be filtered by State, Month, Channel, H_Tactic, Detail_Tactic, and Product Funded. Downloads include the full unfiltered dataset as CSV or Excel.

## Deployment

The app is deployed on DigitalOcean App Platform and configured to redeploy automatically on every push to the `main` branch of this repository. The deployment spec is in `.do/app.yaml`.

Live URL: https://purpose-dashboard-vsylm.ondigitalocean.app/
