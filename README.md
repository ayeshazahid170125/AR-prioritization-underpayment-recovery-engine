# AR Prioritization & Underpayment Recovery Engine

Project 02 for WellMind Data Solutions. This project estimates expected Medicare payment, identifies underpayment variance, trains a recovery-priority model, and creates an AR workqueue for high-value recovery review.

## Repository Order

```text
.
|-- app/                 FastAPI service and legacy Streamlit dashboard
|-- frontend/            React dashboard
|-- src/                 step-by-step payment variance and recovery pipeline
|-- docs/                technical report and model card
|-- reports/             curated summary reports for GitHub review
|-- scripts/             helper scripts
|-- run_pipeline.ps1     rebuilds the local pipeline outputs
|-- run_app.ps1          starts the API and dashboard
|-- setup_project.ps1    creates environment and installs requirements
|-- requirements.txt
`-- README.md
```

Large raw CMS/RVU files, model binaries, and generated multi-million-row CSVs are intentionally ignored so the repository stays clean.

## Problem Statement

Revenue Cycle teams need a practical way to identify claims that are likely underpaid and worth follow-up. This project builds a public-data proof of concept that answers:

1. What should the expected payment be?
2. Which claims appear materially underpaid?
3. Which underpaid claims should AR teams review first?
4. Which states, HCPCS codes, and provider types drive the largest recovery opportunity?

## Data Sources

- CMS Medicare Physician & Other Practitioners public use data
- CMS Physician Fee Schedule / RVU reference files
- Derived expected-payment and variance tables

No PHI or patient-level records are used.

## Pipeline

1. `step_01_eda.py` and `step01b_reload_headers.py` profile and normalize source files.
2. `step02_expected_payment.py` builds expected-payment tables from RVU references.
3. `step03_join_actual_expected.py` joins actual CMS payment with expected payment.
4. `step04a` through `step04d` audit nulls, outliers, cleaning, and benchmark applicability.
5. `step05_premodel_eda.py` explores underpayment patterns.
6. `step06_feature_engineering.py` creates model-ready recovery features.
7. `step07_target_definition.py` defines the high-recovery-priority proxy target.
8. `step08_collection_model.py` trains the recovery-priority model.
9. `step09_ar_priority_queue.py` scores and ranks the AR workqueue.
10. `step10_isolation_forest_anomalies.py` finds underpayment anomaly patterns.
11. `step11_underpayment_report.py` creates executive recovery summaries.
12. `step14_regression_validation.py` trains supplementary regressors to
    validate the CMS formula benchmark against actual allowed amounts.
13. `step12_fastapi.py` serves the API and `frontend/` serves the React dashboard.

## Model Results

Status note: these metrics are from the original run. The leakage controls
now exclude `Avg_Mdcr_Pymt_Amt`, so rerun `step07_target_definition.py` and
`step08_collection_model.py` before using these metrics as current.

Best model: **LightGBM**

| Model | Mean PR-AUC | Mean ROC-AUC | Mean F1 |
|---|---:|---:|---:|
| LightGBM | 0.8688 | 0.8791 | 0.7479 |
| Hist Gradient Boosting | 0.8663 | 0.8771 | 0.7622 |
| Random Forest | 0.8561 | 0.8653 | 0.7449 |
| Gradient Boosting | 0.8546 | 0.8639 | 0.7284 |
| Logistic Regression | 0.6921 | 0.7078 | 0.6060 |

Final holdout metrics:

| Metric | Value |
|---|---:|
| Test PR-AUC | 0.8754 |
| Test ROC-AUC | 0.8857 |
| Test F1 | 0.7577 |
| Training rows | 6,142,472 |
| Test rows | 1,535,619 |
| Features | 23 |

Immediate threshold tuning on the existing saved model:

| Threshold Strategy | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Default | 0.500000 | 0.8009 | 0.5333 | 0.6402 |
| Max F1 | 0.346990 | 0.6283 | 0.7765 | 0.6946 |
| Business operating | 0.400000 | 0.6764 | 0.7013 | 0.6886 |
| Recall target 70% | 0.400645 | 0.6772 | 0.7000 | 0.6884 |

Run `python scripts/tune_existing_threshold.py` to refresh these artifacts
without retraining.

## Regression Validation

The primary expected-payment benchmark remains the CMS fee schedule formula
because it is audit-defensible. Step 14 adds a supplementary ML regression
cross-check against actual CMS allowed amounts.

| Model | MAE | RMSE | R2 | Median AE |
|---|---:|---:|---:|---:|
| HistGradientBoosting Regressor | 14.4837 | 53.2043 | 0.9268 | 5.9604 |
| CMS Formula Benchmark | 28.0693 | 99.3812 | 0.7446 | 9.2193 |
| Linear Regression | 29.3444 | 109.8266 | 0.6881 | 14.7132 |

These results validate that payment patterns are learnable from public
aggregate features, but the ML regressor is a supplementary validation
layer, not the primary pricing engine.

## Recovery Opportunity Summary

The dollar amount below is the gross public-data estimate from the original
run. Step 09 now flags severe surgical-code rows that require modifier
review because the CMS PUF does not expose global surgery, bilateral,
assistant-surgeon, or MPPR modifiers. Client-facing totals should use the
rerun gross amount plus the modifier-review-excluded amount.

| Metric | Value |
|---|---:|
| Underpaid queue rows | 6,056,133 |
| Estimated recovery | $14,993,979,972.64 |
| Modifier-review rows, existing-artifact audit | 9,048 |
| Modifier-review estimated recovery, existing-artifact audit | $2,270,364,614.98 |
| Estimated recovery excluding modifier-review rows, existing-artifact audit | $12,723,615,357.66 |
| Critical tier rows | 9,114 |
| High tier rows | 173,218 |
| Top state by recovery | CA |
| Top HCPCS by recovery | 66984 |
| Top provider type | Diagnostic Radiology |

## Run Locally

Prerequisites:

- Python 3.11 or 3.12
- Node.js LTS, which includes `npm`

```powershell
.\setup_project.ps1
.\run_pipeline.ps1
.\run_app.ps1
```

The app starts:

- FastAPI: `http://localhost:8001`
- Swagger docs: `http://localhost:8001/docs`
- React dashboard: `http://localhost:5173`

The previous Streamlit dashboard is still available at `app/step13_dashboard.py`,
but the default local app now uses the React frontend. To run the legacy
Streamlit dashboard, install `requirements-streamlit-legacy.txt` separately.

For Vercel deployment, deploy `frontend/` as the frontend project and set
`VITE_API_BASE_URL` to the hosted FastAPI URL. Host the FastAPI service
separately on Render, Railway, AWS, Azure, GCP, or another Python web host.

## Important Notes

- This is a portfolio/demo system based on public aggregate data.
- The target is a documented proxy for recovery priority, not true collection probability.
- Surgical-code underpayment totals can be inflated by modifier/pricing
  artifacts that are not visible in the public PUF.
- Production use would require payer-specific contracts, actual claim adjudication history, and validated recovery outcomes.
