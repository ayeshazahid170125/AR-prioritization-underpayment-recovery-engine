# AR Prioritization & Underpayment Recovery Engine

Project 02 for WellMind Data Solutions. This project estimates expected Medicare payment, identifies underpayment variance, trains a recovery-priority model, and creates an AR workqueue for high-value recovery review.

## Repository Order

```text
.
|-- app/                 FastAPI service and Streamlit dashboard
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
12. `step12_fastapi.py` and `step13_dashboard.py` serve the local demo.

## Model Results

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

## Recovery Opportunity Summary

| Metric | Value |
|---|---:|
| Underpaid queue rows | 6,056,133 |
| Estimated recovery | $14,993,979,972.64 |
| Critical tier rows | 8,609 |
| High tier rows | 162,776 |
| Top state by recovery | CA |
| Top HCPCS by recovery | 66984 |
| Top provider type | Diagnostic Radiology |

## Run Locally

```powershell
.\setup_project.ps1
.\run_pipeline.ps1
.\run_app.ps1
```

The app starts:

- FastAPI: `http://localhost:8001`
- Swagger docs: `http://localhost:8001/docs`
- Streamlit dashboard: launched by Streamlit

## Important Notes

- This is a portfolio/demo system based on public aggregate data.
- The target is a documented proxy for recovery priority, not true collection probability.
- Production use would require payer-specific contracts, actual claim adjudication history, and validated recovery outcomes.
