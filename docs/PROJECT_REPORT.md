# Project 02 Technical Summary Report

## Project

**AR Prioritization & Underpayment Recovery Engine**

This project estimates expected Medicare payment, compares it against actual allowed/payment amounts, identifies likely underpayment variance, and ranks claims for AR recovery review.

## Goal

Help Revenue Cycle teams prioritize underpaid claims by estimated recovery value, severity, and model confidence. The system is designed as a portfolio proof of concept using public CMS data, not as a production payer-contract adjudication engine.

## Data Sources

- CMS Medicare Physician & Other Practitioners public use data
- CMS Physician Fee Schedule / RVU reference files
- Derived expected-payment and variance outputs

## Target Definition

Target variable: `high_recovery_priority`

Positive class is defined when both conditions are true:

- `Total_Dollar_Gap < -$50`
- `Payment_Gap_Pct < -10%`

This target represents a material underpayment proxy that is worth AR review. It does not represent confirmed collection success.

## Model Comparison

Current results are from the leakage-safe rerun. `Avg_Mdcr_Pymt_Amt` is
excluded from the classifier feature set because it is a strong proxy for
allowed amount.

| Model | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 |
|---|---:|---:|---:|---:|
| LightGBM | 0.8785 | 0.0009 | 0.8819 | 0.7690 |
| Hist Gradient Boosting | 0.8763 | 0.0010 | 0.8795 | 0.7658 |
| Random Forest | 0.8712 | 0.0017 | 0.8750 | 0.7601 |
| Gradient Boosting | 0.8617 | 0.0017 | 0.8659 | 0.7315 |
| Logistic Regression | 0.7063 | 0.0031 | 0.7326 | 0.6312 |

## Selected Model

Best model: **LightGBM**

Best parameters:

| Parameter | Value |
|---|---:|
| learning_rate | 0.1 |
| max_depth | 10 |
| n_estimators | 200 |

Final test metrics:

| Metric | Value |
|---|---:|
| Test PR-AUC | 0.8859 |
| Test ROC-AUC | 0.8889 |
| Test F1, default threshold | 0.7769 |
| Test F1, max-F1 threshold | 0.7786 |
| Training rows | 1,600,000 |
| Test rows | 400,000 |
| Features | 39 |

Threshold tuning on the leakage-safe saved model:

| Threshold Strategy | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Default | 0.500000 | 0.7942 | 0.7601 | 0.7767 |
| Max F1 | 0.470006 | 0.7724 | 0.7845 | 0.7784 |
| Business operating | 0.400000 | 0.7209 | 0.8359 | 0.7742 |
| Recall target 70% | 0.564424 | 0.8418 | 0.7000 | 0.7644 |

The selected operating threshold is saved in
`model_outputs/decision_threshold_config.json` and consumed by the API and
AR queue logic. The business operating threshold is 0.40 to prioritize
recall because the cost of missing high-value underpayment candidates is
higher than the cost of extra review. This creates a larger review queue:
3,564,981 underpaid rows, or 58.9% of the queue, are scored above the
operating threshold, with about 28% expected false positives at the model
precision shown above.

## Regression Validation

Step 14 trains supplementary regressors to predict `Avg_Mdcr_Alowd_Amt`
and compares them against the CMS formula benchmark. This satisfies the
brief's regression requirement while keeping the fee schedule formula as
the primary expected-payment engine.

| Model | MAE | RMSE | R2 | Median AE |
|---|---:|---:|---:|---:|
| HistGradientBoosting Regressor | 14.5420 | 52.7397 | 0.9281 | 6.0419 |
| CMS Formula Benchmark | 28.0693 | 99.3812 | 0.7446 | 9.2193 |
| Linear Regression | 29.0328 | 108.6126 | 0.6950 | 14.8625 |

The regression result is a validation cross-check, not a replacement for
the CMS formula benchmark or payer-contract adjudication.

## AR Priority Queue

**Important:** the recovery amount below is a gross public-data estimate,
not a client-ready recoverable amount. Step 09 now flags severe surgical
HCPCS gaps that may be modifier/pricing artifacts because the PUF does not
include modifiers 54/55/56, bilateral adjustment, assistant-surgeon, or
MPPR detail. Both gross and modifier-review-excluded totals are reported
below.

| Metric | Value |
|---|---:|
| Total modeling rows | 7,678,091 |
| Underpaid queue rows | 6,056,133 |
| Model scored rows | 6,056,133 |
| Rule fallback rows | 0 |
| Predicted high priority rows | 3,564,981 |
| Total estimated recovery | $14,993,979,972.64 |
| Modifier-review rows, existing-artifact audit | 9,048 |
| Modifier-review estimated recovery, existing-artifact audit | $2,270,364,614.98 |
| Estimated recovery excluding modifier-review rows, existing-artifact audit | $12,723,615,357.66 |
| Critical tier rows | 9,114 |
| High tier rows | 173,218 |

## Underpayment Report

| Metric | Value |
|---|---:|
| Total underpaid rows | 6,056,133 |
| Total estimated recovery | $14,993,979,972.64 |
| Modifier-review rows | 9,048 |
| Modifier-review estimated recovery | $2,270,364,614.98 |
| Estimated recovery excluding modifier-review rows | $12,723,615,357.66 |
| Top state by recovery | CA |
| Top state recovery amount | $1,370,413,568.20 |
| Top HCPCS by recovery | 66984 |
| Top HCPCS recovery amount | $2,208,985,130.83 |
| Top provider type by recovery | Diagnostic Radiology |
| Unique HCPCS codes | 3,764 |
| Unique states | 53 |
| Unique provider types | 107 |

## Anomaly Detection

Isolation Forest was used to find unusual underpayment patterns across state, provider type, HCPCS, and place of service groups.

| Metric | Value |
|---|---:|
| Input rows | 7,678,091 |
| Groups formed | 391,005 |
| Groups scored | 57,308 |
| Anomaly patterns | 5,731 |
| Anomaly rate | 10.0003% |

The highest-scored anomaly patterns are concentrated in HCPCS 66984
cataract-related rows, consistent with the surgical/modifier artifact audit
called out in the AR queue and underpayment report.

## Demo App

- FastAPI inference service: `app/step12_fastapi.py`
- Streamlit dashboard: `app/step13_dashboard.py`
- Local launcher: `run_app.ps1`

## Limitations

- Public CMS data does not include actual payer contract adjudication or appeal recovery outcomes.
- Expected-payment logic is based on public fee schedule references and proxy assumptions.
- Public CMS PUF data does not include claim-line modifiers. Global surgical
  period co-management, bilateral adjustment, assistant-surgeon reduction,
  and MPPR can make legitimate payments look like underpayments in this
  benchmark.
- The model predicts recovery priority, not guaranteed recoverability.
- Production use would require payer contract terms, remittance records, denial/appeal outcomes, and compliance review.
