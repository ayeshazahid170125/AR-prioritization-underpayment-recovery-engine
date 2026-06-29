# Project 02 Technical Summary Report

## Project

**AR Prioritization & Underpayment Recovery Engine**

This project estimates expected Medicare payment, compares it against actual allowed/payment amounts, identifies likely underpayment variance, and ranks claims for AR recovery review.

## Business Goal

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

| Model | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 |
|---|---:|---:|---:|---:|
| LightGBM | 0.8688 | 0.0014 | 0.8791 | 0.7479 |
| Hist Gradient Boosting | 0.8663 | 0.0015 | 0.8771 | 0.7622 |
| Random Forest | 0.8561 | 0.0016 | 0.8653 | 0.7449 |
| Gradient Boosting | 0.8546 | 0.0014 | 0.8639 | 0.7284 |
| Logistic Regression | 0.6921 | 0.0040 | 0.7078 | 0.6060 |

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
| Test PR-AUC | 0.8754 |
| Test ROC-AUC | 0.8857 |
| Test F1 | 0.7577 |
| Training rows | 6,142,472 |
| Test rows | 1,535,619 |
| Features | 23 |

## AR Priority Queue

| Metric | Value |
|---|---:|
| Total modeling rows | 7,678,091 |
| Underpaid queue rows | 6,056,133 |
| Model scored rows | 6,056,133 |
| Rule fallback rows | 0 |
| Total estimated recovery | $14,993,979,972.64 |
| Critical tier rows | 8,609 |
| High tier rows | 162,776 |

## Underpayment Report

| Metric | Value |
|---|---:|
| Total underpaid rows | 6,056,133 |
| Total estimated recovery | $14,993,979,972.64 |
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
| Anomaly patterns | 5,491 |
| Anomaly rate | 9.5816% |

## Demo App

- FastAPI inference service: `app/step12_fastapi.py`
- Streamlit dashboard: `app/step13_dashboard.py`
- Local launcher: `run_app.ps1`

## Limitations

- Public CMS data does not include actual payer contract adjudication or appeal recovery outcomes.
- Expected-payment logic is based on public fee schedule references and proxy assumptions.
- The model predicts recovery priority, not guaranteed recoverability.
- Production use would require payer contract terms, remittance records, denial/appeal outcomes, and compliance review.
