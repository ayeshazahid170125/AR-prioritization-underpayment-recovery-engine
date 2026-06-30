# Model Card: AR Prioritization & Underpayment Recovery Engine

## Overview

This model ranks likely underpaid claims for AR review using expected-payment variance, provider/service features, and recovery-priority signals.

## Model Type

LightGBM classifier selected from a multi-model comparison.

## Intended Use

Portfolio demonstration for Revenue Cycle Management analytics:

- prioritize underpaid claims
- surface high-value recovery opportunities
- support AR workqueue triage
- summarize underpayment patterns by state, HCPCS, and provider type

## Training Data

- Public CMS Medicare provider-service data
- CMS RVU / fee schedule reference files
- Derived expected-payment and variance features

## Target

`high_recovery_priority`

Positive class when:

- `Total_Dollar_Gap < -$50`
- `Payment_Gap_Pct < -10%`

## Performance

| Metric | Value |
|---|---:|
| Best model | LightGBM |
| Test PR-AUC | 0.8754 |
| Test ROC-AUC | 0.8857 |
| Test F1 | 0.7577 |
| Training rows | 6,142,472 |
| Test rows | 1,535,619 |
| Features | 23 |

These metrics are from the pre-remediation run. The leakage audit was
tightened after review to exclude `Avg_Mdcr_Pymt_Amt`, a proxy for
`Avg_Mdcr_Alowd_Amt`. Retrain Step 07 and Step 08 before treating these
scores as current.

## Operating Threshold

The default probability threshold of 0.5000 gave recall of 0.5333 on the
existing held-out split. A no-retraining threshold tuning pass found
0.346990 as the max-F1 threshold, but the selected AR operating threshold
is 0.40 because it keeps recall near 70% with a smaller review queue.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.500000 | 0.8009 | 0.5333 | 0.6402 |
| 0.346990 | 0.6283 | 0.7765 | 0.6946 |
| 0.400000 | 0.6764 | 0.7013 | 0.6886 |

These threshold results are still based on the existing saved model; rerun
Step 08 after leakage-safe feature updates for final model-card metrics.

## Limitations

- The target is a recovery-priority proxy, not confirmed collection success.
- Public CMS data does not include payer contract terms or appeal outcomes.
- Public CMS PUF rows do not expose claim-line modifiers. Large apparent
  surgical-code gaps may reflect correct modifier-based payment reductions
  such as global surgery co-management modifiers 54/55/56, bilateral
  adjustments, assistant-surgeon reductions, or MPPR.
- Production deployment would require validation against real AR and remittance records.
- Do not use this model to make payment or clinical decisions.
