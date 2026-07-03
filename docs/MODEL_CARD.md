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
| CV PR-AUC | 0.8785 |
| Test PR-AUC | 0.8859 |
| Test ROC-AUC | 0.8889 |
| Test F1, default threshold | 0.7769 |
| Test F1, max-F1 threshold | 0.7786 |
| Training rows | 1,600,000 |
| Test rows | 400,000 |
| Features | 39 |

These metrics are from the leakage-safe rerun. The classifier excludes
`Avg_Mdcr_Pymt_Amt`, a proxy for `Avg_Mdcr_Alowd_Amt`, from the feature
set.

## Operating Threshold

The default probability threshold of 0.5000 gave recall of 0.7601 on the
held-out split. Threshold tuning found 0.470006 as the max-F1 threshold,
but the selected AR operating threshold is 0.40 because it prioritizes
recall for high-value underpayment review.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.500000 | 0.7942 | 0.7601 | 0.7767 |
| 0.470006 | 0.7724 | 0.7845 | 0.7784 |
| 0.400000 | 0.7209 | 0.8359 | 0.7742 |

At the 0.40 operating threshold, 3,564,981 underpaid queue rows are scored
high priority, about 58.9% of the queue. This favors recall over precision:
roughly 28% of reviewed high-priority rows are expected false positives
based on held-out precision.

## Limitations

- The target is a recovery-priority proxy, not confirmed collection success.
- Public CMS data does not include payer contract terms or appeal outcomes.
- Public CMS PUF rows do not expose claim-line modifiers. Large apparent
  surgical-code gaps may reflect correct modifier-based payment reductions
  such as global surgery co-management modifiers 54/55/56, bilateral
  adjustments, assistant-surgeon reductions, or MPPR.
- Production deployment would require validation against real AR and remittance records.
- Do not use this model to make payment or clinical decisions.
