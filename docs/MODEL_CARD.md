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

## Limitations

- The target is a recovery-priority proxy, not confirmed collection success.
- Public CMS data does not include payer contract terms or appeal outcomes.
- Production deployment would require validation against real AR and remittance records.
- Do not use this model to make payment or clinical decisions.
