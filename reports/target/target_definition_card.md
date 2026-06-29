# Target Definition Card -- Project 2 Collection Priority Model

## Target Variable
`high_recovery_priority` (binary: 0 or 1)

## Definition
1 when BOTH conditions are true:
- Total_Dollar_Gap < -$50 (financially material underpayment)
- Payment_Gap_Pct < -10% (not a borderline/noise-level variance)

## What This Target IS
A documented proxy for "is this underpayment large and clear enough to be
worth an AR team's time to investigate."

## What This Target IS NOT
A measure of true collection probability (whether the claim would actually
be successfully appealed/recovered). The CMS Provider Payment PUF contains
no real AR transaction history, appeal outcomes, or collection records.
This limitation is consistent with Project 1's approach to its denial-risk
proxy target -- the modeling methodology is demonstrated honestly on the
best available public data, not presented as ground-truth collection data.

## Class Distribution
Positive class (high_recovery_priority=1): 43.61%

## Leakage Controls Applied
Excluded from Step 08 features:
- Direct: Total_Dollar_Gap, Payment_Gap_Pct, Payment_Gap, Is_Underpaid
- Derived: claim_severity_proxy, balance_size_bucket
- Indirect (used to calculate the gap): Avg_Mdcr_Alowd_Amt, Avg_Mdcr_Alowd_Amt_log, Expected_Payment_NonFacility_Avg, Expected_Payment_Facility_Avg, Expected_Payment_Used

## Next Step
Step 08 trains a Logistic Regression (per the original project brief) on
the remaining, leakage-checked feature set, using PR-AUC as the primary
evaluation metric given the class imbalance noted above.
