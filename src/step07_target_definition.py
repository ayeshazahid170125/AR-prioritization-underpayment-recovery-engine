"""
STEP 07 - Target Definition + Leakage Check
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step07_target_definition.py

Purpose:
Define the target variable for the Step 08 collection probability model,
and run an EXPLICIT leakage check before any model training happens. This
is the most critical step in the pipeline: if a feature is mathematically
derived from the same numbers used to build the target, the model will
"cheat" and report misleadingly high accuracy that will not hold up on
real claims.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "feature_outputs" / "feature_engineered_dataset.csv"

OUTPUT_DIR = BASE_DIR / "target_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TARGET_PATH = OUTPUT_DIR / "modeling_dataset.csv"
TARGET_CARD_PATH = OUTPUT_DIR / "target_definition_card.md"
LEAKAGE_AUDIT_PATH = OUTPUT_DIR / "leakage_check_audit.csv"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD feature_engineered_dataset.csv")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ============================================================
# TARGET DEFINITION -- HONESTY FIRST
# ============================================================
print_section("DEFINING THE TARGET VARIABLE")
print("""
WHAT THE PROJECT BRIEF ASKS FOR:
"Collection probability scorer: logistic model using claim age, procedure
category, payer type, balance"

WHAT THE PUBLIC DATA ACTUALLY CONTAINS:
The CMS Provider Payment PUF has NO real collection outcome -- no record of
whether an underpaid claim was ever appealed, corrected, or recovered. This
is annual aggregate data, not claim-level AR transaction history.

HONEST TARGET DEFINITION USED HERE:
We define a documented PROXY target: "high_recovery_priority" = 1 when a
claim-group's underpayment is both (a) financially significant
(Total_Dollar_Gap below a materiality threshold) and (b) not a borderline/
noise-level variance (Payment_Gap_Pct beyond a minimum severity threshold).
This proxy answers a related but DIFFERENT question than true collection
probability: "is this underpayment large and clear enough to be worth an AR
team's time to investigate," not "will this claim actually be recovered."
That distinction will be stated explicitly in the model card and in any
client-facing materials -- this is a portfolio demonstration of the
modeling approach, not a claim of measured real-world collection rates.
""")

# Materiality threshold: a $50 total dollar gap is a defensible floor for
# "worth investigating" in an AR context (small enough to catch genuine
# underpayments, large enough to exclude rounding-level noise).
DOLLAR_THRESHOLD = 50
PCT_THRESHOLD = -10  # at least 10% underpaid, not just noise

df["high_recovery_priority"] = (
    (df["Total_Dollar_Gap"] < -DOLLAR_THRESHOLD) &
    (df["Payment_Gap_Pct"] < PCT_THRESHOLD)
).astype(int)

target_counts = df["high_recovery_priority"].value_counts()
target_pct = df["high_recovery_priority"].mean() * 100
print(f"Target distribution:")
print(target_counts.to_string())
print(f"\nPositive class (high_recovery_priority=1): {target_pct:.2f}%")


# ============================================================
# CLASS IMBALANCE CHECK
# ============================================================
print_section("CLASS IMBALANCE CHECK")
print(f"""
Positive class rate: {target_pct:.2f}%
Negative class rate: {100 - target_pct:.2f}%
""")
if target_pct < 10 or target_pct > 90:
    print("WARNING: significant class imbalance detected.")
    print("Step 08 should use PR-AUC (not plain accuracy) as the primary")
    print("metric, and consider class_weight='balanced' in the logistic model,")
    print("following the same approach used for Project 1's denial model.")
else:
    print("Class balance is reasonable -- standard accuracy metrics combined")
    print("with PR-AUC will both be reported in Step 08 for transparency.")


# ============================================================
# EXPLICIT LEAKAGE CHECK -- THE MOST IMPORTANT PART OF THIS STEP
# ============================================================
print_section("LEAKAGE CHECK")
print("""
The target was built directly from Total_Dollar_Gap and Payment_Gap_Pct.
Therefore these two columns -- and anything mathematically derived from
them -- MUST be excluded from the feature set used to train Step 08's
model. Including them would not be "a strong feature," it would be the
model looking at the answer key.
""")

# Columns that are part of the target's own definition (mathematically
# guaranteed leakage -- these literally ARE the target).
DIRECT_LEAKAGE_COLS = [
    "Total_Dollar_Gap", "Payment_Gap_Pct", "Payment_Gap", "Is_Underpaid",
]

# Columns derived FROM the leakage columns during feature engineering
# (Step 06) -- these encode the same information in bucketed form.
DERIVED_LEAKAGE_COLS = [
    "claim_severity_proxy",  # built directly from Payment_Gap_Pct bins
    "balance_size_bucket",   # built directly from Total_Dollar_Gap
]

# Columns used to CALCULATE the gap in the first place -- including these
# would let the model reconstruct the gap almost exactly (e.g. Allowed -
# Expected = Gap), which is also leakage even though they don't contain
# "Gap" in the name.
INDIRECT_LEAKAGE_RISK_COLS = [
    "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Alowd_Amt_log",
    "Expected_Payment_NonFacility_Avg", "Expected_Payment_Facility_Avg",
    "Expected_Payment_Used",
]

print("DIRECT leakage columns (the target's own ingredients):")
for col in DIRECT_LEAKAGE_COLS:
    present = "FOUND -- will be excluded" if col in df.columns else "not in dataset"
    print(f"  {col:<35} {present}")

print("\nDERIVED leakage columns (bucketed from the target's ingredients):")
for col in DERIVED_LEAKAGE_COLS:
    present = "FOUND -- will be excluded" if col in df.columns else "not in dataset"
    print(f"  {col:<35} {present}")

print("\nINDIRECT leakage-risk columns (used to calculate the gap):")
for col in INDIRECT_LEAKAGE_RISK_COLS:
    present = "FOUND -- will be excluded" if col in df.columns else "not in dataset"
    print(f"  {col:<35} {present}")

print("""
DECISION: all columns above are EXCLUDED from the Step 08 feature set.
The model will be trained ONLY on: procedure_category, payer_type_proxy,
provider type/state, Tot_Srvcs/Tot_Benes (and their logs), Avg_Sbmtd_Chrg,
services_per_beneficiary, Place_Of_Srvc, and entity type -- none of which
were used to construct high_recovery_priority.
""")


# ============================================================
# CORRELATION SANITY CHECK -- VERIFY NO HIDDEN LEAKAGE REMAINS
# ============================================================
print_section("CORRELATION SANITY CHECK ON REMAINING FEATURES")

all_leakage_cols = DIRECT_LEAKAGE_COLS + DERIVED_LEAKAGE_COLS + INDIRECT_LEAKAGE_RISK_COLS
candidate_features = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in all_leakage_cols and c != "high_recovery_priority"
]

leakage_audit_rows = []
for col in candidate_features:
    corr = df[col].corr(df["high_recovery_priority"])
    leakage_audit_rows.append({"Feature": col, "Correlation_with_Target": round(corr, 4)})

audit_df = pd.DataFrame(leakage_audit_rows).sort_values(
    "Correlation_with_Target", key=abs, ascending=False
)
print("Remaining numeric features, sorted by |correlation| with target:")
print(audit_df.to_string(index=False))

high_corr = audit_df[audit_df["Correlation_with_Target"].abs() > 0.5]
if len(high_corr) > 0:
    print(f"\nWARNING: {len(high_corr)} feature(s) still show |correlation| > 0.5.")
    print("Review these manually before Step 08 -- this threshold does not")
    print("automatically mean leakage, but it warrants a closer look.")
else:
    print("\nNo remaining feature exceeds |correlation| > 0.5 with the target.")
    print("This supports (but does not by itself prove) that direct leakage")
    print("has been removed from the candidate feature set.")

audit_df.to_csv(LEAKAGE_AUDIT_PATH, index=False)
print(f"\nSaved: {LEAKAGE_AUDIT_PATH}")


# ============================================================
# SAVE MODELING DATASET
# ============================================================
print_section("SAVING MODELING DATASET")

df.to_csv(TARGET_PATH, index=False)
print(f"Saved: {TARGET_PATH}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("(Leakage columns are still PRESENT in this file for audit purposes --")
print(" Step 08 must explicitly drop them when building the model's X matrix.)")


# ============================================================
# WRITE TARGET DEFINITION CARD
# ============================================================
card = f"""# Target Definition Card -- Project 2 Collection Priority Model

## Target Variable
`high_recovery_priority` (binary: 0 or 1)

## Definition
1 when BOTH conditions are true:
- Total_Dollar_Gap < -${DOLLAR_THRESHOLD} (financially material underpayment)
- Payment_Gap_Pct < {PCT_THRESHOLD}% (not a borderline/noise-level variance)

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
Positive class (high_recovery_priority=1): {target_pct:.2f}%

## Leakage Controls Applied
Excluded from Step 08 features:
- Direct: {', '.join(DIRECT_LEAKAGE_COLS)}
- Derived: {', '.join(DERIVED_LEAKAGE_COLS)}
- Indirect (used to calculate the gap): {', '.join(INDIRECT_LEAKAGE_RISK_COLS)}

## Next Step
Step 08 trains a Logistic Regression (per the original project brief) on
the remaining, leakage-checked feature set, using PR-AUC as the primary
evaluation metric given the class imbalance noted above.
"""
TARGET_CARD_PATH.write_text(card, encoding="utf-8")
print(f"Saved: {TARGET_CARD_PATH}")

print_section("STEP 07 COMPLETE")