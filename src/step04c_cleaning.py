"""
STEP 04C - Final Cleaning Decisions
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step04c_cleaning.py

Purpose:
Apply final, documented cleaning decisions to claims_with_variance.csv using
the null findings (Step 04a) and outlier findings (Step 04b). Following the
same conservative principle as Project 1: nothing is blindly deleted. Every
row that is excluded from the "clean" analysis set has a documented reason
and is preserved in a separate audit file.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "variance_outputs" / "claims_with_variance.csv"

OUTPUT_DIR = BASE_DIR / "cleaning_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CLEANED_PATH = OUTPUT_DIR / "claims_cleaned.csv"
EXCLUDED_PATH = OUTPUT_DIR / "excluded_rows_with_reason.csv"
CLEANING_SUMMARY_PATH = OUTPUT_DIR / "cleaning_summary.csv"
BEFORE_AFTER_PATH = OUTPUT_DIR / "cleaning_before_after_comparison.csv"

# CPT/HCPCS ranges that use pricing systems OTHER than the physician fee
# schedule RVU formula. Confirmed during Step 04b investigation: these are
# the real reason behind ~1,041 codes showing $0 expected payment, not a
# calculation bug.
NON_PFS_PRICING_PATTERNS = {
    "Pathology/Laboratory (CPT 80000-89999)": r"^8\d{4}$",
    "Anesthesia (CPT 00100-00999, time-based pricing)": r"^00\d{3}$",
    "Drugs - ASP pricing (J-codes)": r"^J\d{4}$",
    "DME/Supplies (HCPCS Level II)": r"^[AQ]\d{4}$",
    "Other HCPCS Level II (G/K/etc.)": r"^[GK]\d{4}$",
}


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD claims_with_variance.csv")

df = pd.read_csv(INPUT_PATH, low_memory=False)
rows_before = len(df)
print(f"Loaded: {rows_before:,} rows x {df.shape[1]} columns")


# ============================================================
# DECISION 1 - ZERO EXPECTED PAYMENT ROWS
# ============================================================
print_section("DECISION 1 - ROWS WITH $0 EXPECTED PAYMENT")

zero_expected_mask = df["Expected_Payment_NonFacility_Avg"] == 0
zero_expected_count = zero_expected_mask.sum()

print(f"""
Found {zero_expected_count:,} rows ({zero_expected_count/len(df)*100:.2f}%) where the
physician fee schedule formula calculated $0 expected payment.

Investigation in Step 04b traced this to specific CPT/HCPCS code ranges
that are priced OUTSIDE the physician fee schedule RVU formula:
""")

df["zero_expected_reason"] = "N/A - has valid expected payment"
zero_expected_df = df[zero_expected_mask].copy()

reason_counts = {}
unclassified_mask = pd.Series(True, index=zero_expected_df.index)

for reason, pattern in NON_PFS_PRICING_PATTERNS.items():
    match_mask = zero_expected_df["HCPCS_Cd"].astype(str).str.match(pattern)
    match_mask = match_mask & unclassified_mask
    count = match_mask.sum()
    if count > 0:
        reason_counts[reason] = count
        df.loc[zero_expected_df[match_mask].index, "zero_expected_reason"] = reason
        unclassified_mask = unclassified_mask & ~match_mask

remaining_count = unclassified_mask.sum()
if remaining_count > 0:
    reason_counts["Other/Unclassified HCPCS code"] = remaining_count
    df.loc[zero_expected_df[unclassified_mask].index, "zero_expected_reason"] = "Other/Unclassified HCPCS code"

for reason, count in reason_counts.items():
    pct = count / zero_expected_count * 100
    print(f"  {reason:<55} {count:>10,} ({pct:>5.1f}%)")

print("""
Decision: these rows are EXCLUDED from the variance/underpayment analysis
(Step 05 onward) because there is no valid fee-schedule benchmark to compare
them against -- not because anything is wrong with the claim itself. They
are preserved in excluded_rows_with_reason.csv for transparency.
""")


# ============================================================
# DECISION 2 - EXTREME OUTLIERS (FROM STEP 04B FLAGS)
# ============================================================
print_section("DECISION 2 - EXTREME OUTLIERS")

print("""
Step 04b flagged rows using two methods: the domain rule (gap % beyond
-100%/+200%) and the Modified Z-Score. For valid-benchmark rows only (zero-
expected rows are already handled above), we now decide what to do with
the remaining flagged rows.

Decision: KEEP all flagged rows in the cleaned dataset, but add a
'review_flag' column. These are not deleted because:
  1. Some genuinely are real, large underpayments (the entire point of this
     project is to find unusual variance).
  2. Statistically extreme is not the same as factually wrong -- removing
     them would bias the dataset toward "boring" claims only.
  3. Step 05 (the variance ranking engine) and Step 08 (the collection
     model) can use the review_flag as a feature/filter rather than losing
     the information entirely.
""")

# Recompute the domain rule and modified z-score flags here, scoped to rows
# that have a valid (non-zero) expected payment.
valid_mask = ~zero_expected_mask
gap_pct_valid = df.loc[valid_mask, "Payment_Gap_Pct"]

domain_flag = (gap_pct_valid < -100) | (gap_pct_valid > 200)

median = gap_pct_valid.median()
mad = (gap_pct_valid - median).abs().median()
modified_z = 0.6745 * (gap_pct_valid - median) / mad if mad != 0 else pd.Series(0, index=gap_pct_valid.index)
modz_flag = modified_z.abs() > 3.5

df["review_flag"] = False
df.loc[gap_pct_valid.index, "review_flag"] = (domain_flag | modz_flag).values

review_count = df["review_flag"].sum()
print(f"Rows flagged for review (kept, not deleted): {review_count:,} ({review_count/len(df)*100:.2f}%)")


# ============================================================
# DECISION 3 - NEGATIVE/INVALID ALLOWED AMOUNTS
# ============================================================
print_section("DECISION 3 - INVALID ALLOWED AMOUNT VALUES")

invalid_allowed_mask = df["Avg_Mdcr_Alowd_Amt"] < 0
invalid_count = invalid_allowed_mask.sum()
print(f"Rows with negative Avg_Mdcr_Alowd_Amt: {invalid_count:,}")

if invalid_count > 0:
    print("Decision: these rows are excluded -- a negative allowed amount is")
    print("a data quality issue, not a real claim variance.")
else:
    print("Decision: none found, no action needed.")


# ============================================================
# DECISION 4 - DUPLICATE ROW CHECK
# ============================================================
print_section("DECISION 4 - DUPLICATE ROW CHECK")

dedup_cols = ["Rndrng_NPI", "HCPCS_Cd", "Place_Of_Srvc"]
duplicate_mask = df.duplicated(subset=dedup_cols, keep=False)
duplicate_count = duplicate_mask.sum()
print(f"Rows sharing the same Provider + HCPCS + Place of Service: {duplicate_count:,}")
print("""
Note: a small number of exact duplicates on these keys can be legitimate in
the PUF (e.g. different modifiers collapse to the same row in this extract),
so duplicates are flagged for visibility but not automatically dropped.
""")
df["duplicate_key_flag"] = duplicate_mask


# ============================================================
# BUILD FINAL CLEANED + EXCLUDED DATASETS
# ============================================================
print_section("BUILDING FINAL CLEANED DATASET")

exclude_mask = zero_expected_mask | invalid_allowed_mask
cleaned_df = df[~exclude_mask].copy()
excluded_df = df[exclude_mask].copy()

print(f"Rows before cleaning  : {rows_before:,}")
print(f"Rows excluded (documented): {len(excluded_df):,}")
print(f"Rows in final cleaned dataset: {len(cleaned_df):,}")
print(f"  - of which flagged for review (kept): {cleaned_df['review_flag'].sum():,}")


# ============================================================
# BEFORE / AFTER COMPARISON
# ============================================================
print_section("BEFORE VS AFTER COMPARISON")

before_after = pd.DataFrame([
    {"Metric": "Total rows", "Before": rows_before, "After": len(cleaned_df)},
    {"Metric": "Mean Payment_Gap_Pct", "Before": round(df["Payment_Gap_Pct"].replace([np.inf, -np.inf], np.nan).mean(), 2),
     "After": round(cleaned_df["Payment_Gap_Pct"].mean(), 2)},
    {"Metric": "Median Payment_Gap_Pct", "Before": round(df["Payment_Gap_Pct"].replace([np.inf, -np.inf], np.nan).median(), 2),
     "After": round(cleaned_df["Payment_Gap_Pct"].median(), 2)},
    {"Metric": "Underpaid rate (%)", "Before": round(df["Is_Underpaid"].mean() * 100, 2),
     "After": round(cleaned_df["Is_Underpaid"].mean() * 100, 2)},
    {"Metric": "Rows with infinite/invalid gap", "Before": int(zero_expected_count), "After": 0},
])
print(before_after.to_string(index=False))
before_after.to_csv(BEFORE_AFTER_PATH, index=False)


# ============================================================
# SAVE OUTPUTS
# ============================================================
print_section("SAVING OUTPUTS")

cleaned_df.to_csv(CLEANED_PATH, index=False)
print(f"Saved cleaned dataset: {CLEANED_PATH}")

excluded_df.to_csv(EXCLUDED_PATH, index=False)
print(f"Saved excluded rows (with reasons): {EXCLUDED_PATH}")

summary_df = pd.DataFrame([
    {"Metric": "Rows before cleaning", "Value": rows_before},
    {"Metric": "Rows excluded - zero expected payment", "Value": int(zero_expected_count)},
    {"Metric": "Rows excluded - invalid allowed amount", "Value": int(invalid_count)},
    {"Metric": "Rows in final cleaned dataset", "Value": len(cleaned_df)},
    {"Metric": "Rows flagged for review (kept)", "Value": int(cleaned_df["review_flag"].sum())},
    {"Metric": "Rows flagged as duplicate key", "Value": int(cleaned_df["duplicate_key_flag"].sum())},
])
summary_df.to_csv(CLEANING_SUMMARY_PATH, index=False)
print(f"Saved cleaning summary: {CLEANING_SUMMARY_PATH}")
print(f"Saved before/after comparison: {BEFORE_AFTER_PATH}")

print_section("STEP 04C COMPLETE")
print("""
claims_cleaned.csv is now the trusted base file for:
  - Step 05: pre-model EDA (20 charts)
  - Step 06: feature engineering
  - Step 07: target definition + leakage check
""")