"""
STEP 04D - Flag Facility-Pricing Provider Types
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step04d_benchmark_applicability.py

Purpose:
Step 05 EDA revealed that Ambulatory Surgical Center, Nuclear Medicine, and
Independent Diagnostic Testing Facility (IDTF) show implausibly large
POSITIVE payment gaps (+150% to +270%). Investigation confirmed these
provider types are paid under separate CMS pricing systems (ASC Payment
System, facility-bundled imaging rates) rather than the Physician Fee
Schedule RVU formula used in Step 02. Comparing them against the physician
fee schedule is not a valid apples-to-apples benchmark.

Following the same conservative principle as every prior cleaning step:
these rows are NOT deleted. They are flagged so that Step 06 onward can
correctly exclude them from underpayment statistics while keeping them
visible in the full dataset for transparency.
"""

from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "cleaning_outputs" / "claims_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "cleaning_outputs"
FINAL_CLEANED_PATH = OUTPUT_DIR / "claims_cleaned_final.csv"
BENCHMARK_AUDIT_PATH = OUTPUT_DIR / "benchmark_applicability_audit.csv"

# Confirmed via Step 05 investigation: these provider types use a CMS
# pricing system other than the Physician Fee Schedule for the bulk of
# their billed procedures (facility/bundled rates, not RVU-based).
NON_PFS_PROVIDER_TYPES = [
    "Ambulatory Surgical Center",
    "Nuclear Medicine",
    "Independent Diagnostic Testing Facility (IDTF)",
]


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD claims_cleaned.csv")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ============================================================
# INVESTIGATE AND CONFIRM THE PATTERN
# ============================================================
print_section("CONFIRMING THE FACILITY-PRICING PATTERN")

print("Mean Payment Gap % for the three flagged provider types:")
for ptype in NON_PFS_PROVIDER_TYPES:
    subset = df[df["Rndrng_Prvdr_Type"] == ptype]
    if len(subset) > 0:
        print(f"  {ptype:<50} n={len(subset):>7,}  mean gap={subset['Payment_Gap_Pct'].mean():>8.1f}%")
    else:
        print(f"  {ptype:<50} NOT FOUND in dataset")

print("""
Confirmed: actual allowed amounts for these provider types are consistently
2-3x the physician-fee-schedule-calculated expected payment, concentrated
on major procedure codes (e.g. HCPCS 29827 shoulder arthroscopy showed
$4,232 actual vs $1,145 physician-schedule expected). This is a benchmark
mismatch, not a real overpayment finding.
""")


# ============================================================
# APPLY THE FLAG
# ============================================================
print_section("APPLYING benchmark_applicable FLAG")

df["benchmark_applicable"] = ~df["Rndrng_Prvdr_Type"].isin(NON_PFS_PROVIDER_TYPES)

not_applicable_count = (~df["benchmark_applicable"]).sum()
applicable_count = df["benchmark_applicable"].sum()

print(f"Rows where physician fee schedule benchmark IS applicable  : {applicable_count:,} ({applicable_count/len(df)*100:.2f}%)")
print(f"Rows where benchmark is NOT applicable (facility pricing)  : {not_applicable_count:,} ({not_applicable_count/len(df)*100:.2f}%)")

print("""
Decision: rows are KEPT in claims_cleaned_final.csv with this flag set.
Step 06 onward should filter to benchmark_applicable == True before
calculating any underpayment rate, dollar total, or training the
collection probability model -- unless the analysis is specifically about
these facility-priced provider types using their own correct benchmark
(out of scope for this project; would require the separate ASC Payment
System file, which was not part of the original data sources).
""")


# ============================================================
# RECOMPUTE HEADLINE STATS WITH THE FLAG APPLIED
# ============================================================
print_section("HEADLINE STATS -- BEFORE VS AFTER THIS FLAG")

before = df
after = df[df["benchmark_applicable"]]

comparison = pd.DataFrame([
    {"Metric": "Total rows", "Before_Flag": len(before), "After_Flag": len(after)},
    {"Metric": "Mean Payment_Gap_Pct", "Before_Flag": round(before["Payment_Gap_Pct"].mean(), 2),
     "After_Flag": round(after["Payment_Gap_Pct"].mean(), 2)},
    {"Metric": "Underpaid rate (%)", "Before_Flag": round(before["Is_Underpaid"].mean() * 100, 2),
     "After_Flag": round(after["Is_Underpaid"].mean() * 100, 2)},
    {"Metric": "Total estimated underpayment ($)",
     "Before_Flag": round(before.loc[before["Is_Underpaid"], "Total_Dollar_Gap"].sum(), 2),
     "After_Flag": round(after.loc[after["Is_Underpaid"], "Total_Dollar_Gap"].sum(), 2)},
])
print(comparison.to_string(index=False))


# ============================================================
# SAVE OUTPUTS
# ============================================================
print_section("SAVING OUTPUTS")

df.to_csv(FINAL_CLEANED_PATH, index=False)
print(f"Saved: {FINAL_CLEANED_PATH}")

audit_rows = []
for ptype in NON_PFS_PROVIDER_TYPES:
    subset = df[df["Rndrng_Prvdr_Type"] == ptype]
    audit_rows.append({
        "Provider_Type": ptype,
        "Row_Count": len(subset),
        "Mean_Gap_Pct": round(subset["Payment_Gap_Pct"].mean(), 2) if len(subset) > 0 else None,
        "Reason_Excluded": "Uses ASC Payment System or bundled facility rates, not Physician Fee Schedule RVU formula",
    })
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(BENCHMARK_AUDIT_PATH, index=False)
print(f"Saved: {BENCHMARK_AUDIT_PATH}")

print_section("STEP 04D COMPLETE")
print(f"""
claims_cleaned_final.csv is now the trusted base file for Step 06 onward.

IMPORTANT for Step 06/07: filter on benchmark_applicable == True before
computing any underpayment-based feature or target value.
""")