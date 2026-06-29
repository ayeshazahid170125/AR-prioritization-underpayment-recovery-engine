"""
STEP 03 - Join Actual vs Expected Payment
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step03_join_actual_expected.py

Purpose:
Join the real CMS Provider Payment PUF (actual amounts Medicare paid) with
the Expected Payment Table built in Step 02 (what the fee schedule says
should have been paid), then calculate the gap between them.

Output of this step is the master file most later steps build on:
claims_with_variance.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
RAW_DIR = BASE_DIR / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = BASE_DIR

# Same Medicare Physician & Other Practitioners PUF used in Project 1.
# Adjust this path if your folder name differs.
PUF_PATH = (
    RAW_DIR
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "2023"
    / "MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv"
)

if not PUF_PATH.exists():
    puf_root = RAW_DIR / "Medicare Physician & Other Practitioners - by Provider and Service"
    puf_candidates = sorted(puf_root.rglob("*Prov_Svc.csv")) if puf_root.exists() else []
    if puf_candidates:
        PUF_PATH = puf_candidates[-1]

FEE_SCHEDULE_DIR = BASE_DIR / "fee_schedule_outputs"
EXPECTED_PAYMENT_PATH = FEE_SCHEDULE_DIR / "expected_payment_table.csv"

OUTPUT_DIR = BASE_DIR / "variance_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CLAIMS_WITH_VARIANCE_PATH = OUTPUT_DIR / "claims_with_variance.csv"
JOIN_AUDIT_PATH = OUTPUT_DIR / "join_audit_report.csv"
UNMATCHED_SAMPLE_PATH = OUTPUT_DIR / "unmatched_rows_sample.csv"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# STEP 1 - LOAD ACTUAL PAYMENTS (CMS PUF)
# ============================================================
print_section("LOADING ACTUAL PAYMENTS (CMS PROVIDER PAYMENT PUF)")

if not PUF_PATH.exists():
    raise FileNotFoundError(
        f"PUF file not found at: {PUF_PATH}\n"
        "Update PUF_PATH in the CONFIGURATION section to point at your "
        "Medicare Physician & Other Practitioners CSV file."
    )

puf_df = pd.read_csv(PUF_PATH, low_memory=False)
print(f"Loaded PUF: {puf_df.shape[0]:,} rows x {puf_df.shape[1]} columns")

required_puf_cols = [
    "Rndrng_NPI", "Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_Type",
    "HCPCS_Cd", "Place_Of_Srvc", "Tot_Benes", "Tot_Srvcs",
    "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Pymt_Amt",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind", "Rndrng_Prvdr_Ent_Cd",
]
missing_cols = [c for c in required_puf_cols if c not in puf_df.columns]
if missing_cols:
    raise KeyError(f"PUF file is missing expected columns: {missing_cols}")

print(f"\nActual ALLOWED amount summary (Avg_Mdcr_Alowd_Amt):")
print(puf_df["Avg_Mdcr_Alowd_Amt"].describe().to_string())
print("""
IMPORTANT: We compare against Avg_Mdcr_Alowd_Amt (the CMS-approved rate),
not Avg_Mdcr_Pymt_Amt (Medicare's 80% share after patient co-insurance).
The fee schedule formula calculates the full approved rate, so Allowed
Amount is the correct apples-to-apples benchmark. Using Payment Amount
instead would falsely show ~70-75% "underpayment" across nearly all
claims, which was confirmed as a measurement error during manual
verification against the CMS PFS Look-up Tool, not a real finding.
""")


# ============================================================
# STEP 2 - LOAD EXPECTED PAYMENT TABLE (FROM STEP 02)
# ============================================================
print_section("LOADING EXPECTED PAYMENT TABLE (STEP 02 OUTPUT)")

if not EXPECTED_PAYMENT_PATH.exists():
    raise FileNotFoundError(
        f"Expected payment table not found at: {EXPECTED_PAYMENT_PATH}\n"
        "Run step02_expected_payment.py first."
    )

expected_df = pd.read_csv(EXPECTED_PAYMENT_PATH, low_memory=False)
print(f"Loaded expected payment table: {expected_df.shape[0]:,} rows x {expected_df.shape[1]} columns")
print(f"Unique HCPCS codes: {expected_df['HCPCS'].nunique():,}")
print(f"Unique states: {expected_df['State'].nunique():,}")


# ============================================================
# STEP 3 - HANDLE MULTIPLE LOCALITIES PER STATE
# ============================================================
print_section("COLLAPSING MULTIPLE LOCALITIES PER STATE")
print("""
Several states have more than one Medicare locality (e.g., California has
locality-specific rates for Los Angeles, San Francisco, etc.), but the CMS
Provider Payment PUF only identifies the provider's state, not their exact
locality. To make an honest, conservative comparison, we use the AVERAGE
expected payment across all localities within a state as the benchmark for
that state. This means our underpayment flags are a reasonable estimate,
not an exact locality-level match -- this limitation will be documented in
the model card for this project.
""")

state_avg_expected = (
    expected_df
    .groupby(["HCPCS", "State"], as_index=False)
    .agg(
        Expected_Payment_NonFacility_Avg=("Expected_Payment_NonFacility", "mean"),
        Expected_Payment_Facility_Avg=("Expected_Payment_Facility", "mean"),
        Expected_Payment_NonFacility_Min=("Expected_Payment_NonFacility", "min"),
        Expected_Payment_NonFacility_Max=("Expected_Payment_NonFacility", "max"),
        Locality_Count=("Locality_Number", "nunique"),
    )
)

print(f"Collapsed table: {state_avg_expected.shape[0]:,} rows (HCPCS x State combinations)")
print("\nSample rows:")
print(state_avg_expected.head(10).to_string())


# ============================================================
# STEP 4 - JOIN ACTUAL (PUF) WITH EXPECTED (FEE SCHEDULE)
# ============================================================
print_section("JOINING ACTUAL PAYMENTS WITH EXPECTED PAYMENTS")

print(f"PUF rows before join: {len(puf_df):,}")

joined_df = puf_df.merge(
    state_avg_expected,
    left_on=["HCPCS_Cd", "Rndrng_Prvdr_State_Abrvtn"],
    right_on=["HCPCS", "State"],
    how="left",
)

matched = joined_df["Expected_Payment_NonFacility_Avg"].notna().sum()
unmatched = joined_df["Expected_Payment_NonFacility_Avg"].isna().sum()
match_rate = matched / len(joined_df) * 100

print(f"\nRows matched to a fee schedule rate : {matched:,} ({match_rate:.2f}%)")
print(f"Rows NOT matched (no fee schedule rate found): {unmatched:,} ({100 - match_rate:.2f}%)")
print("""
Unmatched rows are expected and honest -- not every HCPCS code in the PUF
has a corresponding payable rate in the physician fee schedule (for example,
lab codes, drug codes (J-codes) at certain CMS pricing, and codes with a
"bundled" or "not separately priced" status are common reasons for no
match). These rows will be excluded from variance analysis since there is
no valid benchmark to compare them against.
""")


# ============================================================
# STEP 5 - AUDIT WHY ROWS DIDN'T MATCH
# ============================================================
print_section("AUDITING UNMATCHED ROWS")

unmatched_df = joined_df[joined_df["Expected_Payment_NonFacility_Avg"].isna()]

if len(unmatched_df) > 0:
    print("Top 15 HCPCS codes most frequently unmatched:")
    top_unmatched_codes = (
        unmatched_df["HCPCS_Cd"].value_counts().head(15)
    )
    print(top_unmatched_codes.to_string())

    unmatched_sample = unmatched_df[
        ["HCPCS_Cd", "Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_Type", "Avg_Mdcr_Pymt_Amt"]
    ].head(200)
    unmatched_sample.to_csv(UNMATCHED_SAMPLE_PATH, index=False)
    print(f"\nSaved a 200-row sample of unmatched rows: {UNMATCHED_SAMPLE_PATH}")


# ============================================================
# STEP 6 - CALCULATE THE PAYMENT GAP
# ============================================================
print_section("CALCULATING PAYMENT GAP (ACTUAL - EXPECTED)")

# Only calculate variance for rows that actually matched a fee schedule rate.
matched_df = joined_df[joined_df["Expected_Payment_NonFacility_Avg"].notna()].copy()

# FIX (methodology correction): select the facility or non-facility benchmark
# based on Place_Of_Srvc, instead of always using the non-facility benchmark
# regardless of setting. CMS pays a DIFFERENT (typically lower) PE RVU for
# facility-setting claims (Place_Of_Srvc == "F") because the facility itself
# absorbs part of the practice expense overhead. Comparing a facility claim
# against the non-facility benchmark systematically and falsely inflates the
# apparent underpayment for every facility-setting row.
matched_df["Expected_Payment_Used"] = np.where(
    matched_df["Place_Of_Srvc"] == "F",
    matched_df["Expected_Payment_Facility_Avg"],
    matched_df["Expected_Payment_NonFacility_Avg"],
)

matched_df["Payment_Gap"] = (
    matched_df["Avg_Mdcr_Alowd_Amt"] - matched_df["Expected_Payment_Used"]
)
matched_df["Payment_Gap_Pct"] = (
    matched_df["Payment_Gap"] / matched_df["Expected_Payment_Used"] * 100
)
matched_df["Is_Underpaid"] = matched_df["Payment_Gap"] < 0
matched_df["Total_Dollar_Gap"] = matched_df["Payment_Gap"] * matched_df["Tot_Srvcs"]

print("Payment Gap summary statistics:")
print(matched_df["Payment_Gap"].describe().to_string())

underpaid_count = matched_df["Is_Underpaid"].sum()
underpaid_pct = underpaid_count / len(matched_df) * 100
print(f"\nRows flagged as underpaid (Actual < Expected): {underpaid_count:,} ({underpaid_pct:.2f}%)")

total_underpaid_dollars = matched_df.loc[matched_df["Is_Underpaid"], "Total_Dollar_Gap"].sum()
print(f"Total estimated underpayment across all flagged rows: ${total_underpaid_dollars:,.2f}")
print("""
NOTE: this total reflects the gap between fee-schedule-calculated rates and
PUF average actual payments, aggregated across all providers and claims in
this dataset. It is a portfolio-scale estimate, not a single practice's
real recoverable amount -- that distinction will be made clear in any
client-facing report.
""")


# ============================================================
# STEP 7 - SAVE OUTPUTS
# ============================================================
print_section("SAVING OUTPUTS")

output_cols = [
    "Rndrng_NPI", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn",
    "HCPCS_Cd", "Place_Of_Srvc", "Tot_Benes", "Tot_Srvcs",
    "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Pymt_Amt",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind", "Rndrng_Prvdr_Ent_Cd",
    "Expected_Payment_NonFacility_Avg", "Expected_Payment_Facility_Avg",
    "Expected_Payment_Used", "Locality_Count", "Payment_Gap", "Payment_Gap_Pct",
    "Is_Underpaid", "Total_Dollar_Gap",
]
matched_df[output_cols].to_csv(CLAIMS_WITH_VARIANCE_PATH, index=False)
print(f"Saved: {CLAIMS_WITH_VARIANCE_PATH}")
print(f"Final shape: {matched_df.shape[0]:,} rows x {len(output_cols)} columns")

audit_summary = pd.DataFrame([
    {"metric": "Total PUF rows", "value": len(puf_df)},
    {"metric": "Rows matched to fee schedule", "value": matched},
    {"metric": "Rows unmatched", "value": unmatched},
    {"metric": "Match rate (%)", "value": round(match_rate, 2)},
    {"metric": "Rows flagged underpaid", "value": int(underpaid_count)},
    {"metric": "Underpaid rate among matched (%)", "value": round(underpaid_pct, 2)},
    {"metric": "Total estimated underpayment ($)", "value": round(total_underpaid_dollars, 2)},
])
audit_summary.to_csv(JOIN_AUDIT_PATH, index=False)
print(f"Saved: {JOIN_AUDIT_PATH}")

print_section("STEP 03 COMPLETE")
print("""
Next: Step 04 will clean this file -- documenting nulls, reviewing extreme
gap values, and flagging (not blindly dropping) anything that looks like a
data quality issue rather than a real underpayment signal.
""")
