"""
STEP 02 - Build the Expected Payment Table
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step02_expected_payment.py

Purpose:
Apply the official CMS Physician Fee Schedule formula to calculate the
EXPECTED payment for every HCPCS code in every Medicare locality.

Formula (non-facility setting):
  Expected Payment = [(Work RVU x Work GPCI)
                     + (Non-Facility PE RVU x PE GPCI)
                     + (MP RVU x MP GPCI)] x Conversion Factor

This expected-payment table becomes the benchmark that Step 03 will compare
against real claim payments from the CMS Provider Payment PUF.
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
RVU_DIR = RAW_DIR / "RVU23A"

PPRRVU_PATH = RVU_DIR / "PPRRVU23_JAN.csv"
GPCI_PATH = RVU_DIR / "GPCI2023.csv"

OUTPUT_DIR = BASE_DIR / "fee_schedule_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPECTED_PAYMENT_PATH = OUTPUT_DIR / "expected_payment_table.csv"
SANITY_CHECK_PATH = OUTPUT_DIR / "sanity_check_known_codes.csv"

# 2023 CY Physician Fee Schedule Conversion Factor (published by CMS).
# This value is actually present in the PPRRVU file itself (the "FACTOR"
# column), confirmed as 33.8872 in Step 01b raw output. We still keep this
# constant as a documented fallback/cross-check, but the script will use
# the value read directly from the file as the source of truth.
CONVERSION_FACTOR_2023_FALLBACK = 33.8872


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# STEP 1 - LOAD PPRRVU WITH THE CORRECT HEADER ROW
# ============================================================
print_section("LOADING PPRRVU23_JAN.csv")

# The real header is on row index 9 (0-indexed), confirmed in Step 01b.
# pandas correctly read the true header row directly (header=9), giving us
# the real CMS column names. Several are duplicated (PE RVU appears twice,
# RVU appears twice) because the file has facility vs non-facility versions
# side by side -- pandas auto-suffixes the second occurrence with ".1".
pprrvu_raw = pd.read_csv(
    PPRRVU_PATH,
    header=9,
    encoding="utf-8",
    low_memory=False,
)

print(f"Loaded PPRRVU: {pprrvu_raw.shape[0]:,} rows x {pprrvu_raw.shape[1]} columns")
print(f"Columns found: {pprrvu_raw.columns.tolist()}")

# Rename the real CMS columns to clear names for the rest of this script.
# Confirmed mapping from the printed Step 01b output:
#   RVU       -> Work RVU
#   PE RVU    -> Non-Facility PE RVU
#   PE RVU.1  -> Facility PE RVU
#   RVU.1     -> MP RVU
#   CODE      -> Status code (A/I/R/etc.)
#   FACTOR    -> Conversion Factor (confirmed value 33.8872 in the data itself)
pprrvu_raw = pprrvu_raw.rename(columns={
    "RVU": "WORK_RVU",
    "PE RVU": "NON_FAC_PE_RVU",
    "PE RVU.1": "FACILITY_PE_RVU",
    "RVU.1": "MP_RVU",
    "CODE": "STATUS_CODE",
    "FACTOR": "CONVERSION_FACTOR_FROM_FILE",
})

print("\nFirst 5 rows after column rename:")
print(pprrvu_raw[["HCPCS", "DESCRIPTION", "STATUS_CODE", "WORK_RVU",
                   "NON_FAC_PE_RVU", "FACILITY_PE_RVU", "MP_RVU",
                   "CONVERSION_FACTOR_FROM_FILE"]].head().to_string())


# ============================================================
# STEP 2 - CLEAN PPRRVU NUMERIC COLUMNS
# ============================================================
print_section("CLEANING PPRRVU NUMERIC COLUMNS")

numeric_cols = ["WORK_RVU", "NON_FAC_PE_RVU", "FACILITY_PE_RVU", "MP_RVU"]
for col in numeric_cols:
    before_nulls = pprrvu_raw[col].isna().sum()
    pprrvu_raw[col] = pd.to_numeric(pprrvu_raw[col], errors="coerce")
    after_nulls = pprrvu_raw[col].isna().sum()
    print(f"  {col:<18} non-numeric/blank converted to NaN: {after_nulls - before_nulls:,} "
          f"(total NaN now: {after_nulls:,})")

# Drop rows where HCPCS code itself is missing (these are stray footer/blank rows)
rows_before = len(pprrvu_raw)
pprrvu_clean = pprrvu_raw[pprrvu_raw["HCPCS"].notna()].copy()
rows_after = len(pprrvu_clean)
print(f"\nDropped {rows_before - rows_after:,} rows with no HCPCS code (footer/blank rows).")
print(f"Remaining rows: {rows_after:,}")

# Keep only rows with usable RVU values for the expected-payment calculation
pprrvu_clean = pprrvu_clean.dropna(subset=["WORK_RVU", "MP_RVU"])
print(f"Rows with usable Work RVU and MP RVU: {len(pprrvu_clean):,}")


# ============================================================
# STEP 3 - LOAD GPCI WITH THE CORRECT HEADER ROW
# ============================================================
print_section("LOADING GPCI2023.csv")

gpci_df = pd.read_csv(GPCI_PATH, skiprows=2, encoding="utf-8")
gpci_df = gpci_df.dropna(how="all")

gpci_df = gpci_df.rename(columns={
    "Medicare Administrative Contractor (MAC)": "MAC",
    "State": "State",
    "Locality Number": "Locality_Number",
    "Locality Name": "Locality_Name",
    "2023 PW GPCI (with 1.0 Floor)": "WORK_GPCI",
    "2023 PE GPCI": "PE_GPCI",
    "2023 MP GPCI": "MP_GPCI",
})

for col in ["WORK_GPCI", "PE_GPCI", "MP_GPCI"]:
    gpci_df[col] = pd.to_numeric(gpci_df[col], errors="coerce")

print(f"Loaded GPCI: {gpci_df.shape[0]:,} rows x {gpci_df.shape[1]} columns")
print(gpci_df.head().to_string())


# ============================================================
# STEP 4 - CROSS-JOIN RVU CODES WITH GPCI LOCALITIES
# ============================================================
print_section("BUILDING HCPCS x LOCALITY EXPECTED PAYMENT TABLE")

# This is intentionally a full cross-join: every payable HCPCS code gets a
# row for every Medicare locality, because the fee schedule amount genuinely
# differs by locality. The CMS PUF actual-payment data will later be joined
# back to this table by HCPCS code + state (Step 03).

print(f"HCPCS codes to expand: {pprrvu_clean['HCPCS'].nunique():,}")
print(f"Localities to expand across: {gpci_df['Locality_Number'].nunique():,}")
print("Performing cross-join (this creates a large table)...")

pprrvu_small = pprrvu_clean[["HCPCS", "DESCRIPTION", "STATUS_CODE",
                              "WORK_RVU", "NON_FAC_PE_RVU", "FACILITY_PE_RVU", "MP_RVU"]].copy()
gpci_small = gpci_df[["MAC", "State", "Locality_Number", "Locality_Name",
                       "WORK_GPCI", "PE_GPCI", "MP_GPCI"]].copy()

pprrvu_small["_key"] = 1
gpci_small["_key"] = 1

expected_df = pprrvu_small.merge(gpci_small, on="_key").drop(columns="_key")
print(f"Cross-joined table shape: {expected_df.shape[0]:,} rows x {expected_df.shape[1]} columns")


# ============================================================
# STEP 5 - APPLY THE CMS PAYMENT FORMULA
# ============================================================
print_section("APPLYING CMS PAYMENT FORMULA")

# Use the conversion factor as it appears in the file itself (per-row),
# falling back to the documented constant only if a row is missing it.
conv_factor_values = pprrvu_clean["CONVERSION_FACTOR_FROM_FILE"].dropna().unique()
print(f"Distinct conversion factor values found in file: {conv_factor_values}")

if len(conv_factor_values) == 1:
    CONVERSION_FACTOR_2023 = float(conv_factor_values[0])
    print(f"Using conversion factor confirmed from file: ${CONVERSION_FACTOR_2023}")
else:
    CONVERSION_FACTOR_2023 = CONVERSION_FACTOR_2023_FALLBACK
    print(f"WARNING: multiple/no conversion factor values found in file.")
    print(f"Falling back to documented constant: ${CONVERSION_FACTOR_2023}")

expected_df["Expected_Payment_NonFacility"] = (
    (expected_df["WORK_RVU"] * expected_df["WORK_GPCI"])
    + (expected_df["NON_FAC_PE_RVU"].fillna(0) * expected_df["PE_GPCI"])
    + (expected_df["MP_RVU"] * expected_df["MP_GPCI"])
) * CONVERSION_FACTOR_2023

expected_df["Expected_Payment_Facility"] = (
    (expected_df["WORK_RVU"] * expected_df["WORK_GPCI"])
    + (expected_df["FACILITY_PE_RVU"].fillna(0) * expected_df["PE_GPCI"])
    + (expected_df["MP_RVU"] * expected_df["MP_GPCI"])
) * CONVERSION_FACTOR_2023

print("\nExpected payment summary statistics (non-facility):")
print(expected_df["Expected_Payment_NonFacility"].describe().to_string())


# ============================================================
# STEP 6 - SANITY CHECK AGAINST KNOWN PUBLISHED RATES
# ============================================================
print_section("SANITY CHECK -- KNOWN CODE/LOCALITY COMBINATIONS")
print("""
These are spot-checks only. Compare the printed amounts below against the
CMS PFS Look-up Tool (https://www.cms.gov/medicare/physician-fee-schedule/search)
for the same HCPCS code, locality, and year to confirm the formula is correct
before trusting the full table.
""")

sanity_checks = expected_df[
    (expected_df["HCPCS"] == "99213") & (expected_df["State"].isin(["CA", "TX", "NY"]))
][["HCPCS", "DESCRIPTION", "State", "Locality_Name",
   "Expected_Payment_NonFacility", "Expected_Payment_Facility"]]

if sanity_checks.empty:
    print("No 99213 rows found for CA/TX/NY -- check that HCPCS codes loaded correctly.")
else:
    print(sanity_checks.to_string())
    sanity_checks.to_csv(SANITY_CHECK_PATH, index=False)
    print(f"\nSaved: {SANITY_CHECK_PATH}")


# ============================================================
# SAVE FINAL EXPECTED PAYMENT TABLE
# ============================================================
print_section("SAVING EXPECTED PAYMENT TABLE")

final_cols = [
    "HCPCS", "DESCRIPTION", "STATUS_CODE", "MAC", "State",
    "Locality_Number", "Locality_Name",
    "WORK_RVU", "NON_FAC_PE_RVU", "FACILITY_PE_RVU", "MP_RVU",
    "WORK_GPCI", "PE_GPCI", "MP_GPCI",
    "Expected_Payment_NonFacility", "Expected_Payment_Facility",
]
expected_df[final_cols].to_csv(EXPECTED_PAYMENT_PATH, index=False)

print(f"Saved: {EXPECTED_PAYMENT_PATH}")
print(f"Final table: {expected_df.shape[0]:,} rows x {len(final_cols)} columns")

print_section("STEP 02 COMPLETE")
print("""
IMPORTANT - before moving to Step 03:
1. Open sanity_check_known_codes.csv and manually verify at least 2-3 amounts
   against the official CMS PFS Look-up Tool for HCPCS 99213.
2. If the numbers are close but not exact, the Conversion Factor value or a
   GPCI floor/adjustment rule may need correction -- flag this before Step 03.
3. Once verified, Step 03 will join this table to the CMS Provider Payment
   PUF (actual payments) using HCPCS code + State.
""")
