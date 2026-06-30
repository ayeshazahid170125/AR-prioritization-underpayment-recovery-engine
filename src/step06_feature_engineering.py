"""
STEP 06 - Feature Engineering
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step06_feature_engineering.py

Purpose:
Build the features needed for the Step 08 collection probability model:
claim age proxy, procedure category, payer type proxy, and balance size.

Important honesty note (same principle as Project 1's denial-risk target):
The CMS Provider Payment PUF does NOT contain real AR aging data, real
payer type, or real collection outcomes. This file builds DOCUMENTED
PROXY features from what the public data actually contains, and that
limitation will be stated again in the model card -- this is a portfolio
demonstration of the modeling approach, not a claim that these are real
AR aging or payer fields.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "cleaning_outputs" / "claims_cleaned_final.csv"

OUTPUT_DIR = BASE_DIR / "feature_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
FEATURE_PATH = OUTPUT_DIR / "feature_engineered_dataset.csv"
FEATURE_CARD_PATH = OUTPUT_DIR / "feature_definition_card.md"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA -- FILTER TO BENCHMARK-APPLICABLE ROWS ONLY
# ============================================================
print_section("LOAD claims_cleaned_final.csv")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

rows_before_filter = len(df)
df = df[df["benchmark_applicable"]].copy()
print(f"Filtered to benchmark_applicable == True: {len(df):,} rows "
      f"(removed {rows_before_filter - len(df):,} facility-pricing rows)")


# ============================================================
# FEATURE 1 - PROCEDURE CATEGORY (from HCPCS code ranges)
# ============================================================
print_section("FEATURE 1 - PROCEDURE CATEGORY")
print("""
Built from official CPT/HCPCS code range conventions (the same ranges
confirmed during Step 04c investigation), not an invented grouping.
""")

def categorize_procedure(code):
    code = str(code)
    if not code or (not code[0].isalpha() and not code[0].isdigit()):
        return "Other"
    try:
        if code.isdigit():
            num = int(code)
            if 99201 <= num <= 99499:
                return "Evaluation_and_Management"
            if 10000 <= num <= 19999:
                return "Integumentary_Surgery"
            if 20000 <= num <= 29999:
                return "Musculoskeletal_Surgery"
            if 30000 <= num <= 39999:
                return "Respiratory_Cardiovascular_Surgery"
            if 40000 <= num <= 49999:
                return "Digestive_Surgery"
            if 50000 <= num <= 59999:
                return "Urinary_Genital_Surgery"
            if 60000 <= num <= 69999:
                return "Nervous_System_Surgery"
            if 70000 <= num <= 79999:
                return "Radiology"
            if 80000 <= num <= 89999:
                return "Pathology_Laboratory"
            if 90000 <= num <= 99199:
                return "Medicine_Services"
        if code.startswith("J"):
            return "Drugs"
        if code.startswith("G"):
            return "HCPCS_G_Codes"
        if code.startswith("Q"):
            return "HCPCS_Q_Codes_Supplies"
        if code.startswith("A"):
            return "HCPCS_A_Codes_Supplies"
    except (ValueError, IndexError):
        pass
    return "Other"

df["procedure_category"] = df["HCPCS_Cd"].apply(categorize_procedure)

print("Procedure category distribution:")
print(df["procedure_category"].value_counts().to_string())


# ============================================================
# FEATURE 2 - PAYER TYPE PROXY
# ============================================================
print_section("FEATURE 2 - PAYER TYPE PROXY")
print("""
The CMS Provider Payment PUF is exclusively Medicare Fee-for-Service data,
so there is no real "payer type" variation to observe here (no commercial,
Medicaid, or Medicare Advantage claims in this file). The only genuine
payer-relevant signal available is the provider's Medicare participation
status, which affects allowed amounts. This is documented as a proxy, not
a substitute for real multi-payer claims data.
""")

if "Rndrng_Prvdr_Mdcr_Prtcptg_Ind" in df.columns:
    df["payer_type_proxy"] = df["Rndrng_Prvdr_Mdcr_Prtcptg_Ind"].map({
        "Y": "Medicare_Participating",
        "N": "Medicare_NonParticipating",
    }).fillna("Unknown")
    print(df["payer_type_proxy"].value_counts().to_string())
else:
    print("WARNING: Rndrng_Prvdr_Mdcr_Prtcptg_Ind not found in this file.")
    print("This column was dropped during Step 03's column selection.")
    print("Setting payer_type_proxy to 'Medicare_FFS_Only' for all rows.")
    df["payer_type_proxy"] = "Medicare_FFS_Only"


# ============================================================
# FEATURE 3 - CLAIM AGE PROXY
# ============================================================
print_section("FEATURE 3 - CLAIM AGE PROXY (documented limitation)")
print("""
IMPORTANT: the CMS Provider Payment PUF has NO claim submission date, no
payment date, and no AR aging information at all -- it is an annual
aggregate file by provider and HCPCS code. There is no genuine way to
derive a real claim age from this data.

Rather than inventing a fake date, we use Total_Dollar_Gap magnitude and
Payment_Gap_Pct severity as an AGE-RELATED RISK PROXY: in real AR
operations, larger and more severe variances are statistically more likely
to still be open/unresolved (small variances often get corrected quickly
through normal reconciliation; large ones tend to require manual review
and sit longer). This is an honest proxy assumption, not a real aging
field, and will be clearly labeled in the model card.
""")

df["claim_severity_proxy"] = pd.cut(
    df["Payment_Gap_Pct"],
    bins=[-np.inf, -50, -20, -5, np.inf],
    labels=["Severe_Underpayment", "Moderate_Underpayment", "Minor_Underpayment", "At_or_Above_Expected"],
)
print("Claim severity proxy distribution:")
print(df["claim_severity_proxy"].value_counts().to_string())


# ============================================================
# FEATURE 4 - BALANCE SIZE (real, not a proxy)
# ============================================================
print_section("FEATURE 4 - BALANCE SIZE (real field)")
print("""
Total_Dollar_Gap is a real, calculated value (not a proxy) -- it is the
actual dollar difference between allowed and expected payment, multiplied
by total services. This is the genuine "balance at risk" for AR
prioritization purposes.
""")

df["balance_size_bucket"] = pd.cut(
    df["Total_Dollar_Gap"].abs(),
    bins=[0, 100, 1000, 10000, np.inf],
    labels=["Small_<$100", "Medium_$100-1k", "Large_$1k-10k", "Very_Large_>$10k"],
)
print("Balance size bucket distribution:")
print(df["balance_size_bucket"].value_counts().to_string())


# ============================================================
# FEATURE 5 - LOG-TRANSFORM SKEWED NUMERIC FEATURES
# ============================================================
print_section("FEATURE 5 - LOG TRANSFORMS FOR SKEWED FEATURES")

for col in ["Tot_Srvcs", "Tot_Benes", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt"]:
    if col in df.columns:
        new_col = f"{col}_log"
        skew_before = df[col].skew()
        df[new_col] = np.log1p(df[col].clip(lower=0))
        skew_after = df[new_col].skew()
        print(f"  {col:<25} skew {skew_before:>8.2f} -> {skew_after:>8.2f}")


# ============================================================
# FEATURE 6 - MULTICOLLINEARITY DECISION (from Step 05 chart 15/17)
# ============================================================
print_section("FEATURE 6 - MULTICOLLINEARITY DECISION")
print("""
Step 05 EDA (chart 15/17) found Tot_Benes and Tot_Srvcs correlated at 0.73.
This is high but not severe (severe is typically >0.90), so BOTH are kept
as features. However, we add a derived ratio feature (services per
beneficiary) which often carries more independent signal than either raw
count -- a high ratio can indicate repeat/bundled billing patterns that
correlate with documentation or coding-related variance.
""")

df["services_per_beneficiary"] = df["Tot_Srvcs"] / df["Tot_Benes"].replace(0, np.nan)
df["services_per_beneficiary"] = df["services_per_beneficiary"].fillna(0)
print(f"services_per_beneficiary -- mean: {df['services_per_beneficiary'].mean():.2f}, "
      f"median: {df['services_per_beneficiary'].median():.2f}")


# ============================================================
# FEATURE 7 - LEAKAGE-SAFE CONTEXT FEATURES
# ============================================================
print_section("FEATURE 7 - LEAKAGE-SAFE CONTEXT FEATURES")
print("""
We add provider/code/geography context features that do NOT use
Payment_Gap, Payment_Gap_Pct, Total_Dollar_Gap, Is_Underpaid, expected
payment, allowed amount, or Medicare payment amount. This keeps the model
from learning the answer key while still giving it useful utilization and
market-pattern signals.
""")

df["provider_row_count"] = df.groupby("Rndrng_NPI")["HCPCS_Cd"].transform("size")
df["provider_total_services"] = df.groupby("Rndrng_NPI")["Tot_Srvcs"].transform("sum")
df["hcpcs_row_count"] = df.groupby("HCPCS_Cd")["Rndrng_NPI"].transform("size")
df["hcpcs_total_services"] = df.groupby("HCPCS_Cd")["Tot_Srvcs"].transform("sum")
df["state_row_count"] = df.groupby("Rndrng_Prvdr_State_Abrvtn")["HCPCS_Cd"].transform("size")
df["state_total_services"] = df.groupby("Rndrng_Prvdr_State_Abrvtn")["Tot_Srvcs"].transform("sum")

state_hcpcs_rows = df.groupby(
    ["Rndrng_Prvdr_State_Abrvtn", "HCPCS_Cd"]
)["Rndrng_NPI"].transform("size")
df["hcpcs_state_row_share"] = (
    state_hcpcs_rows / df["state_row_count"].replace(0, np.nan)
).fillna(0)

df["avg_charge_per_beneficiary"] = (
    df["Avg_Sbmtd_Chrg"] * df["Tot_Srvcs"] / df["Tot_Benes"].replace(0, np.nan)
).fillna(0)
df["provider_service_share"] = (
    df["Tot_Srvcs"] / df["provider_total_services"].replace(0, np.nan)
).fillna(0)
df["hcpcs_service_share_in_state"] = (
    df["Tot_Srvcs"] / df["state_total_services"].replace(0, np.nan)
).fillna(0)

for col in [
    "provider_row_count", "provider_total_services", "hcpcs_row_count",
    "hcpcs_total_services", "state_row_count", "state_total_services",
    "avg_charge_per_beneficiary",
]:
    new_col = f"{col}_log"
    df[new_col] = np.log1p(df[col].clip(lower=0))
    print(f"  added {col} and {new_col}")


# ============================================================
# SAVE FEATURE-ENGINEERED DATASET
# ============================================================
print_section("SAVING FEATURE-ENGINEERED DATASET")

df.to_csv(FEATURE_PATH, index=False)
print(f"Saved: {FEATURE_PATH}")
print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ============================================================
# WRITE FEATURE DEFINITION CARD (HONESTY DOCUMENTATION)
# ============================================================
card = """# Feature Definition Card -- Project 2 Collection Model

## Real Fields (not proxies)
- Total_Dollar_Gap: actual calculated dollar variance (Allowed - Expected) x Tot_Srvcs
- Payment_Gap_Pct: actual calculated percentage variance
- Tot_Srvcs, Tot_Benes, Avg_Sbmtd_Chrg, Avg_Mdcr_Alowd_Amt: real PUF fields

## Engineered Features
- procedure_category: derived from official CPT/HCPCS numeric code ranges
- services_per_beneficiary: Tot_Srvcs / Tot_Benes (utilization intensity signal)
- provider/state/HCPCS context features: row counts, service totals, and
  service shares based only on volume, charge, code, provider, state, and
  place-of-service information.
- *_log: log1p transforms of skewed numeric fields

## Leakage-Avoided Feature Ideas
The following tempting features are intentionally NOT added because the
target is built from payment variance:
- provider_avg_gap_pct
- hcpcs_historic_underpaid_rate
- state_avg_gap_pct
- charge_to_expected_ratio

They use Payment_Gap_Pct, Is_Underpaid, or Expected_Payment_Used, which
would leak target ingredients back into the model.

## Documented Proxies (IMPORTANT LIMITATION)
- payer_type_proxy: derived from Medicare participation status only.
  The CMS Provider Payment PUF contains ONLY Medicare Fee-for-Service
  claims -- there is no real multi-payer variation in this dataset.
- claim_severity_proxy: derived from Payment_Gap_Pct severity buckets,
  used as a stand-in for claim age/aging risk. The PUF contains NO real
  claim submission date, payment date, or AR aging field. This is an
  honest assumption (larger variances are more likely to remain
  unresolved), not a measured aging value.

## Excluded From Modeling
- Provider rows where benchmark_applicable == False (Ambulatory Surgical
  Center, Nuclear Medicine, Independent Diagnostic Testing Facility) --
  confirmed in Step 04d to use non-physician-fee-schedule pricing systems.

## Next Step
Step 07 will define the actual modeling target (collection probability
proxy) and run an explicit leakage check before any model training begins.
"""
FEATURE_CARD_PATH.write_text(card, encoding="utf-8")
print(f"Saved: {FEATURE_CARD_PATH}")

print_section("STEP 06 COMPLETE")
