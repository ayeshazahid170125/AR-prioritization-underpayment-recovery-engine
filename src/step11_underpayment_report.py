"""
STEP 11 - Generate Underpayment Report
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step11_underpayment_report.py

Purpose:
Produce the business-ready underpayment report required by the project
brief: "Underpayment report on real CMS data showing $ gap by payer and
CPT code." This reads the Step 09 AR priority queue output (already
filtered to underpaid, benchmark-applicable rows) and builds summary
tables by state, HCPCS code, and provider type, plus a top-N report and
an executive summary.

Design note: this pipeline does not have a real multi-payer "payer" field
(CMS Provider Payment PUF is Medicare-only -- see Step 06's payer_type_proxy
disclosure). "By payer" in the brief is satisfied here via payer_type_proxy
(Medicare Participating vs Non-Participating), which is the only genuine
payer-relevant signal in this dataset.
"""

from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
QUEUE_PATH = BASE_DIR / "ar_priority_outputs" / "ar_priority_queue.csv"
FEATURE_PATH = BASE_DIR / "feature_outputs" / "feature_engineered_dataset.csv"

OUTPUT_DIR = BASE_DIR / "report_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_N = 500


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD AR PRIORITY QUEUE (STEP 09 OUTPUT)")

if not QUEUE_PATH.exists():
    raise FileNotFoundError(f"File not found: {QUEUE_PATH}\nRun step09_ar_priority_queue.py first.")

queue_df = pd.read_csv(QUEUE_PATH, low_memory=False)
print(f"Loaded: {queue_df.shape[0]:,} rows x {queue_df.shape[1]} columns")

# Bring in payer_type_proxy from the feature-engineered file (Step 09's
# output doesn't carry it, but we need it for the "by payer" requirement).
print_section("JOINING payer_type_proxy FOR 'BY PAYER' REPORTING")

if FEATURE_PATH.exists():
    payer_lookup = pd.read_csv(
        FEATURE_PATH,
        usecols=["Rndrng_NPI", "HCPCS_Cd", "Place_Of_Srvc", "payer_type_proxy"],
        low_memory=False,
    ).drop_duplicates(subset=["Rndrng_NPI", "HCPCS_Cd", "Place_Of_Srvc"])

    queue_df = queue_df.merge(
        payer_lookup,
        on=["Rndrng_NPI", "HCPCS_Cd", "Place_Of_Srvc"],
        how="left",
    )
    matched_payer = queue_df["payer_type_proxy"].notna().sum()
    print(f"Matched payer_type_proxy for {matched_payer:,} / {len(queue_df):,} rows "
          f"({matched_payer / len(queue_df) * 100:.2f}%)")
else:
    print("WARNING: feature_engineered_dataset.csv not found -- payer_type_proxy will be 'Unknown'.")
    queue_df["payer_type_proxy"] = "Unknown"

queue_df["payer_type_proxy"] = queue_df["payer_type_proxy"].fillna("Unknown")


# ============================================================
# TOP N UNDERPAYMENTS REPORT
# ============================================================
print_section(f"TOP {TOP_N} UNDERPAYMENTS")

top_underpayments = queue_df.sort_values("estimated_recovery", ascending=False).head(TOP_N).copy()
top_underpayments.insert(0, "report_rank", range(1, len(top_underpayments) + 1))

top_path = OUTPUT_DIR / "top_underpayments.csv"
top_underpayments.to_csv(top_path, index=False)
print(f"Saved: {top_path}")
print(f"\nTop 10 preview:")
print(top_underpayments[["report_rank", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn",
                           "HCPCS_Cd", "estimated_recovery", "priority_tier"]].head(10).to_string(index=False))


# ============================================================
# SUMMARY BY HCPCS CODE
# ============================================================
print_section("UNDERPAYMENT SUMMARY BY HCPCS CODE")

hcpcs_summary = (
    queue_df.groupby("HCPCS_Cd")
    .agg(
        underpaid_rows=("HCPCS_Cd", "size"),
        total_estimated_recovery=("estimated_recovery", "sum"),
        avg_gap_pct=("Payment_Gap_Pct", "mean"),
        avg_confidence=("confidence_score", "mean"),
        critical_or_high_count=("priority_tier", lambda s: s.isin(["Critical", "High"]).sum()),
    )
    .reset_index()
    .sort_values("total_estimated_recovery", ascending=False)
)
hcpcs_summary["total_estimated_recovery"] = hcpcs_summary["total_estimated_recovery"].round(2)
hcpcs_summary["avg_gap_pct"] = hcpcs_summary["avg_gap_pct"].round(2)
hcpcs_summary["avg_confidence"] = hcpcs_summary["avg_confidence"].round(4)

hcpcs_path = OUTPUT_DIR / "underpayment_summary_by_hcpcs.csv"
hcpcs_summary.to_csv(hcpcs_path, index=False)
print(f"Saved: {hcpcs_path} ({len(hcpcs_summary):,} unique HCPCS codes)")
print("\nTop 15 HCPCS codes by total estimated recovery:")
print(hcpcs_summary.head(15).to_string(index=False))


# ============================================================
# SUMMARY BY STATE
# ============================================================
print_section("UNDERPAYMENT SUMMARY BY STATE")

state_summary = (
    queue_df.groupby("Rndrng_Prvdr_State_Abrvtn")
    .agg(
        underpaid_rows=("Rndrng_Prvdr_State_Abrvtn", "size"),
        total_estimated_recovery=("estimated_recovery", "sum"),
        avg_gap_pct=("Payment_Gap_Pct", "mean"),
        critical_or_high_count=("priority_tier", lambda s: s.isin(["Critical", "High"]).sum()),
    )
    .reset_index()
    .rename(columns={"Rndrng_Prvdr_State_Abrvtn": "provider_state"})
    .sort_values("total_estimated_recovery", ascending=False)
)
state_summary["total_estimated_recovery"] = state_summary["total_estimated_recovery"].round(2)
state_summary["avg_gap_pct"] = state_summary["avg_gap_pct"].round(2)

state_path = OUTPUT_DIR / "underpayment_summary_by_state.csv"
state_summary.to_csv(state_path, index=False)
print(f"Saved: {state_path} ({len(state_summary):,} states)")
print("\nTop 15 states by total estimated recovery:")
print(state_summary.head(15).to_string(index=False))


# ============================================================
# SUMMARY BY PROVIDER TYPE
# ============================================================
print_section("UNDERPAYMENT SUMMARY BY PROVIDER TYPE")

provider_type_summary = (
    queue_df.groupby("Rndrng_Prvdr_Type")
    .agg(
        underpaid_rows=("Rndrng_Prvdr_Type", "size"),
        total_estimated_recovery=("estimated_recovery", "sum"),
        avg_gap_pct=("Payment_Gap_Pct", "mean"),
        critical_or_high_count=("priority_tier", lambda s: s.isin(["Critical", "High"]).sum()),
    )
    .reset_index()
    .rename(columns={"Rndrng_Prvdr_Type": "provider_type"})
    .sort_values("total_estimated_recovery", ascending=False)
)
provider_type_summary["total_estimated_recovery"] = provider_type_summary["total_estimated_recovery"].round(2)
provider_type_summary["avg_gap_pct"] = provider_type_summary["avg_gap_pct"].round(2)

provider_path = OUTPUT_DIR / "underpayment_summary_by_provider_type.csv"
provider_type_summary.to_csv(provider_path, index=False)
print(f"Saved: {provider_path} ({len(provider_type_summary):,} provider types)")
print("\nTop 15 provider types by total estimated recovery:")
print(provider_type_summary.head(15).to_string(index=False))


# ============================================================
# SUMMARY BY PAYER TYPE PROXY ("BY PAYER" REQUIREMENT)
# ============================================================
print_section("UNDERPAYMENT SUMMARY BY PAYER TYPE (PROXY)")
print("""
NOTE: 'payer_type_proxy' reflects Medicare participation status, the only
real payer-relevant signal in this Medicare-only public dataset. This is
not a multi-payer (commercial/Medicaid/Medicare Advantage) breakdown -- see
Step 06's documented limitation. Shown here to satisfy the brief's "by
payer" requirement as honestly as the data allows.
""")

payer_summary = (
    queue_df.groupby("payer_type_proxy")
    .agg(
        underpaid_rows=("payer_type_proxy", "size"),
        total_estimated_recovery=("estimated_recovery", "sum"),
        avg_gap_pct=("Payment_Gap_Pct", "mean"),
    )
    .reset_index()
    .sort_values("total_estimated_recovery", ascending=False)
)
payer_summary["total_estimated_recovery"] = payer_summary["total_estimated_recovery"].round(2)

payer_path = OUTPUT_DIR / "underpayment_summary_by_payer_type.csv"
payer_summary.to_csv(payer_path, index=False)
print(f"Saved: {payer_path}")
print(payer_summary.to_string(index=False))


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================
print_section("EXECUTIVE SUMMARY")

total_underpaid_rows = len(queue_df)
total_recovery = queue_df["estimated_recovery"].sum()
critical_rows = (queue_df["priority_tier"] == "Critical").sum()
high_rows = (queue_df["priority_tier"] == "High").sum()
top_state = state_summary.iloc[0]["provider_state"]
top_state_recovery = state_summary.iloc[0]["total_estimated_recovery"]
top_hcpcs = hcpcs_summary.iloc[0]["HCPCS_Cd"]
top_hcpcs_recovery = hcpcs_summary.iloc[0]["total_estimated_recovery"]
top_provider_type = provider_type_summary.iloc[0]["provider_type"]

summary = pd.DataFrame([
    {"metric": "total_underpaid_rows", "value": total_underpaid_rows},
    {"metric": "total_estimated_recovery", "value": round(total_recovery, 2)},
    {"metric": "critical_tier_rows", "value": int(critical_rows)},
    {"metric": "high_tier_rows", "value": int(high_rows)},
    {"metric": "top_state_by_recovery", "value": top_state},
    {"metric": "top_state_recovery_amount", "value": round(top_state_recovery, 2)},
    {"metric": "top_hcpcs_by_recovery", "value": top_hcpcs},
    {"metric": "top_hcpcs_recovery_amount", "value": round(top_hcpcs_recovery, 2)},
    {"metric": "top_provider_type_by_recovery", "value": top_provider_type},
    {"metric": "unique_hcpcs_codes", "value": len(hcpcs_summary)},
    {"metric": "unique_states", "value": len(state_summary)},
    {"metric": "unique_provider_types", "value": len(provider_type_summary)},
])

summary_path = OUTPUT_DIR / "underpayment_report_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")
print(summary.to_string(index=False))

print_section("STEP 11 COMPLETE")
print(f"""
All report files saved to: {OUTPUT_DIR}
  - top_underpayments.csv (top {TOP_N} rows)
  - underpayment_summary_by_hcpcs.csv
  - underpayment_summary_by_state.csv
  - underpayment_summary_by_provider_type.csv
  - underpayment_summary_by_payer_type.csv
  - underpayment_report_summary.csv

Next: Step 12 will build the AR workqueue dashboard using these summary
files, so the dashboard never needs to load the full 5.7M-row queue
directly.
""")