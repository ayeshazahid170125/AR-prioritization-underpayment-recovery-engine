"""
STEP 04B - Outlier Detection (Statistical + Domain Rules + Charts)
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step04b_outlier_detection.py

Purpose:
Detect outliers in the payment variance numbers using multiple statistical
methods AND domain-specific business rules, then visualize the results.
Nothing is removed here -- every method just flags rows for review, exactly
like Project 1's outlier detection step. The final cleaning decision happens
in Step 04c.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "variance_outputs" / "claims_with_variance.csv"

OUTPUT_DIR = BASE_DIR / "cleaning_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

OUTLIER_SUMMARY_PATH = OUTPUT_DIR / "outlier_method_comparison.csv"
OUTLIER_FLAGGED_PATH = OUTPUT_DIR / "outlier_flagged_rows.csv"
HCPCS_OUTLIER_SUMMARY_PATH = OUTPUT_DIR / "outlier_by_hcpcs_summary.csv"

NUMERIC_COLS = ["Payment_Gap", "Payment_Gap_Pct", "Total_Dollar_Gap"]


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD claims_with_variance.csv")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"File not found: {INPUT_PATH}\nRun step03_join_actual_expected.py first.")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

for col in NUMERIC_COLS:
    if col not in df.columns:
        raise KeyError(f"Expected column not found: {col}")

# Handle infinite values caused by dividing by a $0 Expected Payment.
# This happens when Expected_Payment_NonFacility_Avg is 0 (no fee schedule
# rate could be calculated for that code/state -- a data issue, not a real
# 100% underpayment). We exclude these from outlier detection and flag them
# separately, since "inf" would otherwise corrupt every statistical method.
inf_mask = np.isinf(df["Payment_Gap_Pct"])
inf_count = inf_mask.sum()
print(f"\nRows with infinite Payment_Gap_Pct (Expected Payment = $0): {inf_count:,}")
print("These rows are excluded from outlier statistics below and will be")
print("flagged separately as 'zero_expected_payment' in Step 04c cleaning.")

df["zero_expected_payment_flag"] = inf_mask
df_for_outliers = df[~inf_mask].copy()


# ============================================================
# METHOD 1 - IQR (Interquartile Range)
# ============================================================
def flag_iqr(series, multiplier=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (series < lower) | (series > upper), lower, upper


# ============================================================
# METHOD 2 - Z-SCORE
# ============================================================
def flag_zscore(series, threshold=3.0):
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.Series(False, index=series.index), mean, mean
    z = (series - mean) / std
    return z.abs() > threshold, mean - threshold * std, mean + threshold * std


# ============================================================
# METHOD 3 - PERCENTILE (P1 / P99)
# ============================================================
def flag_percentile(series, lower_pct=1, upper_pct=99):
    lower, upper = series.quantile(lower_pct / 100), series.quantile(upper_pct / 100)
    return (series < lower) | (series > upper), lower, upper


# ============================================================
# METHOD 4 - MODIFIED Z-SCORE (median-based, more robust to skew)
# ============================================================
def flag_modified_zscore(series, threshold=3.5):
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index), median, median
    modified_z = 0.6745 * (series - median) / mad
    lower = median - (threshold / 0.6745) * mad
    upper = median + (threshold / 0.6745) * mad
    return modified_z.abs() > threshold, lower, upper


# ============================================================
# METHOD 5 - DOMAIN RULE (specific to this project)
# ============================================================
def flag_domain_rule(gap_pct_series):
    """
    Domain knowledge for Medicare payment variance: a genuine fee-schedule
    variance rarely exceeds +/-100% of the expected rate. Anything beyond
    that is far more likely to be a data quality issue (wrong code match,
    bundled service, multiple-unit billing) than a real underpayment.
    """
    return (gap_pct_series < -100) | (gap_pct_series > 200)


# ============================================================
# RUN ALL METHODS ON EACH NUMERIC COLUMN
# ============================================================
print_section("RUNNING OUTLIER DETECTION METHODS")

method_results = {}
comparison_rows = []

for col in NUMERIC_COLS:
    series = df_for_outliers[col].dropna()
    print(f"\nColumn: {col}")
    print(f"  Range: [{series.min():,.2f}, {series.max():,.2f}]  Mean: {series.mean():,.2f}  Median: {series.median():,.2f}")

    iqr_flag, iqr_lo, iqr_hi = flag_iqr(series)
    z_flag, z_lo, z_hi = flag_zscore(series)
    pct_flag, pct_lo, pct_hi = flag_percentile(series)
    modz_flag, modz_lo, modz_hi = flag_modified_zscore(series)

    method_results[col] = {
        "IQR": iqr_flag, "ZScore": z_flag,
        "Percentile": pct_flag, "ModifiedZ": modz_flag,
    }

    for method_name, flag_series, lo, hi in [
        ("IQR", iqr_flag, iqr_lo, iqr_hi),
        ("Z-Score", z_flag, z_lo, z_hi),
        ("Percentile (1/99)", pct_flag, pct_lo, pct_hi),
        ("Modified Z-Score", modz_flag, modz_lo, modz_hi),
    ]:
        count = flag_series.sum()
        pct = count / len(series) * 100
        print(f"  {method_name:<20} bounds=[{lo:,.2f}, {hi:,.2f}]  flagged={count:,} ({pct:.2f}%)")
        comparison_rows.append({
            "Column": col, "Method": method_name,
            "Lower_Bound": round(lo, 2), "Upper_Bound": round(hi, 2),
            "Flagged_Count": int(count), "Flagged_Pct": round(pct, 2),
        })

# Domain rule only applies to the percentage column
domain_flag = flag_domain_rule(df_for_outliers["Payment_Gap_Pct"])
domain_count = domain_flag.sum()
domain_pct = domain_count / len(df_for_outliers) * 100
print(f"\nDomain rule (gap % beyond -100%/+200%): flagged={domain_count:,} ({domain_pct:.2f}%)")
comparison_rows.append({
    "Column": "Payment_Gap_Pct", "Method": "Domain Rule (CMS context)",
    "Lower_Bound": -100, "Upper_Bound": 200,
    "Flagged_Count": int(domain_count), "Flagged_Pct": round(domain_pct, 2),
})

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTLIER_SUMMARY_PATH, index=False)
print(f"\nSaved method comparison: {OUTLIER_SUMMARY_PATH}")


# ============================================================
# FINAL DECISION LOGIC -- WHICH METHOD TO TRUST
# ============================================================
print_section("FINAL DECISION -- WHICH OUTLIER FLAG TO USE")
print("""
Decision: use the DOMAIN RULE as the primary outlier flag for this project,
not a purely statistical method. Reason: Payment_Gap_Pct is bounded by real
business logic (a true fee-schedule variance is rarely beyond -100%/+200%),
so a domain-aware threshold is more defensible than an arbitrary statistical
cutoff for a CMS payment dataset. The four statistical methods above are
kept for transparency and comparison, but ROW-LEVEL FLAGGING in Step 04c
will be based on the domain rule plus the Modified Z-Score (most robust to
skewed financial data) as a secondary signal.
""")

df["outlier_domain_rule"] = False
df.loc[domain_flag.index, "outlier_domain_rule"] = domain_flag.values

df["outlier_modified_zscore"] = False
modz_idx = method_results["Payment_Gap_Pct"]["ModifiedZ"].index
df.loc[modz_idx, "outlier_modified_zscore"] = method_results["Payment_Gap_Pct"]["ModifiedZ"].values

# Rows with zero expected payment are their own category, not double-counted
# inside the statistical outlier flags (since they were excluded above).
df["outlier_any_flag"] = (
    df["outlier_domain_rule"] | df["outlier_modified_zscore"] | df["zero_expected_payment_flag"]
)
both_flagged = (df["outlier_domain_rule"] & df["outlier_modified_zscore"]).sum()
either_flagged = df["outlier_any_flag"].sum()

print(f"Flagged by domain rule only            : {(df['outlier_domain_rule'] & ~df['outlier_modified_zscore']).sum():,}")
print(f"Flagged by modified z-score only         : {(~df['outlier_domain_rule'] & df['outlier_modified_zscore']).sum():,}")
print(f"Flagged by BOTH methods (high confidence): {both_flagged:,}")
print(f"Flagged by EITHER method (total review)  : {either_flagged:,} ({either_flagged/len(df)*100:.2f}%)")


# ============================================================
# HCPCS-LEVEL OUTLIER PATTERN CHECK
# ============================================================
print_section("WHICH HCPCS CODES HAVE THE MOST OUTLIERS")

# Use only non-infinite rows for the Mean_Gap_Pct calculation, so a handful
# of $0-expected-payment codes don't make every average show as "inf".
hcpcs_outlier_summary = (
    df.groupby("HCPCS_Cd")
    .agg(
        Total_Rows=("HCPCS_Cd", "size"),
        Outlier_Rows=("outlier_any_flag", "sum"),
        Zero_Expected_Rows=("zero_expected_payment_flag", "sum"),
    )
    .reset_index()
)
mean_gap_by_code = (
    df_for_outliers.groupby("HCPCS_Cd")["Payment_Gap_Pct"].mean().rename("Mean_Gap_Pct")
)
hcpcs_outlier_summary = hcpcs_outlier_summary.merge(mean_gap_by_code, on="HCPCS_Cd", how="left")
hcpcs_outlier_summary["Outlier_Rate_Pct"] = (
    hcpcs_outlier_summary["Outlier_Rows"] / hcpcs_outlier_summary["Total_Rows"] * 100
)
hcpcs_outlier_summary = hcpcs_outlier_summary[hcpcs_outlier_summary["Total_Rows"] >= 30]
hcpcs_outlier_summary = hcpcs_outlier_summary.sort_values("Outlier_Rate_Pct", ascending=False)

print("Top 15 HCPCS codes by outlier rate (min 30 claims):")
print(hcpcs_outlier_summary.head(15).to_string(index=False))

hcpcs_outlier_summary.to_csv(HCPCS_OUTLIER_SUMMARY_PATH, index=False)
print(f"\nSaved: {HCPCS_OUTLIER_SUMMARY_PATH}")


# ============================================================
# SAVE FLAGGED ROWS SAMPLE
# ============================================================
flagged_sample = df[df["outlier_any_flag"]].head(500)
flagged_sample.to_csv(OUTLIER_FLAGGED_PATH, index=False)
print(f"\nSaved 500-row sample of flagged outlier rows: {OUTLIER_FLAGGED_PATH}")


# ============================================================
# CHARTS
# ============================================================
print_section("GENERATING CHARTS")

plt.style.use("default")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Payment Gap % Distribution -- Outlier Detection", fontsize=14, fontweight="bold")

# Chart 1: Full distribution histogram
ax = axes[0, 0]
ax.hist(df_for_outliers["Payment_Gap_Pct"].clip(-200, 300), bins=100, color="#2563EB", alpha=0.7)
ax.axvline(-100, color="red", linestyle="--", label="Domain rule bounds")
ax.axvline(200, color="red", linestyle="--")
ax.set_title("Distribution of Payment Gap % (clipped for display)")
ax.set_xlabel("Payment Gap %")
ax.set_ylabel("Count")
ax.legend()

# Chart 2: Boxplot comparing flagged vs not flagged
ax = axes[0, 1]
flagged_data = df.loc[df["outlier_any_flag"], "Payment_Gap_Pct"].clip(-300, 400)
not_flagged_data = df.loc[~df["outlier_any_flag"], "Payment_Gap_Pct"].clip(-300, 400)
ax.boxplot([not_flagged_data, flagged_data], tick_labels=["Not Flagged", "Flagged"])
ax.set_title("Gap % -- Flagged vs Not Flagged")
ax.set_ylabel("Payment Gap %")

# Chart 3: Method comparison bar chart
ax = axes[1, 0]
method_summary = comparison_df[comparison_df["Column"] == "Payment_Gap_Pct"]
ax.bar(method_summary["Method"], method_summary["Flagged_Pct"], color="#16A34A")
ax.set_title("Flagged % by Method (Payment_Gap_Pct)")
ax.set_ylabel("% of Rows Flagged")
ax.tick_params(axis="x", rotation=30)

# Chart 4: Outlier rate by Place of Service
ax = axes[1, 1]
pos_outlier = df.groupby("Place_Of_Srvc")["outlier_any_flag"].mean() * 100
ax.bar(pos_outlier.index.astype(str), pos_outlier.values, color="#D97706")
ax.set_title("Outlier Rate by Place of Service")
ax.set_ylabel("% Flagged as Outlier")
ax.set_xlabel("Place of Service (O=Office, F=Facility)")

plt.tight_layout()
chart_path = CHARTS_DIR / "outlier_detection_overview.png"
plt.savefig(chart_path, dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved chart: {chart_path}")

print_section("STEP 04B COMPLETE")
print("""
Next: Step 04c will use the outlier_any_flag column created here to decide
what to do with flagged rows -- document them, NOT blindly drop them -- and
produce the final claims_cleaned.csv file.
""")