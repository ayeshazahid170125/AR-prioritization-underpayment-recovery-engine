"""
STEP 05 - Pre-Model EDA (20 Charts)
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step05_premodel_eda.py

Purpose:
Generate 20 exploratory charts on claims_cleaned.csv (Step 04c output) to
understand payment variance patterns before feature engineering and model
building begin. Mirrors the depth of Project 1's pre-model EDA step.
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
INPUT_PATH = BASE_DIR / "cleaning_outputs" / "claims_cleaned_final.csv"

OUTPUT_DIR = BASE_DIR / "eda_premodel_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 100
COLOR_MAIN = "#2563EB"
COLOR_UNDERPAID = "#DC2626"
COLOR_OK = "#16A34A"
COLOR_NEUTRAL = "#64748B"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_chart(fig, name, chart_num):
    path = CHARTS_DIR / f"{chart_num:02d}_{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{chart_num:02d}] Saved: {path.name}")


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD claims_cleaned.csv")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

chart_num = 1
print_section("GENERATING 20 CHARTS")


# ------------------------------------------------------------
# 1. Distribution of Payment Gap %
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["Payment_Gap_Pct"].clip(-150, 150), bins=80, color=COLOR_MAIN, alpha=0.8)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("1. Distribution of Payment Gap %")
ax.set_xlabel("Payment Gap % (clipped to -150/+150 for display)")
ax.set_ylabel("Count")
save_chart(fig, "payment_gap_pct_distribution", chart_num); chart_num += 1


# ------------------------------------------------------------
# 2. Underpaid vs Not Underpaid (pie)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
counts = df["Is_Underpaid"].value_counts()
ax.pie(counts.values, labels=["Underpaid", "At/Above Expected"], autopct="%1.1f%%",
       colors=[COLOR_UNDERPAID, COLOR_OK], startangle=90)
ax.set_title("2. Share of Claims Underpaid vs Not")
save_chart(fig, "underpaid_share_pie", chart_num); chart_num += 1


# ------------------------------------------------------------
# 3. Mean Gap % by Provider Type (top 15)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
by_type = df.groupby("Rndrng_Prvdr_Type")["Payment_Gap_Pct"].mean().sort_values().head(15)
ax.barh(by_type.index, by_type.values, color=COLOR_UNDERPAID)
ax.set_title("3. Mean Payment Gap % -- 15 Most Underpaid Provider Types")
ax.set_xlabel("Mean Payment Gap %")
save_chart(fig, "gap_by_provider_type_worst15", chart_num); chart_num += 1


# ------------------------------------------------------------
# 4. Mean Gap % by Provider Type (best 15)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
by_type_best = df.groupby("Rndrng_Prvdr_Type")["Payment_Gap_Pct"].mean().sort_values(ascending=False).head(15)
ax.barh(by_type_best.index, by_type_best.values, color=COLOR_OK)
ax.set_title("4. Mean Payment Gap % -- 15 Best-Paid Provider Types")
ax.set_xlabel("Mean Payment Gap %")
save_chart(fig, "gap_by_provider_type_best15", chart_num); chart_num += 1


# ------------------------------------------------------------
# 5. Underpaid Rate by State (top 15)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
by_state = df.groupby("Rndrng_Prvdr_State_Abrvtn")["Is_Underpaid"].mean().sort_values(ascending=False).head(15) * 100
ax.barh(by_state.index, by_state.values, color=COLOR_UNDERPAID)
ax.set_title("5. Underpaid Rate by State -- Top 15 States")
ax.set_xlabel("% of Claims Underpaid")
save_chart(fig, "underpaid_rate_by_state_top15", chart_num); chart_num += 1


# ------------------------------------------------------------
# 6. Office vs Facility Gap % Comparison
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
pos_data = [
    df.loc[df["Place_Of_Srvc"] == "O", "Payment_Gap_Pct"].clip(-150, 150),
    df.loc[df["Place_Of_Srvc"] == "F", "Payment_Gap_Pct"].clip(-150, 150),
]
ax.boxplot(pos_data, tick_labels=["Office", "Facility"])
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("6. Payment Gap % -- Office vs Facility")
ax.set_ylabel("Payment Gap %")
save_chart(fig, "gap_office_vs_facility", chart_num); chart_num += 1


# ------------------------------------------------------------
# 7. Top 15 Most Underpaid HCPCS Codes (by avg gap %, min volume)
# ------------------------------------------------------------
hcpcs_stats = df.groupby("HCPCS_Cd").agg(
    Mean_Gap_Pct=("Payment_Gap_Pct", "mean"),
    Count=("HCPCS_Cd", "size"),
)
hcpcs_stats = hcpcs_stats[hcpcs_stats["Count"] >= 100]
worst_codes = hcpcs_stats.sort_values("Mean_Gap_Pct").head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(worst_codes.index.astype(str), worst_codes["Mean_Gap_Pct"], color=COLOR_UNDERPAID)
ax.set_title("7. 15 Most Underpaid HCPCS Codes (min 100 claims)")
ax.set_xlabel("Mean Payment Gap %")
save_chart(fig, "worst_hcpcs_codes", chart_num); chart_num += 1


# ------------------------------------------------------------
# 8. Total Dollar Gap by Provider Type (top 15, absolute)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
total_gap_type = df.groupby("Rndrng_Prvdr_Type")["Total_Dollar_Gap"].sum().sort_values().head(15)
ax.barh(total_gap_type.index, total_gap_type.values / 1_000_000, color=COLOR_UNDERPAID)
ax.set_title("8. Total Dollar Gap by Provider Type ($ Millions) -- Top 15 Most Negative")
ax.set_xlabel("Total Dollar Gap ($ Millions)")
save_chart(fig, "total_dollar_gap_by_provider_type", chart_num); chart_num += 1


# ------------------------------------------------------------
# 9. Scatter: Tot_Srvcs vs Payment_Gap_Pct
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
sample = df.sample(min(20000, len(df)), random_state=42)
ax.scatter(sample["Tot_Srvcs"].clip(upper=500), sample["Payment_Gap_Pct"].clip(-150, 150),
           alpha=0.15, s=8, color=COLOR_MAIN)
ax.set_title("9. Total Services vs Payment Gap % (sample of 20,000 rows)")
ax.set_xlabel("Total Services (clipped at 500)")
ax.set_ylabel("Payment Gap %")
save_chart(fig, "scatter_services_vs_gap", chart_num); chart_num += 1


# ------------------------------------------------------------
# 10. Avg Submitted Charge vs Avg Allowed Amount (scatter)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
sample2 = df.sample(min(20000, len(df)), random_state=42)
ax.scatter(sample2["Avg_Sbmtd_Chrg"].clip(upper=1000), sample2["Avg_Mdcr_Alowd_Amt"].clip(upper=1000),
           alpha=0.15, s=8, color=COLOR_NEUTRAL)
ax.plot([0, 1000], [0, 1000], color="red", linestyle="--", linewidth=1, label="y = x")
ax.set_title("10. Submitted Charge vs Allowed Amount")
ax.set_xlabel("Avg Submitted Charge")
ax.set_ylabel("Avg Medicare Allowed Amount")
ax.legend()
save_chart(fig, "scatter_charge_vs_allowed", chart_num); chart_num += 1


# ------------------------------------------------------------
# 11. Review Flag Rate by Provider Type (top 15)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
review_by_type = df.groupby("Rndrng_Prvdr_Type")["review_flag"].mean().sort_values(ascending=False).head(15) * 100
ax.barh(review_by_type.index, review_by_type.values, color="#D97706")
ax.set_title("11. Outlier Review-Flag Rate by Provider Type -- Top 15")
ax.set_xlabel("% of Claims Flagged for Review")
save_chart(fig, "review_flag_rate_by_type", chart_num); chart_num += 1


# ------------------------------------------------------------
# 12. Locality Count Distribution (averaging quality signal)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
locality_counts = df["Locality_Count"].value_counts().sort_index()
ax.bar(locality_counts.index.astype(str), locality_counts.values, color=COLOR_NEUTRAL)
ax.set_title("12. Distribution of Locality Count per State (Averaging Granularity)")
ax.set_xlabel("Number of Localities Averaged for State Benchmark")
ax.set_ylabel("Row Count")
save_chart(fig, "locality_count_distribution", chart_num); chart_num += 1


# ------------------------------------------------------------
# 13. Gap % by Locality Count (does averaging dilution matter?)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
gap_by_loc_count = df.groupby("Locality_Count")["Payment_Gap_Pct"].median()
ax.plot(gap_by_loc_count.index, gap_by_loc_count.values, marker="o", color=COLOR_MAIN)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("13. Median Payment Gap % vs Locality Count")
ax.set_xlabel("Number of Localities Averaged")
ax.set_ylabel("Median Payment Gap %")
save_chart(fig, "gap_vs_locality_count", chart_num); chart_num += 1


# ------------------------------------------------------------
# 14. Top 15 States by Total Dollar Underpayment
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
state_dollar = df[df["Is_Underpaid"]].groupby("Rndrng_Prvdr_State_Abrvtn")["Total_Dollar_Gap"].sum().sort_values().head(15)
ax.barh(state_dollar.index, state_dollar.values / 1_000_000, color=COLOR_UNDERPAID)
ax.set_title("14. Top 15 States by Total Underpayment ($ Millions)")
ax.set_xlabel("Total Dollar Gap ($ Millions, underpaid claims only)")
save_chart(fig, "underpayment_by_state_top15", chart_num); chart_num += 1


# ------------------------------------------------------------
# 15. Tot_Benes vs Tot_Srvcs correlation (multicollinearity check)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
sample3 = df.sample(min(20000, len(df)), random_state=42)
ax.scatter(sample3["Tot_Benes"].clip(upper=300), sample3["Tot_Srvcs"].clip(upper=300),
           alpha=0.15, s=8, color=COLOR_MAIN)
corr = df[["Tot_Benes", "Tot_Srvcs"]].corr().iloc[0, 1]
ax.set_title(f"15. Tot_Benes vs Tot_Srvcs (correlation = {corr:.3f})")
ax.set_xlabel("Total Beneficiaries (clipped at 300)")
ax.set_ylabel("Total Services (clipped at 300)")
save_chart(fig, "multicollinearity_benes_vs_srvcs", chart_num); chart_num += 1


# ------------------------------------------------------------
# 16. Underpaid Rate by Entity Type (Individual vs Organization)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
if "Rndrng_Prvdr_Ent_Cd" in df.columns:
    ent_rate = df.groupby("Rndrng_Prvdr_Ent_Cd")["Is_Underpaid"].mean() * 100
    ax.bar(ent_rate.index.astype(str), ent_rate.values, color=[COLOR_MAIN, COLOR_NEUTRAL])
    ax.set_title("16. Underpaid Rate by Entity Type")
    ax.set_xlabel("Entity Type (I=Individual, O=Organization)")
    ax.set_ylabel("% Underpaid")
else:
    ax.text(0.5, 0.5, "Rndrng_Prvdr_Ent_Cd not in cleaned dataset", ha="center")
save_chart(fig, "underpaid_by_entity_type", chart_num); chart_num += 1


# ------------------------------------------------------------
# 17. Correlation Heatmap of Key Numeric Features
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
numeric_for_corr = df[["Tot_Benes", "Tot_Srvcs", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt",
                        "Expected_Payment_NonFacility_Avg", "Payment_Gap", "Payment_Gap_Pct"]]
corr_matrix = numeric_for_corr.corr()
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr_matrix.columns, fontsize=8)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("17. Correlation Heatmap -- Key Numeric Features")
fig.colorbar(im, ax=ax, shrink=0.8)
save_chart(fig, "correlation_heatmap", chart_num); chart_num += 1


# ------------------------------------------------------------
# 18. Distribution of Total Dollar Gap (clipped)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["Total_Dollar_Gap"].clip(-10000, 10000), bins=80, color=COLOR_MAIN, alpha=0.8)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("18. Distribution of Total Dollar Gap per Claim Group (clipped +/-$10k)")
ax.set_xlabel("Total Dollar Gap ($)")
ax.set_ylabel("Count")
save_chart(fig, "total_dollar_gap_distribution", chart_num); chart_num += 1


# ------------------------------------------------------------
# 19. Underpaid Rate by Review Flag (sanity check)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
flag_rate = df.groupby("review_flag")["Is_Underpaid"].mean() * 100
ax.bar(["Not Flagged", "Flagged for Review"], flag_rate.values, color=[COLOR_OK, "#D97706"])
ax.set_title("19. Underpaid Rate -- Flagged vs Not Flagged Claims")
ax.set_ylabel("% Underpaid")
save_chart(fig, "underpaid_rate_by_review_flag", chart_num); chart_num += 1


# ------------------------------------------------------------
# 20. Top 15 Provider Types by Claim Volume (context chart)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
volume_by_type = df["Rndrng_Prvdr_Type"].value_counts().head(15)
ax.barh(volume_by_type.index, volume_by_type.values, color=COLOR_NEUTRAL)
ax.set_title("20. Top 15 Provider Types by Claim Volume")
ax.set_xlabel("Number of Claims")
save_chart(fig, "top_provider_types_by_volume", chart_num); chart_num += 1


# ============================================================
# SAVE NUMERIC SUMMARY TABLES TO ACCOMPANY CHARTS
# ============================================================
print_section("SAVING SUPPORTING SUMMARY TABLES")

by_type.to_frame("Mean_Gap_Pct").to_csv(OUTPUT_DIR / "summary_gap_by_provider_type.csv")
by_state.to_frame("Underpaid_Rate_Pct").to_csv(OUTPUT_DIR / "summary_underpaid_by_state.csv")
hcpcs_stats.to_csv(OUTPUT_DIR / "summary_hcpcs_stats.csv")
corr_matrix.to_csv(OUTPUT_DIR / "summary_correlation_matrix.csv")

print(f"Saved 4 supporting summary CSVs to: {OUTPUT_DIR}")

print_section("STEP 05 COMPLETE")
print(f"""
Generated {chart_num - 1} charts in: {CHARTS_DIR}

Key patterns to note before Step 06 (feature engineering):
- Check chart 15 (correlation): if Tot_Benes and Tot_Srvcs are highly
  correlated, only one may be needed as a model feature to avoid
  multicollinearity.
- Check chart 6 (Office vs Facility): confirms whether place of service
  should be a feature in the collection probability model.
- Check chart 3/4 (provider type gaps): these become candidate features
  for "procedure category" and "payer type" proxies needed in Step 07.
""")