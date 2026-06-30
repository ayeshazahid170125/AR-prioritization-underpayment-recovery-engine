"""
STEP 10 - Isolation Forest Underpayment Pattern Detection
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step10_isolation_forest_anomalies.py

Purpose:
The project brief explicitly asks for an "Isolation Forest outlier detector
for systematic underpayment patterns by payer." Step 04b's five statistical
methods (IQR, Z-score, Percentile, Modified Z-score, Domain rule) flag
individual ROW-level outliers. This step is different and complementary:
it groups claims into (state, provider type, HCPCS code, place of service)
PATTERNS first, then uses Isolation Forest to find which whole PATTERNS
look anomalous -- i.e. systematic underpayment behavior concentrated in a
specific corner of the data, not just one extreme row.

Methodology ported from the parallel pipeline's detect_underpayment_
anomalies.py, adapted to this pipeline's column names and the
Is_Underpaid / Total_Dollar_Gap / Payment_Gap_Pct fields already built in
Steps 03-04.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "target_outputs" / "modeling_dataset.csv"

OUTPUT_DIR = BASE_DIR / "anomaly_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
PATTERNS_PATH = OUTPUT_DIR / "underpayment_anomaly_patterns.csv"
SUMMARY_PATH = OUTPUT_DIR / "underpayment_anomaly_summary.csv"

GROUP_COLUMNS = ["Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc"]
MIN_GROUP_ROWS = 20
# Flag roughly the top 10% most unusual grouped underpayment patterns.
# This is more reviewer-friendly than sklearn's "auto" threshold because the
# review volume is explicit and reproducible.
CONTAMINATION = 0.10
NUMERIC_FEATURES = [
    "row_count",
    "underpaid_rate",
    "total_estimated_recovery",
    "avg_gap_pct",
    "avg_recovery_per_row",
    "avg_total_services",
]


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD modeling_dataset.csv")
required_cols = GROUP_COLUMNS + [
    "Is_Underpaid", "Total_Dollar_Gap", "Payment_Gap_Pct", "Tot_Srvcs",
]
df = pd.read_csv(INPUT_PATH, usecols=required_cols, low_memory=True)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

for col in GROUP_COLUMNS:
    if col not in df.columns:
        raise KeyError(f"Required grouping column not found: {col}")


# ============================================================
# BUILD GROUP-LEVEL PATTERNS
# ============================================================
print_section("BUILDING GROUP-LEVEL UNDERPAYMENT PATTERNS")
print(f"Grouping by: {GROUP_COLUMNS}")
print(f"Minimum rows per group to be scored: {MIN_GROUP_ROWS}")

grouped = (
    df.groupby(GROUP_COLUMNS, dropna=False)
    .agg(
        row_count=("Is_Underpaid", "size"),
        underpaid_rows=("Is_Underpaid", "sum"),
        total_estimated_recovery=("Total_Dollar_Gap", lambda s: s[s < 0].abs().sum()),
        gap_pct_sum=("Payment_Gap_Pct", lambda s: s[s < 0].abs().sum()),
        avg_total_services=("Tot_Srvcs", "mean"),
    )
    .reset_index()
)

patterns = grouped[grouped["row_count"] >= MIN_GROUP_ROWS].copy()
print(f"Total groups formed: {len(grouped):,}")
print(f"Groups with >= {MIN_GROUP_ROWS} rows (scored): {len(patterns):,}")

if len(patterns) < 20:
    raise ValueError(
        "Not enough grouped patterns for Isolation Forest "
        f"(found {len(patterns)}, need at least 20). Check MIN_GROUP_ROWS or input data."
    )

patterns["underpaid_rate"] = patterns["underpaid_rows"] / patterns["row_count"]
patterns["avg_gap_pct"] = np.where(
    patterns["underpaid_rows"] > 0,
    patterns["gap_pct_sum"] / patterns["underpaid_rows"],
    0.0,
)
patterns["avg_recovery_per_row"] = patterns["total_estimated_recovery"] / patterns["row_count"]
patterns = patterns.drop(columns=["gap_pct_sum"])

print("\nSample patterns:")
print(patterns.head(10).to_string(index=False))


# ============================================================
# FIT ISOLATION FOREST ON GROUP-LEVEL FEATURES
# ============================================================
print_section("FITTING ISOLATION FOREST")

features = patterns[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
scaled = StandardScaler().fit_transform(features)

model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1,
)
labels = model.fit_predict(scaled)
patterns["anomaly_flag"] = labels == -1
patterns["anomaly_score"] = (-model.decision_function(scaled)).round(6)

patterns = patterns.sort_values(
    ["anomaly_flag", "anomaly_score", "total_estimated_recovery"],
    ascending=[False, False, False],
)

anomaly_count = int(patterns["anomaly_flag"].sum())
print(f"Patterns flagged as anomalous: {anomaly_count:,} ({anomaly_count / len(patterns) * 100:.2f}%)")

print("\nTop 15 anomalous patterns (by anomaly score, then $ recovery):")
print(
    patterns[patterns["anomaly_flag"]]
    .head(15)[GROUP_COLUMNS + ["row_count", "underpaid_rate", "total_estimated_recovery", "anomaly_score"]]
    .to_string(index=False)
)


# ============================================================
# WHICH PROVIDER TYPES / STATES DOMINATE THE ANOMALOUS PATTERNS
# ============================================================
print_section("WHERE THE ANOMALIES CONCENTRATE")

anomalous = patterns[patterns["anomaly_flag"]]
if not anomalous.empty:
    print("By provider type:")
    print(anomalous["Rndrng_Prvdr_Type"].value_counts().head(10).to_string())
    print("\nBy state:")
    print(anomalous["Rndrng_Prvdr_State_Abrvtn"].value_counts().head(10).to_string())


# ============================================================
# SAVE OUTPUTS
# ============================================================
print_section("SAVING OUTPUTS")

patterns.to_csv(PATTERNS_PATH, index=False)
print(f"Saved: {PATTERNS_PATH}")

summary = pd.DataFrame([
    {"metric": "input_rows", "value": len(df)},
    {"metric": "total_groups_formed", "value": len(grouped)},
    {"metric": "groups_scored_min_rows", "value": MIN_GROUP_ROWS},
    {"metric": "groups_scored", "value": len(patterns)},
    {"metric": "anomaly_patterns", "value": anomaly_count},
    {"metric": "anomaly_rate_pct", "value": round(anomaly_count / len(patterns) * 100, 4)},
    {"metric": "contamination_setting", "value": str(CONTAMINATION)},
    {"metric": "grouping_columns", "value": ", ".join(GROUP_COLUMNS)},
])
summary.to_csv(SUMMARY_PATH, index=False)
print(f"Saved: {SUMMARY_PATH}")

print_section("STEP 10 COMPLETE")
print(summary.to_string(index=False))
print("""
Next: Step 11 will build the business underpayment report. Step 14 provides
the supplementary regression validation layer for the project brief's
"regression" requirement while keeping the CMS fee schedule formula as the
primary benchmark.
""")
