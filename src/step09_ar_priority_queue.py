"""
STEP 09 - Build AR Priority Queue
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step09_ar_priority_queue.py

Purpose:
Turn the underpaid claims into an actionable AR workqueue, ranked by
estimated dollar recovery x collection confidence. This was the project
brief's missing piece flagged after Step 08: "AR priority queue: accounts
ranked by expected recovery x probability."

Methodology (ported from the parallel pipeline's build_ar_priority_queue.py,
adapted to this pipeline's column names and sign convention):
  - estimated_recovery  = abs(Total_Dollar_Gap) for underpaid rows only
                           (Total_Dollar_Gap is negative when underpaid here,
                           since Payment_Gap = Allowed - Expected < 0).
  - confidence_score    = the Step 08 model's predicted probability for
                           high_recovery_priority == 1, when the saved model
                           can be loaded and applied; otherwise a documented
                           RULE-BASED fallback score (same idea as the other
                           pipeline's confidence tiers, re-tuned to this
                           pipeline's Payment_Gap_Pct/Total_Dollar_Gap scale).
  - priority_score      = estimated_recovery * confidence_score
  - priority_tier       = Critical / High / Medium / Low, from priority_score

This keeps the same "model probability when available, rule-based fallback
otherwise" design as the parallel pipeline, but is self-contained: it loads
Step 08's saved model directly rather than requiring a separate pre-scored
file.
"""

from pathlib import Path
import warnings

import joblib
import json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "target_outputs" / "modeling_dataset.csv"

MODEL_DIR = BASE_DIR / "model_outputs"
BEST_MODEL_PATH = MODEL_DIR / "best_collection_model.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "model_feature_columns.json"

OUTPUT_DIR = BASE_DIR / "ar_priority_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
QUEUE_PATH = OUTPUT_DIR / "ar_priority_queue.csv"
TOP_QUEUE_PATH = OUTPUT_DIR / "ar_priority_queue_top_500.csv"
SUMMARY_PATH = OUTPUT_DIR / "ar_priority_summary.csv"
CONFIDENCE_METHODOLOGY_PATH = OUTPUT_DIR / "confidence_score_methodology.csv"

TOP_N = 500

# Rule-based confidence fallback, tuned to THIS pipeline's scale
# (Payment_Gap_Pct is a percentage; Total_Dollar_Gap is a total dollar
# amount across all services for that provider+HCPCS+place_of_service row,
# not a per-service gap like the other pipeline's payment_gap_per_service).
# Same design principle: bigger + clearer underpayments get higher
# confidence that an AR team's review will result in real recovery.
CONFIDENCE_RULES = [
    {
        "rule_order": 1,
        "min_abs_dollar_gap": 5000,
        "min_severity_pct": 50,
        "confidence_score": 0.95,
        "rationale": "Large total dollar gap and severe underpayment (>50%); strongest recovery signal.",
    },
    {
        "rule_order": 2,
        "min_abs_dollar_gap": 1000,
        "min_severity_pct": 25,
        "confidence_score": 0.85,
        "rationale": "Material dollar gap and clear underpayment; strong review candidate.",
    },
    {
        "rule_order": 3,
        "min_abs_dollar_gap": 100,
        "min_severity_pct": 10,
        "confidence_score": 0.70,
        "rationale": "Meaningful gap, above the Step 07 materiality threshold.",
    },
    {
        "rule_order": 4,
        "min_abs_dollar_gap": 0,
        "min_severity_pct": 0,
        "confidence_score": 0.50,
        "rationale": "Default confidence for smaller underpayment rows.",
    },
]


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA -- ONLY UNDERPAID ROWS BELONG IN AN AR QUEUE
# ============================================================
print_section("LOAD modeling_dataset.csv -- FILTER TO UNDERPAID ROWS")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

queue_df = df[df["Is_Underpaid"]].copy()
print(f"Underpaid rows (AR queue candidates): {len(queue_df):,} "
      f"({len(queue_df) / len(df) * 100:.2f}% of modeling dataset)")

if queue_df.empty:
    raise ValueError("No underpaid rows found -- check Step 07 output before building the queue.")

# estimated_recovery must be positive dollars, regardless of Total_Dollar_Gap's
# negative sign convention in this pipeline (negative = underpaid).
queue_df["estimated_recovery"] = queue_df["Total_Dollar_Gap"].abs().round(2)


# ============================================================
# TRY TO LOAD STEP 08'S TRAINED MODEL FOR REAL PROBABILITIES
# ============================================================
print_section("ATTEMPTING TO LOAD STEP 08 COLLECTION MODEL")

model = None
scaler = None
feature_cols = None
model_name = "rule_based_fallback"

try:
    if BEST_MODEL_PATH.exists() and FEATURE_COLS_PATH.exists():
        model = joblib.load(BEST_MODEL_PATH)
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = json.load(f)
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
        model_name = type(model).__name__
        print(f"Loaded Step 08 model: {model_name}")
        print(f"Expected feature columns: {len(feature_cols)}")
    else:
        print("Step 08 model artifacts not found -- using rule-based confidence fallback only.")
        print(f"  Looked for: {BEST_MODEL_PATH}")
except Exception as e:
    print(f"Could not load Step 08 model ({e}) -- using rule-based confidence fallback only.")
    model = None


# ============================================================
# RULE-BASED CONFIDENCE SCORE (ALWAYS COMPUTED, AS FALLBACK/AUDIT BASELINE)
# ============================================================
print_section("COMPUTING RULE-BASED CONFIDENCE SCORE")

abs_gap = queue_df["Total_Dollar_Gap"].abs()
severity_pct = queue_df["Payment_Gap_Pct"].abs()

conditions = [
    (abs_gap >= 5000) & (severity_pct >= 50),
    (abs_gap >= 1000) & (severity_pct >= 25),
    (abs_gap >= 100) & (severity_pct >= 10),
]
confidence_values = [0.95, 0.85, 0.70]
queue_df["rule_confidence_score"] = np.select(conditions, confidence_values, default=0.50)

print("Rule-based confidence tier distribution:")
print(queue_df["rule_confidence_score"].value_counts().sort_index(ascending=False).to_string())


# ============================================================
# MODEL-BASED PROBABILITY (WHEN AVAILABLE) -- SAME FEATURE PIPELINE AS STEP 08
# ============================================================
queue_df["collection_probability"] = np.nan

if model is not None and feature_cols is not None:
    print_section("SCORING QUEUE ROWS WITH STEP 08 MODEL")
    try:
        categorical_cols = ["procedure_category", "payer_type_proxy", "Place_Of_Srvc"]
        numeric_source_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c in feature_cols or any(c == fc for fc in feature_cols)
        ]
        # Rebuild the same one-hot feature frame Step 08 trained on, then
        # reindex to the exact saved feature_cols so column order/presence
        # always matches what the model expects (missing dummy columns ->
        # 0, extra columns dropped).
        encode_source = queue_df[
            [c for c in categorical_cols if c in queue_df.columns]
            + [c for c in df.select_dtypes(include=[np.number]).columns if c in queue_df.columns]
        ].copy()
        encoded = pd.get_dummies(
            encode_source,
            columns=[c for c in categorical_cols if c in encode_source.columns],
            drop_first=True,
        )
        encoded = encoded.reindex(columns=feature_cols, fill_value=0).fillna(0)

        X_queue = scaler.transform(encoded) if scaler is not None else encoded.values
        probabilities = model.predict_proba(X_queue)[:, 1]
        queue_df["collection_probability"] = probabilities
        print(f"Scored {len(queue_df):,} rows. "
              f"Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    except Exception as e:
        print(f"WARNING: model scoring failed ({e}). Falling back to rule-based confidence only.")
        queue_df["collection_probability"] = np.nan

queue_df["collection_model_name"] = model_name
queue_df["confidence_score"] = queue_df["collection_probability"].fillna(
    queue_df["rule_confidence_score"]
)
model_scored_rows = int(queue_df["collection_probability"].notna().sum())
print(f"\nRows scored by trained model : {model_scored_rows:,}")
print(f"Rows on rule-based fallback   : {len(queue_df) - model_scored_rows:,}")


# ============================================================
# PRIORITY SCORE + TIER + RECOMMENDED ACTION
# ============================================================
print_section("RANKING -- PRIORITY SCORE = ESTIMATED RECOVERY x CONFIDENCE")

queue_df["priority_score"] = (
    queue_df["estimated_recovery"] * queue_df["confidence_score"]
).round(2)

queue_df["priority_tier"] = np.select(
    [
        queue_df["priority_score"] >= 50000,
        queue_df["priority_score"] >= 10000,
        queue_df["priority_score"] >= 1000,
    ],
    ["Critical", "High", "Medium"],
    default="Low",
)

queue_df["recommended_action"] = np.select(
    [
        queue_df["priority_tier"].eq("Critical"),
        queue_df["priority_tier"].eq("High"),
        queue_df["priority_tier"].eq("Medium"),
    ],
    [
        "Immediate contract variance review",
        "Prioritize reimbursement audit",
        "Batch review underpayment pattern",
    ],
    default="Monitor or include in batch audit",
)

queue_df = queue_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
queue_df.insert(0, "rank", range(1, len(queue_df) + 1))

print("Priority tier distribution:")
print(queue_df["priority_tier"].value_counts().to_string())


# ============================================================
# SAVE FULL QUEUE, TOP 500, AND SUMMARY
# ============================================================
print_section("SAVING AR PRIORITY QUEUE")

output_cols = [
    "rank", "Rndrng_NPI", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn",
    "HCPCS_Cd", "Place_Of_Srvc", "Tot_Srvcs",
    "Avg_Mdcr_Alowd_Amt", "Expected_Payment_Used",
    "Payment_Gap", "Payment_Gap_Pct", "estimated_recovery",
    "rule_confidence_score", "collection_probability", "collection_model_name",
    "confidence_score", "priority_score", "priority_tier", "recommended_action",
]
output_cols = [c for c in output_cols if c in queue_df.columns]

queue_df[output_cols].to_csv(QUEUE_PATH, index=False)
print(f"Saved full queue: {QUEUE_PATH} ({len(queue_df):,} rows)")

top_queue = queue_df[output_cols].head(TOP_N)
top_queue.to_csv(TOP_QUEUE_PATH, index=False)
print(f"Saved top {TOP_N}: {TOP_QUEUE_PATH}")

pd.DataFrame(CONFIDENCE_RULES).to_csv(CONFIDENCE_METHODOLOGY_PATH, index=False)
print(f"Saved confidence methodology: {CONFIDENCE_METHODOLOGY_PATH}")

summary = pd.DataFrame([
    {"metric": "total_modeling_rows", "value": len(df)},
    {"metric": "underpaid_queue_rows", "value": len(queue_df)},
    {"metric": "model_used", "value": model_name},
    {"metric": "model_scored_rows", "value": model_scored_rows},
    {"metric": "rule_fallback_rows", "value": len(queue_df) - model_scored_rows},
    {"metric": "total_estimated_recovery", "value": round(queue_df["estimated_recovery"].sum(), 2)},
    {"metric": "total_priority_score", "value": round(queue_df["priority_score"].sum(), 2)},
    {"metric": "critical_tier_rows", "value": int((queue_df["priority_tier"] == "Critical").sum())},
    {"metric": "high_tier_rows", "value": int((queue_df["priority_tier"] == "High").sum())},
])
summary.to_csv(SUMMARY_PATH, index=False)
print(f"Saved summary: {SUMMARY_PATH}")

print_section("STEP 09 COMPLETE")
print(summary.to_string(index=False))
print("""
Next: Step 10 will add an Isolation Forest outlier detector to find
systematic underpayment patterns by state / provider type / HCPCS code
(the project brief's remaining flagged gap).
""")