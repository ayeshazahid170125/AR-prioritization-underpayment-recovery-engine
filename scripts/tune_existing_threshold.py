"""
Tune the decision threshold for the currently saved Step 08 model.

This is the fast path when you want better recall/F1 without retraining.
It rebuilds the same deterministic 80/20 split used in Step 08, scores the
held-out rows with the saved model, and writes:
  - model_outputs/threshold_tuning_results.csv
  - model_outputs/decision_threshold_config.json

Run:
    python scripts/tune_existing_threshold.py
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "target_outputs" / "modeling_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "model_outputs"
MODEL_PATH = MODEL_DIR / "best_collection_model.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "model_feature_columns.json"
THRESHOLD_CONFIG_PATH = MODEL_DIR / "decision_threshold_config.json"
THRESHOLD_TABLE_PATH = MODEL_DIR / "threshold_tuning_results.csv"

TARGET_COL = "high_recovery_priority"
RANDOM_STATE = 42
HIGH_RECALL_TARGET = 0.70
BUSINESS_OPERATING_THRESHOLD = 0.40


def metric_row(y_true, y_proba, threshold, label):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "label": label,
        "threshold": round(float(threshold), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "predicted_positive_rate": round(float(y_pred.mean()), 4),
    }


def main():
    print("Loading saved model artifacts...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    print("Loading modeling dataset...")
    available_cols = set(pd.read_csv(INPUT_PATH, nrows=0).columns)

    categorical_cols = ["procedure_category", "payer_type_proxy", "Place_Of_Srvc"]
    source_cols = [c for c in categorical_cols if c in available_cols]
    source_cols += [c for c in feature_cols if c in available_cols and c != TARGET_COL]
    source_cols = sorted(set(source_cols))
    required_cols = sorted(set(source_cols + [TARGET_COL]))
    df = pd.read_csv(INPUT_PATH, usecols=required_cols, low_memory=True)

    encoded = pd.get_dummies(
        df[source_cols + [TARGET_COL]],
        columns=[c for c in categorical_cols if c in source_cols],
        drop_first=True,
    )
    encoded = encoded.reindex(columns=feature_cols + [TARGET_COL], fill_value=0).fillna(0)

    X = encoded[feature_cols]
    y = encoded[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Scoring held-out rows: {len(X_test):,}")
    X_model = scaler.transform(X_test) if scaler is not None else X_test.values
    proba = model.predict_proba(X_model)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])

    valid = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "f1": f1_scores[:-1],
    })
    high_recall = valid[valid["recall"] >= HIGH_RECALL_TARGET].sort_values(
        ["precision", "f1"], ascending=False
    )
    high_recall_threshold = (
        float(high_recall.iloc[0]["threshold"]) if len(high_recall) else best_threshold
    )

    rows = [
        metric_row(y_test, proba, 0.5, "default_0_50"),
        metric_row(y_test, proba, best_threshold, "max_f1"),
        metric_row(y_test, proba, BUSINESS_OPERATING_THRESHOLD, "business_operating_0_40"),
        metric_row(y_test, proba, high_recall_threshold, f"recall_{HIGH_RECALL_TARGET:.2f}"),
    ]
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        rows.append(metric_row(y_test, proba, threshold, f"grid_{threshold:.2f}"))

    table = pd.DataFrame(rows).drop_duplicates(subset=["label", "threshold"])
    table = table.sort_values(["f1", "recall"], ascending=False)
    table.to_csv(THRESHOLD_TABLE_PATH, index=False)

    config = {
        "operating_threshold": BUSINESS_OPERATING_THRESHOLD,
        "operating_threshold_strategy": "business_balanced_recall_precision",
        "default_threshold": 0.5,
        "max_f1_threshold": round(best_threshold, 6),
        "high_recall_threshold": round(high_recall_threshold, 6),
        "high_recall_target": HIGH_RECALL_TARGET,
        "model_name": type(model).__name__,
        "heldout_rows": int(len(y_test)),
        "note": (
            "0.40 is selected for AR operations because it keeps recall near "
            "70% with materially less queue volume than the max-F1 threshold."
        ),
    }
    with open(THRESHOLD_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print("\nTop threshold rows:")
    print(table.head(8).to_string(index=False))
    print(f"\nSaved: {THRESHOLD_TABLE_PATH}")
    print(f"Saved: {THRESHOLD_CONFIG_PATH}")


if __name__ == "__main__":
    main()
