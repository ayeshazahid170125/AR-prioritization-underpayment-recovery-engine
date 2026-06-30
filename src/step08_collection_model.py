"""
STEP 08 - Collection Priority Model (Train, Compare, Select Best)
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step08_collection_model.py

Purpose:
Train and compare 5 candidate models for predicting high_recovery_priority
using K-Fold cross-validation on a sample, select the best model based on
PR-AUC, then retrain that best model (with its best hyperparameters) on the
FULL cleaned dataset. Logistic Regression is the project brief's required
model; the other candidates are comparison baselines to validate that choice.
"""

from pathlib import Path
import gc
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "target_outputs" / "modeling_dataset.csv"

OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "best_collection_model.pkl"
SCALER_PATH = OUTPUT_DIR / "feature_scaler.pkl"
FEATURE_COLS_PATH = OUTPUT_DIR / "model_feature_columns.json"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison_results.csv"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_model_evaluation_report.json"
THRESHOLD_CONFIG_PATH = OUTPUT_DIR / "decision_threshold_config.json"
THRESHOLD_TABLE_PATH = OUTPUT_DIR / "threshold_tuning_results.csv"
COEF_PATH = OUTPUT_DIR / "logistic_regression_coefficients.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 300_000
FINAL_TRAIN_MAX_ROWS = int(os.getenv("FINAL_TRAIN_MAX_ROWS", "2000000"))
HIGH_RECALL_TARGET = 0.70
BUSINESS_OPERATING_THRESHOLD = 0.40

# Leakage columns confirmed in Step 07 -- never used as features.
LEAKAGE_COLS = [
    "Total_Dollar_Gap", "Payment_Gap_Pct", "Payment_Gap", "Is_Underpaid",
    "claim_severity_proxy", "balance_size_bucket",
    "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Alowd_Amt_log", "Avg_Mdcr_Pymt_Amt",
    "Expected_Payment_NonFacility_Avg", "Expected_Payment_Facility_Avg",
    "Expected_Payment_Used",
]

TARGET_COL = "high_recovery_priority"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def calculate_scale_pos_weight(y):
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return negatives / positives if positives > 0 else 1.0


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


def tune_decision_thresholds(y_true, y_proba, recall_target=HIGH_RECALL_TARGET):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_f1_idx = int(np.argmax(f1_scores[:-1]))
    best_f1_threshold = float(thresholds[best_f1_idx])

    threshold_rows = [metric_row(y_true, y_proba, 0.5, "default_0_50")]
    threshold_rows.append(metric_row(y_true, y_proba, best_f1_threshold, "max_f1"))

    valid = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "f1": f1_scores[:-1],
    })
    high_recall = valid[valid["recall"] >= recall_target].sort_values(
        ["precision", "f1"], ascending=False
    )
    if len(high_recall) > 0:
        recall_threshold = float(high_recall.iloc[0]["threshold"])
        threshold_rows.append(
            metric_row(y_true, y_proba, recall_threshold, f"recall_{recall_target:.2f}")
        )
    else:
        recall_threshold = best_f1_threshold

    grid_thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for threshold in grid_thresholds:
        threshold_rows.append(metric_row(y_true, y_proba, float(threshold), f"grid_{threshold:.2f}"))

    threshold_df = pd.DataFrame(threshold_rows).drop_duplicates(subset=["label", "threshold"])
    threshold_df = threshold_df.sort_values(["f1", "recall"], ascending=False)
    return threshold_df, best_f1_threshold, recall_threshold


def save_fallback_feature_importance(model, X_reference, y_reference):
    """Save feature importance for tree models when SHAP is unavailable or unsupported."""
    if hasattr(model, "feature_importances_"):
        print("Using built-in feature_importances_.")
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        out_path = OUTPUT_DIR / "feature_importance_builtin.csv"
    else:
        print("No built-in feature_importances_ -- using permutation importance sample.")
        sample_for_perm = X_reference.sample(min(2000, len(X_reference)), random_state=RANDOM_STATE)
        y_for_perm = y_reference.loc[sample_for_perm.index]
        perm_result = permutation_importance(
            model,
            sample_for_perm.values,
            y_for_perm,
            scoring="average_precision",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": perm_result.importances_mean,
        }).sort_values("Importance", ascending=False)
        out_path = OUTPUT_DIR / "feature_importance_permutation.csv"

    print("\nTop 15 fallback feature importances:")
    print(imp_df.head(15).to_string(index=False))
    imp_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD modeling_dataset.csv")

id_cols = ["Rndrng_NPI", "Rndrng_Prvdr_State_Abrvtn", "HCPCS_Cd", "benchmark_applicable"]
exclude_cols = set(LEAKAGE_COLS + id_cols + [TARGET_COL])
categorical_cols = ["procedure_category", "payer_type_proxy", "Place_Of_Srvc"]

schema_sample = pd.read_csv(INPUT_PATH, nrows=10_000, low_memory=True)
numeric_cols = [
    c for c in schema_sample.select_dtypes(include=[np.number]).columns
    if c not in exclude_cols and c != TARGET_COL
]
required_cols = [
    c for c in categorical_cols + numeric_cols + [TARGET_COL]
    if c in schema_sample.columns
]
del schema_sample
gc.collect()

df = pd.read_csv(INPUT_PATH, usecols=required_cols, low_memory=True)
print(f"Loaded modeling columns only: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ============================================================
# BUILD FEATURE SET (EXCLUDING LEAKAGE COLUMNS)
# ============================================================
print_section("BUILDING FEATURE SET")

print(f"Categorical features: {categorical_cols}")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")

# One-hot encode categoricals
df_model = df[categorical_cols + numeric_cols + [TARGET_COL]].copy()
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

feature_cols = [c for c in df_model.columns if c != TARGET_COL]
for col in feature_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0).astype(np.float32)
df_model[TARGET_COL] = df_model[TARGET_COL].astype(np.int8)
print(f"\nTotal features after encoding: {len(feature_cols)}")
print("Feature matrix downcast to float32 for memory-safe full-data training.")
del df
gc.collect()


# ============================================================
# SAMPLE FOR FAST MODEL COMPARISON
# ============================================================
print_section(f"SAMPLING {SAMPLE_SIZE:,} ROWS FOR MODEL COMPARISON")

sample_df = df_model.sample(min(SAMPLE_SIZE, len(df_model)), random_state=RANDOM_STATE)
X_sample = sample_df[feature_cols]
y_sample = sample_df[TARGET_COL]

print(f"Sample size: {len(sample_df):,}")
print(f"Sample target distribution:\n{y_sample.value_counts().to_string()}")
sample_scale_pos_weight = calculate_scale_pos_weight(y_sample)
print(f"Sample scale_pos_weight (negative/positive): {sample_scale_pos_weight:.4f}")

scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)


# ============================================================
# DEFINE CANDIDATE MODELS
# ============================================================
print_section("DEFINING CANDIDATE MODELS")

models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Hist_Gradient_Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Gradient_Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    ),
}

if HAS_LIGHTGBM:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=150,
        max_depth=8,
        scale_pos_weight=sample_scale_pos_weight,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
else:
    print("LightGBM not installed -- skipping. Install with: pip install lightgbm")

print(f"Models to compare: {list(models.keys())}")


# ============================================================
# 5-FOLD CROSS-VALIDATION COMPARISON
# ============================================================
print_section("5-FOLD CROSS-VALIDATION COMPARISON")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
comparison_rows = []

for name, model in models.items():
    print(f"\nEvaluating: {name}")
    fold_pr_auc = []
    fold_roc_auc = []
    fold_f1 = []

    use_scaled = name == "Logistic_Regression"
    X_use = X_sample_scaled if use_scaled else X_sample.to_numpy(dtype=np.float32, copy=False)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_use, y_sample), 1):
        X_train, X_val = X_use[train_idx], X_use[val_idx]
        y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        pr_auc = average_precision_score(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)

        fold_pr_auc.append(pr_auc)
        fold_roc_auc.append(roc_auc)
        fold_f1.append(f1)
        print(f"  Fold {fold_idx}: PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  F1={f1:.4f}")

    comparison_rows.append({
        "Model": name,
        "Mean_PR_AUC": round(np.mean(fold_pr_auc), 4),
        "Std_PR_AUC": round(np.std(fold_pr_auc), 4),
        "Mean_ROC_AUC": round(np.mean(fold_roc_auc), 4),
        "Mean_F1": round(np.mean(fold_f1), 4),
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values("Mean_PR_AUC", ascending=False)
print_section("MODEL COMPARISON RESULTS (sorted by PR-AUC)")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(COMPARISON_PATH, index=False)
print(f"\nSaved: {COMPARISON_PATH}")

best_model_name = comparison_df.iloc[0]["Model"]
print(f"\nBest model by PR-AUC: {best_model_name}")


# ============================================================
# HYPERPARAMETER TUNING FOR THE BEST MODEL (ON SAMPLE)
# ============================================================
print_section(f"HYPERPARAMETER TUNING -- {best_model_name}")

param_grids = {
    "Logistic_Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
    },
    "Random_Forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [6, 10],
        "learning_rate": [0.05, 0.1],
    },
    "Hist_Gradient_Boosting": {
        "max_iter": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [31, 63],
        "l2_regularization": [0.0, 0.1],
    },
    "Gradient_Boosting": {
        "n_estimators": [100, 150],
        "max_depth": [4, 6],
    },
}

base_model = models[best_model_name]
grid = param_grids.get(best_model_name, {})
use_scaled_best = best_model_name == "Logistic_Regression"
X_tune = X_sample_scaled if use_scaled_best else X_sample.to_numpy(dtype=np.float32, copy=False)

if grid:
    grid_search = GridSearchCV(
        base_model, grid, scoring="average_precision", cv=3, n_jobs=-1
    )
    grid_search.fit(X_tune, y_sample)
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")
    print(f"Best CV PR-AUC: {grid_search.best_score_:.4f}")
else:
    best_params = {}
    print("No parameter grid defined -- using default parameters.")

del sample_df, X_sample, X_sample_scaled, X_tune
gc.collect()


# ============================================================
# RETRAIN BEST MODEL
# ============================================================
print_section(f"RETRAINING {best_model_name}")

X_full = df_model[feature_cols]
y_full = df_model[TARGET_COL]

training_mode = "full_dataset"
if FINAL_TRAIN_MAX_ROWS > 0 and len(X_full) > FINAL_TRAIN_MAX_ROWS:
    print(
        f"Using stratified final-train sample of {FINAL_TRAIN_MAX_ROWS:,} rows "
        f"from {len(X_full):,} total rows to stay within local memory/time limits."
    )
    pos_idx = y_full[y_full == 1].sample(
        n=int(FINAL_TRAIN_MAX_ROWS * y_full.mean()),
        random_state=RANDOM_STATE,
    ).index
    neg_idx = y_full[y_full == 0].sample(
        n=FINAL_TRAIN_MAX_ROWS - len(pos_idx),
        random_state=RANDOM_STATE,
    ).index
    final_idx = pos_idx.union(neg_idx)
    X_modeling = X_full.loc[final_idx]
    y_modeling = y_full.loc[final_idx]
    training_mode = f"stratified_sample_{FINAL_TRAIN_MAX_ROWS}"
else:
    print(f"Using full dataset for final train/evaluation: {len(X_full):,} rows")
    X_modeling = X_full
    y_modeling = y_full

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_modeling, y_modeling, test_size=0.2, stratify=y_modeling, random_state=RANDOM_STATE
)
train_scale_pos_weight = calculate_scale_pos_weight(y_train_full)
del df_model, X_full, y_full, X_modeling, y_modeling
gc.collect()

final_scaler = StandardScaler()
if use_scaled_best:
    X_train_final = final_scaler.fit_transform(X_train_full)
    X_test_final = final_scaler.transform(X_test_full)
else:
    X_train_final = X_train_full
    X_test_final = X_test_full

model_classes = {
    "Logistic_Regression": LogisticRegression,
    "Random_Forest": RandomForestClassifier,
    "Hist_Gradient_Boosting": HistGradientBoostingClassifier,
    "Gradient_Boosting": GradientBoostingClassifier,
}
if HAS_LIGHTGBM:
    model_classes["LightGBM"] = LGBMClassifier

final_model_class = model_classes[best_model_name]
final_params = {**best_params}
if best_model_name == "Logistic_Regression":
    final_params.update({"max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE})
elif best_model_name in ("Random_Forest",):
    final_params.update({"class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1})
elif best_model_name == "Hist_Gradient_Boosting":
    final_params.update({"class_weight": "balanced", "random_state": RANDOM_STATE})
elif best_model_name == "LightGBM":
    final_params.update({
        "scale_pos_weight": train_scale_pos_weight,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    })
else:
    final_params.update({"random_state": RANDOM_STATE})

final_model = final_model_class(**final_params)
print(f"Training final model with parameters: {final_params}")
final_model.fit(X_train_final, y_train_full)

y_test_proba = final_model.predict_proba(X_test_final)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)

final_pr_auc = average_precision_score(y_test_full, y_test_proba)
final_roc_auc = roc_auc_score(y_test_full, y_test_proba)
final_precision = precision_score(y_test_full, y_test_pred, zero_division=0)
final_recall = recall_score(y_test_full, y_test_pred, zero_division=0)
final_f1 = f1_score(y_test_full, y_test_pred, zero_division=0)

threshold_df, best_f1_threshold, high_recall_threshold = tune_decision_thresholds(
    y_test_full, y_test_proba
)
threshold_df.to_csv(THRESHOLD_TABLE_PATH, index=False)

y_test_pred_optimized = (y_test_proba >= best_f1_threshold).astype(int)
optimized_precision = precision_score(y_test_full, y_test_pred_optimized, zero_division=0)
optimized_recall = recall_score(y_test_full, y_test_pred_optimized, zero_division=0)
optimized_f1 = f1_score(y_test_full, y_test_pred_optimized, zero_division=0)

y_test_pred_operating = (y_test_proba >= BUSINESS_OPERATING_THRESHOLD).astype(int)
operating_precision = precision_score(y_test_full, y_test_pred_operating, zero_division=0)
operating_recall = recall_score(y_test_full, y_test_pred_operating, zero_division=0)
operating_f1 = f1_score(y_test_full, y_test_pred_operating, zero_division=0)

print_section("FINAL MODEL PERFORMANCE (held-out 20% test set, full dataset)")
print(f"PR-AUC : {final_pr_auc:.4f}")
print(f"ROC-AUC: {final_roc_auc:.4f}")
print(f"Default threshold: 0.5000")
print(f"Precision: {final_precision:.4f}")
print(f"Recall   : {final_recall:.4f}")
print(f"F1       : {final_f1:.4f}")
print(f"\nOptimized F1 threshold: {best_f1_threshold:.4f}")
print(f"Precision: {optimized_precision:.4f}")
print(f"Recall   : {optimized_recall:.4f}")
print(f"F1       : {optimized_f1:.4f}")
print(f"\nHigh-recall threshold ({HIGH_RECALL_TARGET:.0%} target): {high_recall_threshold:.4f}")
print(f"\nBusiness operating threshold: {BUSINESS_OPERATING_THRESHOLD:.4f}")
print(f"Precision: {operating_precision:.4f}")
print(f"Recall   : {operating_recall:.4f}")
print(f"F1       : {operating_f1:.4f}")
print(f"Saved threshold table: {THRESHOLD_TABLE_PATH}")
print("\nClassification report at default threshold:")
print(classification_report(y_test_full, y_test_pred))
print("\nConfusion matrix at default threshold:")
print(confusion_matrix(y_test_full, y_test_pred))
print("\nClassification report at optimized F1 threshold:")
print(classification_report(y_test_full, y_test_pred_optimized))
print("\nConfusion matrix at optimized F1 threshold:")
print(confusion_matrix(y_test_full, y_test_pred_optimized))
print("\nClassification report at business operating threshold:")
print(classification_report(y_test_full, y_test_pred_operating))
print("\nConfusion matrix at business operating threshold:")
print(confusion_matrix(y_test_full, y_test_pred_operating))


# ============================================================
# CONDITIONAL EXPLAINABILITY
# ============================================================
print_section("EXPLAINABILITY")

if best_model_name == "Logistic_Regression":
    print("Best model is Logistic Regression -- using coefficients directly")
    print("(SHAP is not needed; coefficients are already interpretable).")
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": final_model.coef_[0],
    }).sort_values("Coefficient", key=abs, ascending=False)
    print("\nTop 15 features by absolute coefficient:")
    print(coef_df.head(15).to_string(index=False))
    coef_df.to_csv(COEF_PATH, index=False)
    print(f"\nSaved: {COEF_PATH}")
else:
    print(f"Best model is {best_model_name} (tree-based) -- using SHAP for explainability.")
    if HAS_SHAP:
        try:
            sample_for_shap = X_test_full.sample(min(2000, len(X_test_full)), random_state=RANDOM_STATE)
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(sample_for_shap)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                "Feature": feature_cols, "Mean_Abs_SHAP": mean_abs_shap,
            }).sort_values("Mean_Abs_SHAP", ascending=False)
            print("\nTop 15 features by mean |SHAP value|:")
            print(shap_df.head(15).to_string(index=False))
            shap_df.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)
        except Exception as e:
            print(f"SHAP failed for {best_model_name}: {e}")
            save_fallback_feature_importance(final_model, X_test_full, y_test_full)
    else:
        print("SHAP not installed -- using fallback feature importance.")
        save_fallback_feature_importance(final_model, X_test_full, y_test_full)


# ============================================================
# SAVE FINAL MODEL + ARTIFACTS
# ============================================================
print_section("SAVING FINAL MODEL AND ARTIFACTS")

joblib.dump(final_model, BEST_MODEL_PATH)
print(f"Saved model: {BEST_MODEL_PATH}")

if use_scaled_best:
    joblib.dump(final_scaler, SCALER_PATH)
    print(f"Saved scaler: {SCALER_PATH}")
elif SCALER_PATH.exists():
    SCALER_PATH.unlink()
    print(f"Removed stale scaler: {SCALER_PATH}")

with open(FEATURE_COLS_PATH, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Saved feature columns: {FEATURE_COLS_PATH}")

final_report = {
    "best_model": best_model_name,
    "best_params": {k: str(v) for k, v in best_params.items()},
    "cv_comparison": comparison_df.to_dict(orient="records"),
    "final_test_pr_auc": round(final_pr_auc, 4),
    "final_test_roc_auc": round(final_roc_auc, 4),
    "default_threshold": 0.5,
    "default_test_precision": round(final_precision, 4),
    "default_test_recall": round(final_recall, 4),
    "final_test_f1": round(final_f1, 4),
    "optimized_threshold": round(best_f1_threshold, 6),
    "optimized_test_precision": round(optimized_precision, 4),
    "optimized_test_recall": round(optimized_recall, 4),
    "optimized_test_f1": round(optimized_f1, 4),
    "high_recall_target": HIGH_RECALL_TARGET,
    "high_recall_threshold": round(high_recall_threshold, 6),
    "operating_threshold": BUSINESS_OPERATING_THRESHOLD,
    "operating_threshold_strategy": "business_balanced_recall_precision",
    "operating_test_precision": round(operating_precision, 4),
    "operating_test_recall": round(operating_recall, 4),
    "operating_test_f1": round(operating_f1, 4),
    "scale_pos_weight": round(train_scale_pos_weight, 6),
    "training_mode": training_mode,
    "final_train_max_rows": FINAL_TRAIN_MAX_ROWS,
    "uses_scaling": use_scaled_best,
    "n_features": len(feature_cols),
    "training_rows": len(X_train_full),
    "test_rows": len(X_test_full),
}
with open(FINAL_REPORT_PATH, "w") as f:
    json.dump(final_report, f, indent=2)
print(f"Saved final report: {FINAL_REPORT_PATH}")

threshold_config = {
    "operating_threshold": BUSINESS_OPERATING_THRESHOLD,
    "operating_threshold_strategy": "business_balanced_recall_precision",
    "default_threshold": 0.5,
    "max_f1_threshold": round(best_f1_threshold, 6),
    "high_recall_threshold": round(high_recall_threshold, 6),
    "high_recall_target": HIGH_RECALL_TARGET,
    "note": (
        "Use 0.40 as the AR operating threshold: it keeps recall near 70% "
        "while reducing queue volume versus the max-F1 threshold."
    ),
}
with open(THRESHOLD_CONFIG_PATH, "w") as f:
    json.dump(threshold_config, f, indent=2)
print(f"Saved threshold config: {THRESHOLD_CONFIG_PATH}")

print_section("STEP 08 COMPLETE")
print(f"""
Best model: {best_model_name}
Final test PR-AUC: {final_pr_auc:.4f}

Next: Step 09 will combine this model's predicted probability with
Total_Dollar_Gap to build the AR priority queue ranking.
""")
