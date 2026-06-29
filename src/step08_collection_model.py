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
import json
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
    roc_auc_score, classification_report, confusion_matrix,
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
COEF_PATH = OUTPUT_DIR / "logistic_regression_coefficients.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 300_000

# Leakage columns confirmed in Step 07 -- never used as features.
LEAKAGE_COLS = [
    "Total_Dollar_Gap", "Payment_Gap_Pct", "Payment_Gap", "Is_Underpaid",
    "claim_severity_proxy", "balance_size_bucket",
    "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Alowd_Amt_log",
    "Expected_Payment_NonFacility_Avg", "Expected_Payment_Facility_Avg",
    "Expected_Payment_Used",
]

TARGET_COL = "high_recovery_priority"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ============================================================
# BUILD FEATURE SET (EXCLUDING LEAKAGE COLUMNS)
# ============================================================
print_section("BUILDING FEATURE SET")

id_cols = ["Rndrng_NPI", "Rndrng_Prvdr_State_Abrvtn", "HCPCS_Cd", "benchmark_applicable"]
exclude_cols = set(LEAKAGE_COLS + id_cols + [TARGET_COL])

categorical_cols = ["procedure_category", "payer_type_proxy", "Place_Of_Srvc"]
numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in exclude_cols and c != TARGET_COL
]

print(f"Categorical features: {categorical_cols}")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")

# One-hot encode categoricals
df_model = df[categorical_cols + numeric_cols + [TARGET_COL]].copy()
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

feature_cols = [c for c in df_model.columns if c != TARGET_COL]
print(f"\nTotal features after encoding: {len(feature_cols)}")


# ============================================================
# SAMPLE FOR FAST MODEL COMPARISON
# ============================================================
print_section(f"SAMPLING {SAMPLE_SIZE:,} ROWS FOR MODEL COMPARISON")

sample_df = df_model.sample(min(SAMPLE_SIZE, len(df_model)), random_state=RANDOM_STATE)
X_sample = sample_df[feature_cols].fillna(0)
y_sample = sample_df[TARGET_COL]

print(f"Sample size: {len(sample_df):,}")
print(f"Sample target distribution:\n{y_sample.value_counts().to_string()}")

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
        n_estimators=150, max_depth=8, random_state=RANDOM_STATE, verbose=-1
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
    X_use = X_sample_scaled if use_scaled else X_sample.values

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
X_tune = X_sample_scaled if use_scaled_best else X_sample.values

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


# ============================================================
# RETRAIN BEST MODEL ON FULL DATASET
# ============================================================
print_section(f"RETRAINING {best_model_name} ON FULL DATASET ({len(df_model):,} rows)")

X_full = df_model[feature_cols].fillna(0)
y_full = df_model[TARGET_COL]

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
)

final_scaler = StandardScaler()
if use_scaled_best:
    X_train_final = final_scaler.fit_transform(X_train_full)
    X_test_final = final_scaler.transform(X_test_full)
else:
    X_train_final = X_train_full.values
    X_test_final = X_test_full.values

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
    final_params.update({"random_state": RANDOM_STATE, "verbose": -1})
else:
    final_params.update({"random_state": RANDOM_STATE})

final_model = final_model_class(**final_params)
print(f"Training final model with parameters: {final_params}")
final_model.fit(X_train_final, y_train_full)

y_test_proba = final_model.predict_proba(X_test_final)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)

final_pr_auc = average_precision_score(y_test_full, y_test_proba)
final_roc_auc = roc_auc_score(y_test_full, y_test_proba)
final_f1 = f1_score(y_test_full, y_test_pred)

print_section("FINAL MODEL PERFORMANCE (held-out 20% test set, full dataset)")
print(f"PR-AUC : {final_pr_auc:.4f}")
print(f"ROC-AUC: {final_roc_auc:.4f}")
print(f"F1     : {final_f1:.4f}")
print("\nClassification report:")
print(classification_report(y_test_full, y_test_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test_full, y_test_pred))


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

with open(FEATURE_COLS_PATH, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Saved feature columns: {FEATURE_COLS_PATH}")

final_report = {
    "best_model": best_model_name,
    "best_params": {k: str(v) for k, v in best_params.items()},
    "cv_comparison": comparison_df.to_dict(orient="records"),
    "final_test_pr_auc": round(final_pr_auc, 4),
    "final_test_roc_auc": round(final_roc_auc, 4),
    "final_test_f1": round(final_f1, 4),
    "uses_scaling": use_scaled_best,
    "n_features": len(feature_cols),
    "training_rows": len(X_train_full),
    "test_rows": len(X_test_full),
}
with open(FINAL_REPORT_PATH, "w") as f:
    json.dump(final_report, f, indent=2)
print(f"Saved final report: {FINAL_REPORT_PATH}")

print_section("STEP 08 COMPLETE")
print(f"""
Best model: {best_model_name}
Final test PR-AUC: {final_pr_auc:.4f}

Next: Step 09 will combine this model's predicted probability with
Total_Dollar_Gap to build the AR priority queue ranking.
""")
