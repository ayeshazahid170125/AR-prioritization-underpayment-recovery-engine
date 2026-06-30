"""
STEP 14 - Supplementary Regression Validation
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step14_regression_validation.py

Purpose:
The primary expected-payment benchmark in this project is the CMS fee
schedule formula from Step 02. That is more audit-defensible than a black-box
pricing model. This step adds a lightweight ML regression validation layer
so the project also satisfies the brief's "regression" wording:

  - Target: Avg_Mdcr_Alowd_Amt (actual CMS allowed amount)
  - Models: Linear Regression baseline + HistGradientBoostingRegressor
  - Features: code/provider/state/place/service/charge context only
  - Holdout comparison: ML predictions vs the formula benchmark

Important: This is NOT the production pricing engine. It is a supplementary
cross-check that shows whether ML can learn similar payment patterns from
public aggregate data.
"""

from pathlib import Path
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "target_outputs" / "modeling_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "regression_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "best_allowed_amount_regressor.pkl"
FEATURE_CONFIG_PATH = OUTPUT_DIR / "regression_feature_config.json"
RESULTS_PATH = OUTPUT_DIR / "regression_validation_results.csv"
REPORT_PATH = OUTPUT_DIR / "regression_validation_report.json"

RANDOM_STATE = 42
SAMPLE_SIZE = int(os.getenv("REGRESSION_SAMPLE_SIZE", "500000"))
TARGET_COL = "Avg_Mdcr_Alowd_Amt"
FORMULA_COL = "Expected_Payment_Used"

CATEGORICAL_COLS = [
    "HCPCS_Cd",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
    "Place_Of_Srvc",
    "procedure_category",
    "payer_type_proxy",
]
NUMERIC_COLS = [
    "Tot_Benes",
    "Tot_Srvcs",
    "Avg_Sbmtd_Chrg",
    "Locality_Count",
    "Tot_Srvcs_log",
    "Tot_Benes_log",
    "Avg_Sbmtd_Chrg_log",
    "services_per_beneficiary",
    "provider_row_count",
    "provider_total_services",
    "hcpcs_row_count",
    "hcpcs_total_services",
    "state_row_count",
    "state_total_services",
    "hcpcs_state_row_share",
    "avg_charge_per_beneficiary",
    "provider_service_share",
    "hcpcs_service_share_in_state",
]


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def regression_metrics(y_true, y_pred):
    mean_actual = float(np.mean(y_true))
    median_actual = float(np.median(y_true))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "Median_AE": round(float(np.median(np.abs(y_true - y_pred))), 4),
        "MAE_Pct_Mean_Actual": round(float(mae / mean_actual * 100), 2) if mean_actual else None,
        "RMSE_Pct_Mean_Actual": round(float(rmse / mean_actual * 100), 2) if mean_actual else None,
        "Mean_Actual": round(mean_actual, 4),
        "Median_Actual": round(median_actual, 4),
    }


def add_target_encoding(train_df, test_df, col, target_col, global_mean, smoothing=25):
    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
        stats["count"] + smoothing
    )
    new_col = f"{col}_target_enc"
    train_df[new_col] = train_df[col].map(smooth).fillna(global_mean).astype(np.float32)
    test_df[new_col] = test_df[col].map(smooth).fillna(global_mean).astype(np.float32)
    return new_col, smooth.to_dict()


print_section("LOAD MODELING DATA FOR REGRESSION VALIDATION")

available_cols = set(pd.read_csv(INPUT_PATH, nrows=0).columns)
usecols = [
    c for c in CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL, FORMULA_COL]
    if c in available_cols
]
df = pd.read_csv(INPUT_PATH, usecols=usecols, low_memory=True)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET_COL, FORMULA_COL])
df = df[(df[TARGET_COL] >= 0) & (df[FORMULA_COL] >= 0)].copy()

if SAMPLE_SIZE > 0 and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

print(f"Loaded regression sample: {len(df):,} rows x {df.shape[1]} columns")
print(f"Target: {TARGET_COL}")
print(f"Formula benchmark column: {FORMULA_COL}")


print_section("TRAIN / TEST SPLIT")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
global_mean = float(train_df[TARGET_COL].mean())
print(f"Train rows: {len(train_df):,}")
print(f"Test rows : {len(test_df):,}")
print(f"Global mean allowed amount: ${global_mean:,.2f}")


print_section("TARGET ENCODING HIGH-CARDINALITY CONTEXT")

encoding_maps = {}
encoded_cols = []
for col in CATEGORICAL_COLS:
    if col in train_df.columns:
        new_col, mapping = add_target_encoding(train_df, test_df, col, TARGET_COL, global_mean)
        encoded_cols.append(new_col)
        encoding_maps[col] = mapping
        print(f"  encoded {col} -> {new_col} ({len(mapping):,} categories)")

numeric_cols = [c for c in NUMERIC_COLS if c in train_df.columns]
feature_cols = numeric_cols + encoded_cols

for col in feature_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0).astype(np.float32)
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0).astype(np.float32)

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL].astype(np.float32)
X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL].astype(np.float32)

print(f"Final regression features: {len(feature_cols)}")


print_section("TRAIN REGRESSION MODELS")

models = {
    "CMS_Formula_Benchmark": None,
    "Linear_Regression": LinearRegression(),
    "HistGradientBoosting_Regressor": HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    ),
}

result_rows = []

formula_pred = test_df[FORMULA_COL].astype(np.float32)
formula_metrics = regression_metrics(y_test, formula_pred)
result_rows.append({"Model": "CMS_Formula_Benchmark", **formula_metrics})
print(f"CMS formula benchmark: {formula_metrics}")

best_model_name = None
best_model = None
best_r2 = -np.inf

for name, model in models.items():
    if model is None:
        continue
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    result_rows.append({"Model": name, **metrics})
    print(f"{name}: {metrics}")
    if metrics["R2"] > best_r2:
        best_r2 = metrics["R2"]
        best_model_name = name
        best_model = model

results_df = pd.DataFrame(result_rows).sort_values("R2", ascending=False)
results_df.to_csv(RESULTS_PATH, index=False)


print_section("SAVE REGRESSION VALIDATION ARTIFACTS")

joblib.dump(best_model, MODEL_PATH)

feature_config = {
    "target": TARGET_COL,
    "formula_benchmark": FORMULA_COL,
    "feature_cols": feature_cols,
    "categorical_target_encoded": CATEGORICAL_COLS,
    "numeric_cols": numeric_cols,
    "sample_size": int(len(df)),
    "global_target_mean": round(global_mean, 6),
    "note": (
        "Regression is supplementary validation only. The CMS formula remains "
        "the primary expected-payment benchmark."
    ),
}
with open(FEATURE_CONFIG_PATH, "w") as f:
    json.dump(feature_config, f, indent=2)

report = {
    "best_ml_regressor": best_model_name,
    "best_ml_r2": best_r2,
    "results": results_df.to_dict(orient="records"),
    "primary_benchmark_positioning": (
        "CMS formula benchmark is primary; ML regression is a validation cross-check."
    ),
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"Saved results: {RESULTS_PATH}")
print(f"Saved report : {REPORT_PATH}")
print(f"Saved model  : {MODEL_PATH}")
print("\nResults:")
print(results_df.to_string(index=False))

print_section("STEP 14 COMPLETE")
