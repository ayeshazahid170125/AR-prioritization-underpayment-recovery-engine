"""
STEP 12 - FastAPI Inference Endpoint
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run:
    uvicorn app.step12_fastapi:app --reload --port 8001

Endpoints:
    GET  /health              -- API status check
    POST /predict/recovery    -- Single claim recovery priority prediction
    POST /predict/batch       -- Batch prediction (list of claims)
    GET  /model/info          -- Model metadata
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
MODEL_DIR = BASE_DIR / "model_outputs"

BEST_MODEL_PATH  = MODEL_DIR / "best_collection_model.pkl"
SCALER_PATH      = MODEL_DIR / "feature_scaler.pkl"
FEATURE_COLS_PATH= MODEL_DIR / "model_feature_columns.json"
FINAL_REPORT_PATH= MODEL_DIR / "final_model_evaluation_report.json"

# ============================================================
# LOAD MODEL ARTIFACTS AT STARTUP
# ============================================================
print("Loading model artifacts...")

try:
    model = joblib.load(BEST_MODEL_PATH)
    print(f"Model loaded: {type(model).__name__}")
except Exception as e:
    raise RuntimeError(f"Cannot load model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded.")
except Exception:
    scaler = None
    print("No scaler found — raw features will be used.")

try:
    with open(FEATURE_COLS_PATH) as f:
        FEATURE_COLS = json.load(f)
    print(f"Feature columns loaded: {len(FEATURE_COLS)} features")
except Exception as e:
    raise RuntimeError(f"Cannot load feature columns: {e}")

try:
    with open(FINAL_REPORT_PATH) as f:
        MODEL_REPORT = json.load(f)
except Exception:
    MODEL_REPORT = {}

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="WellMind AR Recovery Engine",
    description="AR Prioritization & Underpayment Recovery — Project 2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SCHEMAS
# ============================================================
class ClaimInput(BaseModel):
    procedure_category: str = Field(
        default="Evaluation_and_Management",
        description="CPT code category",
        example="Evaluation_and_Management"
    )
    payer_type_proxy: str = Field(
        default="Medicare_Participating",
        description="Medicare participation status",
        example="Medicare_Participating"
    )
    place_of_service: str = Field(
        default="O",
        description="O=Office, F=Facility",
        example="O"
    )
    tot_srvcs: float = Field(default=50.0, description="Total services", ge=0)
    tot_benes: float = Field(default=30.0, description="Total beneficiaries", ge=0)
    avg_sbmtd_chrg: float = Field(default=250.0, description="Avg submitted charge ($)", ge=0)
    tot_srvcs_log: Optional[float] = Field(default=None)
    tot_benes_log: Optional[float] = Field(default=None)
    avg_sbmtd_chrg_log: Optional[float] = Field(default=None)
    services_per_beneficiary: Optional[float] = Field(default=None)
    review_flag: Optional[int] = Field(default=0)
    duplicate_key_flag: Optional[int] = Field(default=0)
    zero_expected_reason_flag: Optional[int] = Field(default=0)
    locality_count: Optional[float] = Field(default=1.0)
    ruca_unknown_flag: Optional[int] = Field(default=0)


class PredictionResponse(BaseModel):
    recovery_probability: float
    priority_tier: str
    recommended_action: str
    estimated_recovery_signal: str
    confidence_score: float
    model_name: str
    top_risk_factors: List[str]


class BatchInput(BaseModel):
    claims: List[ClaimInput]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_claims: int
    high_priority_count: int
    critical_count: int

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def build_feature_row(claim: ClaimInput) -> pd.DataFrame:
    """Convert ClaimInput to the exact feature DataFrame the model expects."""

    # Compute log features if not provided
    tot_srvcs_log      = claim.tot_srvcs_log      or float(np.log1p(max(claim.tot_srvcs, 0)))
    tot_benes_log      = claim.tot_benes_log       or float(np.log1p(max(claim.tot_benes, 0)))
    avg_sbmtd_chrg_log = claim.avg_sbmtd_chrg_log  or float(np.log1p(max(claim.avg_sbmtd_chrg, 0)))
    svc_per_bene       = (claim.services_per_beneficiary
                          if claim.services_per_beneficiary is not None
                          else (claim.tot_srvcs / max(claim.tot_benes, 1)))

    raw = {
        "procedure_category": claim.procedure_category,
        "payer_type_proxy":   claim.payer_type_proxy,
        "Place_Of_Srvc":      claim.place_of_service,
        "Tot_Srvcs":          claim.tot_srvcs,
        "Tot_Benes":          claim.tot_benes,
        "Avg_Sbmtd_Chrg":     claim.avg_sbmtd_chrg,
        "Tot_Srvcs_log":      tot_srvcs_log,
        "Tot_Benes_log":      tot_benes_log,
        "Avg_Sbmtd_Chrg_log": avg_sbmtd_chrg_log,
        "services_per_beneficiary": svc_per_bene,
        "review_flag":             claim.review_flag or 0,
        "duplicate_key_flag":      claim.duplicate_key_flag or 0,
        "Locality_Count":          claim.locality_count or 1.0,
        "RUCA_Unknown_Flag":       claim.ruca_unknown_flag or 0,
    }

    df = pd.DataFrame([raw])

    # One-hot encode categoricals (same as step08)
    cat_cols = ["procedure_category", "payer_type_proxy", "Place_Of_Srvc"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to saved feature columns
    df = df.reindex(columns=FEATURE_COLS, fill_value=0).fillna(0)
    return df


def get_priority_tier(prob: float) -> str:
    if prob >= 0.80:   return "Critical"
    if prob >= 0.60:   return "High"
    if prob >= 0.40:   return "Medium"
    return "Low"


def get_recommended_action(tier: str) -> str:
    return {
        "Critical": "Immediate contract variance review — escalate to billing supervisor",
        "High":     "Prioritize reimbursement audit this week",
        "Medium":   "Include in next batch underpayment review",
        "Low":      "Monitor — low recovery probability",
    }.get(tier, "Review manually")


def get_recovery_signal(prob: float) -> str:
    if prob >= 0.80:   return "Very High — strong underpayment signal"
    if prob >= 0.60:   return "High — material underpayment likely"
    if prob >= 0.40:   return "Moderate — worth reviewing"
    return "Low — minor or no underpayment signal"


def get_top_risk_factors(df_row: pd.DataFrame, n: int = 3) -> List[str]:
    """Return top N feature names with non-zero values as risk factors."""
    nonzero = [(col, float(df_row[col].iloc[0]))
               for col in df_row.columns
               if float(df_row[col].iloc[0]) != 0]
    nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
    factors = []
    label_map = {
        "Tot_Srvcs_log":           "High service volume",
        "Tot_Benes_log":           "High beneficiary count",
        "Avg_Sbmtd_Chrg_log":      "High submitted charge",
        "services_per_beneficiary":"High services per patient",
        "review_flag":             "Flagged for review",
        "duplicate_key_flag":      "Duplicate key flag active",
    }
    for col, val in nonzero[:n]:
        label = label_map.get(col, col.replace("_", " ").title())
        factors.append(label)
    return factors if factors else ["No strong risk factors detected"]


def predict_single(claim: ClaimInput) -> PredictionResponse:
    df_row = build_feature_row(claim)
    X = scaler.transform(df_row) if scaler else df_row.values
    prob = float(model.predict_proba(X)[0][1])
    tier = get_priority_tier(prob)

    return PredictionResponse(
        recovery_probability=round(prob, 4),
        priority_tier=tier,
        recommended_action=get_recommended_action(tier),
        estimated_recovery_signal=get_recovery_signal(prob),
        confidence_score=round(prob, 4),
        model_name=type(model).__name__,
        top_risk_factors=get_top_risk_factors(df_row),
    )

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": type(model).__name__,
        "features": len(FEATURE_COLS),
        "project": "Project 2 - AR Underpayment Recovery Engine",
    }


@app.get("/model/info")
def model_info():
    return {
        "model_name": type(model).__name__,
        "n_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "evaluation": MODEL_REPORT,
        "target": "high_recovery_priority",
        "target_definition": "Claims where Total_Dollar_Gap < -$50 AND Payment_Gap_Pct < -10%",
        "disclaimer": (
            "This model uses CMS Medicare PUF data with a synthetic proxy target. "
            "No real AR collection outcomes are used. Portfolio demonstration only."
        ),
    }


@app.post("/predict/recovery", response_model=PredictionResponse)
def predict_recovery(claim: ClaimInput):
    try:
        return predict_single(claim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    if len(batch.claims) > 500:
        raise HTTPException(status_code=400, detail="Max 500 claims per batch.")
    try:
        results = [predict_single(c) for c in batch.claims]
        high_priority = sum(1 for r in results if r.priority_tier in ["Critical", "High"])
        critical = sum(1 for r in results if r.priority_tier == "Critical")
        return BatchResponse(
            predictions=results,
            total_claims=len(results),
            high_priority_count=high_priority,
            critical_count=critical,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("step12_fastapi:app", host="0.0.0.0", port=8001, reload=True)
