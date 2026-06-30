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
from fastapi import FastAPI, HTTPException, Query
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
THRESHOLD_CONFIG_PATH = MODEL_DIR / "decision_threshold_config.json"

# ============================================================
# LOAD MODEL ARTIFACTS AT STARTUP
# ============================================================
print("Loading model artifacts...")

try:
    model = joblib.load(BEST_MODEL_PATH)
    print(f"Model loaded: {type(model).__name__}")
except Exception as e:
    model = None
    print(f"Model unavailable: {e}")

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
    FEATURE_COLS = []
    print(f"Feature columns unavailable: {e}")

try:
    with open(FINAL_REPORT_PATH) as f:
        MODEL_REPORT = json.load(f)
except Exception:
    MODEL_REPORT = {}

try:
    with open(THRESHOLD_CONFIG_PATH) as f:
        THRESHOLD_CONFIG = json.load(f)
except Exception:
    THRESHOLD_CONFIG = {}

OPERATING_THRESHOLD = float(
    THRESHOLD_CONFIG.get("operating_threshold", MODEL_REPORT.get("optimized_threshold", 0.5))
)

# ============================================================
# DASHBOARD DATA SOURCES
# ============================================================
REPORT_DIR = BASE_DIR / "report_outputs"
QUEUE_PATH = REPORT_DIR / "top_underpayments.csv"
SUMMARY_PATHS = {
    "hcpcs": REPORT_DIR / "underpayment_summary_by_hcpcs.csv",
    "state": REPORT_DIR / "underpayment_summary_by_state.csv",
    "provider": REPORT_DIR / "underpayment_summary_by_provider_type.csv",
    "payer": REPORT_DIR / "underpayment_summary_by_payer_type.csv",
}
EXEC_SUMMARY_PATH = REPORT_DIR / "underpayment_report_summary.csv"

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
    provider_row_count: Optional[float] = Field(default=0.0, ge=0)
    provider_total_services: Optional[float] = Field(default=0.0, ge=0)
    hcpcs_row_count: Optional[float] = Field(default=0.0, ge=0)
    hcpcs_total_services: Optional[float] = Field(default=0.0, ge=0)
    state_row_count: Optional[float] = Field(default=0.0, ge=0)
    state_total_services: Optional[float] = Field(default=0.0, ge=0)
    hcpcs_state_row_share: Optional[float] = Field(default=0.0, ge=0)
    avg_charge_per_beneficiary: Optional[float] = Field(default=None, ge=0)
    provider_service_share: Optional[float] = Field(default=0.0, ge=0)
    hcpcs_service_share_in_state: Optional[float] = Field(default=0.0, ge=0)


class PredictionResponse(BaseModel):
    recovery_probability: float
    operating_threshold: float
    predicted_high_priority: bool
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
        "provider_row_count":      claim.provider_row_count or 0.0,
        "provider_total_services": claim.provider_total_services or 0.0,
        "hcpcs_row_count":         claim.hcpcs_row_count or 0.0,
        "hcpcs_total_services":    claim.hcpcs_total_services or 0.0,
        "state_row_count":         claim.state_row_count or 0.0,
        "state_total_services":    claim.state_total_services or 0.0,
        "hcpcs_state_row_share":   claim.hcpcs_state_row_share or 0.0,
        "avg_charge_per_beneficiary": (
            claim.avg_charge_per_beneficiary
            if claim.avg_charge_per_beneficiary is not None
            else claim.avg_sbmtd_chrg * claim.tot_srvcs / max(claim.tot_benes, 1)
        ),
        "provider_service_share": claim.provider_service_share or 0.0,
        "hcpcs_service_share_in_state": claim.hcpcs_service_share_in_state or 0.0,
    }

    for col in [
        "provider_row_count", "provider_total_services", "hcpcs_row_count",
        "hcpcs_total_services", "state_row_count", "state_total_services",
        "avg_charge_per_beneficiary",
    ]:
        raw[f"{col}_log"] = float(np.log1p(max(raw[col], 0)))

    df = pd.DataFrame([raw])

    # One-hot encode categoricals (same as step08)
    cat_cols = ["procedure_category", "payer_type_proxy", "Place_Of_Srvc"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to saved feature columns
    df = df.reindex(columns=FEATURE_COLS, fill_value=0).fillna(0)
    return df


def get_priority_tier(prob: float) -> str:
    if prob >= 0.80:   return "Critical"
    if prob >= max(0.60, OPERATING_THRESHOLD): return "High"
    if prob >= min(0.40, OPERATING_THRESHOLD): return "Medium"
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
    if model is None or not FEATURE_COLS:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Rebuild or restore model_outputs/*.pkl before prediction.",
        )
    df_row = build_feature_row(claim)
    X = scaler.transform(df_row) if scaler else df_row.values
    prob = float(model.predict_proba(X)[0][1])
    predicted_high_priority = prob >= OPERATING_THRESHOLD
    tier = get_priority_tier(prob)

    return PredictionResponse(
        recovery_probability=round(prob, 4),
        operating_threshold=round(OPERATING_THRESHOLD, 4),
        predicted_high_priority=predicted_high_priority,
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
        "model": type(model).__name__ if model is not None else None,
        "model_available": model is not None and bool(FEATURE_COLS),
        "features": len(FEATURE_COLS),
        "operating_threshold": OPERATING_THRESHOLD,
        "project": "Project 2 - AR Underpayment Recovery Engine",
    }


@app.get("/model/info")
def model_info():
    return {
        "model_name": type(model).__name__ if model is not None else None,
        "model_available": model is not None and bool(FEATURE_COLS),
        "n_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "evaluation": MODEL_REPORT,
        "threshold_config": THRESHOLD_CONFIG or {"operating_threshold": OPERATING_THRESHOLD},
        "target": "high_recovery_priority",
        "target_definition": "Claims where Total_Dollar_Gap < -$50 AND Payment_Gap_Pct < -10%",
        "disclaimer": (
            "This model uses CMS Medicare PUF data with a synthetic proxy target. "
            "No real AR collection outcomes are used. Portfolio demonstration only."
        ),
    }


def read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing data file: {path.name}")
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


@app.get("/dashboard/summary")
def dashboard_summary():
    rows = read_csv_records(EXEC_SUMMARY_PATH)
    return {str(row["metric"]): row["value"] for row in rows if "metric" in row and "value" in row}


@app.get("/dashboard/queue")
def dashboard_queue(
    limit: int = Query(default=500, ge=1, le=500),
    provider_type: Optional[str] = None,
    state: Optional[str] = None,
    tier: Optional[str] = None,
    payer_type: Optional[str] = None,
    hcpcs: Optional[str] = None,
    min_recovery: float = Query(default=0.0, ge=0),
):
    if not QUEUE_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Missing data file: {QUEUE_PATH.name}")

    df = pd.read_csv(QUEUE_PATH)
    filters = {
        "Rndrng_Prvdr_Type": provider_type,
        "Rndrng_Prvdr_State_Abrvtn": state,
        "priority_tier": tier,
        "payer_type_proxy": payer_type,
    }
    for column, value in filters.items():
        if value and column in df.columns:
            df = df[df[column].astype(str) == value]
    if hcpcs and "HCPCS_Cd" in df.columns:
        df = df[df["HCPCS_Cd"].astype(str).str.contains(hcpcs, case=False, na=False)]
    if "estimated_recovery" in df.columns:
        df = df[df["estimated_recovery"] >= min_recovery]

    records = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return {
        "rows": records,
        "total_rows": int(len(df)),
        "total_estimated_recovery": float(df["estimated_recovery"].sum()) if "estimated_recovery" in df.columns else 0.0,
        "critical_high_count": int(df["priority_tier"].isin(["Critical", "High"]).sum()) if "priority_tier" in df.columns else 0,
        "avg_confidence": float(df["confidence_score"].mean()) if "confidence_score" in df.columns and len(df) else None,
    }


@app.get("/dashboard/report/{report_name}")
def dashboard_report(report_name: str, limit: int = Query(default=25, ge=1, le=200)):
    path = SUMMARY_PATHS.get(report_name)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown report. Use hcpcs, state, provider, or payer.")
    rows = read_csv_records(path)
    return {"rows": rows[:limit], "total_rows": len(rows)}


@app.get("/dashboard/filter-options")
def dashboard_filter_options():
    if not QUEUE_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Missing data file: {QUEUE_PATH.name}")
    df = pd.read_csv(QUEUE_PATH)

    def options(column: str) -> list[str]:
        if column not in df.columns:
            return []
        return sorted(str(value) for value in df[column].dropna().unique().tolist())

    return {
        "provider_types": options("Rndrng_Prvdr_Type"),
        "states": options("Rndrng_Prvdr_State_Abrvtn"),
        "tiers": [tier for tier in ["Critical", "High", "Medium", "Low"] if tier in options("priority_tier")],
        "payer_types": options("payer_type_proxy"),
        "max_recovery": float(df["estimated_recovery"].max()) if "estimated_recovery" in df.columns and len(df) else 0.0,
    }


@app.post("/predict/recovery", response_model=PredictionResponse)
def predict_recovery(claim: ClaimInput):
    try:
        return predict_single(claim)
    except HTTPException:
        raise
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("step12_fastapi:app", host="0.0.0.0", port=8001, reload=True)
