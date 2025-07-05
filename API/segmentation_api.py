from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd

# --- Load Segmentation Pipeline ---
try:
    segmentation_bundle = joblib.load("../Models/segmentation_pipeline.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Segmentation pipeline file not found. Error: {e}")

label_encoders = segmentation_bundle["label_encoders"]
scaler = segmentation_bundle["scaler"]
pca = segmentation_bundle["pca"]
kmeans = segmentation_bundle["kmeans"]
seg_features = segmentation_bundle["features"]

app = FastAPI(
    title="Customer Segmentation API",
    description="Assigns risk cluster to customers using PCA and KMeans.",
    version="1.0.0"
)

class SegmentInput(BaseModel):
    monthly_revenue: float
    monthly_expenses: float
    net_profit_margin: float
    interest_coverage_ratio: float
    revenue_trend_3mo: float
    profit_trend_3mo: float
    cash_on_hand: float
    receivables_total: float
    payables_total: float
    current_loan_balance: float
    debt_to_equity_ratio: float
    current_ratio: float
    quick_ratio: float
    cash_burn_rate: float
    loan_utilization_rate: float
    days_sales_outstanding: int
    days_payables_outstanding: int
    avg_balance_last_3mo: float
    overdraft_count_3mo: int
    return_tx_rate: float
    business_age_years: int
    employee_count: int
    industry_code: int
    market_position_score: float
    owner_credit_score: int
    owner_age: int
    owner_experience_years: int
    founder_has_default_history: int
    on_time_payment_ratio: float
    prev_loans_count: int
    industry_default_rate: float
    local_economic_indicator: float
    pandemic_impact_flag: int
    government_subsidy_received: int

class SegmentOutput(BaseModel):
    risk_cluster: int = Field(..., example=2, description="Assigned risk cluster")
    pca_components: List[float] = Field(..., example=[1.2, -0.5, 0.1], description="PCA components")

def safe_label_transform(encoder, series):
    series = series.copy()
    known_classes = set(encoder.classes_)
    return series.apply(lambda x: encoder.transform([x])[0] if x in known_classes else -1)

@app.post("/predict/segment", response_model=SegmentOutput)
def predict_segment(input_data: SegmentInput):
    try:
        row = pd.DataFrame([input_data.model_dump()])

        for col, encoder in label_encoders.items():
            row[col] = safe_label_transform(encoder, row[col].astype(str))

        scaled = scaler.transform(row[seg_features])
        pca_result = pca.transform(scaled)
        cluster = kmeans.predict(scaled).item()

        return SegmentOutput(
            risk_cluster=cluster,
            pca_components=pca_result[0].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation prediction error: {e}")
