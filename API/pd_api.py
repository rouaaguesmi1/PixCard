# pd_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd

# --- Load PD Model Bundle ---
try:
    pd_model_bundle = joblib.load("../Models/pd_model_calibrated.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"PD model file not found. Error: {e}")

# --- Unpack Model and Metadata ---
pd_model = pd_model_bundle["model"]
pd_features = pd_model_bundle["features"]
label_encoders = pd_model_bundle.get("label_encoders", {})  # Optional

# --- FastAPI App ---
app = FastAPI(
    title="PD Model API",
    description="Predicts the probability of default (PD) for a customer.",
    version="1.0.0"
)

# --- Input Schema ---
class PDInput(BaseModel):
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

class PDOutput(BaseModel):
    pd: float = Field(..., example=0.13, description="Probability of Default")

# --- Label Encoding Helper ---
def safe_label_transform(encoder, series):
    series = series.copy()
    known_classes = set(encoder.classes_)
    return series.apply(lambda x: encoder.transform([x])[0] if x in known_classes else -1)

# --- Endpoint ---
@app.post("/predict/pd", response_model=PDOutput)
def predict_pd(input_data: PDInput):
    try:
        row = pd.DataFrame([input_data.model_dump()])

        # Encode if needed
        for col in row.columns:
            if col in label_encoders:
                row[col] = safe_label_transform(label_encoders[col], row[col].astype(str))

        pd_input = row[pd_features]
        prob = pd_model.predict_proba(pd_input)[:, 1].item()

        return PDOutput(pd=prob)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PD prediction error: {e}")
