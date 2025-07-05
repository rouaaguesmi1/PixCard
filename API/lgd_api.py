from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd

# --- Load LGD Model Bundle ---
try:
    lgd_model_bundle = joblib.load("../Models/lgd_model_full.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"LGD model file not found. Error: {e}")

lgd_model = lgd_model_bundle["model"]
lgd_features = lgd_model_bundle["features"]
label_encoders = lgd_model_bundle.get("label_encoders", {})  # optional

app = FastAPI(
    title="LGD Model API",
    description="Predicts Loss Given Default (LGD) for customers who have defaulted.",
    version="1.0.0"
)

class LGDInput(BaseModel):
    # Same features used in lgd_features
    # Fill exactly as in your lgd_features with correct types:
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
    default_in_12_mo: int = Field(..., description="Must be 1 to compute LGD")

class LGDOutput(BaseModel):
    lgd: float = Field(..., example=0.45, description="Predicted Loss Given Default")

def safe_label_transform(encoder, series):
    series = series.copy()
    known_classes = set(encoder.classes_)
    return series.apply(lambda x: encoder.transform([x])[0] if x in known_classes else -1)

@app.post("/predict/lgd", response_model=LGDOutput)
def predict_lgd(input_data: LGDInput):
    try:
        if input_data.default_in_12_mo != 1:
            raise HTTPException(status_code=400, detail="LGD prediction only valid if default_in_12_mo=1")

        row = pd.DataFrame([input_data.model_dump()])

        for col in row.columns:
            if col in label_encoders:
                row[col] = safe_label_transform(label_encoders[col], row[col].astype(str))

        lgd_input = row[lgd_features]
        lgd_pred = lgd_model.predict(lgd_input).item()

        return LGDOutput(lgd=lgd_pred)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LGD prediction error: {e}")
