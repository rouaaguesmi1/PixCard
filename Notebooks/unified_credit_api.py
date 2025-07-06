# unified_credit_api_pytorch.py

import pandas as pd
import numpy as np
import joblib
import os
import random
import time

# --- ML/DL Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# --- PyTorch Imports for the Autoencoder ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional

# ===================================================================
# Step 1: Data Loading and Preprocessing
# ===================================================================

def load_and_preprocess_data(file_path="sme_synthetic_credit_data_FULL.csv"):
    """Loads and preprocesses the raw data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please run generatefinal.py first.")
    
    df = pd.read_csv(file_path)
    
    # Label Encode categorical features if any exist
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df

# ===================================================================
# Step 2: PyTorch Autoencoder Definition
# ===================================================================

# Define the Autoencoder network structure using PyTorch's nn.Module
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        # Decoder layer
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # Use Sigmoid for outputs between 0 and 1 (assuming scaled input)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ===================================================================
# Step 3: Model Training and Saving
# ===================================================================

def train_all_models(df):
    """Orchestrator function to train and save all necessary models."""
    print("--- Starting Model Training ---")

    # Define features and targets
    pd_target = 'default_in_12_mo'
    archetype_target = 'archetype'
    
    leakage_cols = [pd_target, 'loss_given_default', 'credit_limit', 'collateral_value', 
                    'collateral_type', archetype_target, 'fraud_alert_flag', 'num_defaults_last_12mo']
    features = [col for col in df.columns if col not in leakage_cols]

    X = df[features]
    y_pd = df[pd_target]
    y_archetype = df[archetype_target]

    X_train, X_test, y_pd_train, y_pd_test, y_archetype_train, y_archetype_test = train_test_split(
        X, y_pd, y_archetype, test_size=0.2, random_state=42, stratify=y_pd
    )

    # --- Train and Save PD Model ---
    print("Training PD Model (XGBoost)...")
    pd_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    pd_model.fit(X_train, y_pd_train)
    joblib.dump({"model": pd_model, "features": features}, "pd_model.pkl")
    print("PD Model saved as pd_model.pkl")

    # --- Train and Save QuickScore Model ---
    print("Training QuickScore Model (Logistic Regression)...")
    quickscore_features = ['owner_credit_score', 'on_time_payment_ratio', 'debt_to_equity_ratio', 'business_age_years']
    quickscore_model = LogisticRegression(random_state=42)
    quickscore_model.fit(X_train[quickscore_features], y_pd_train)
    joblib.dump({"model": quickscore_model, "features": quickscore_features}, "quickscore_model.pkl")
    print("QuickScore Model saved as quickscore_model.pkl")

    # --- Train and Save Archetype Classifier ---
    print("Training Archetype Classifier (XGBoost)...")
    archetype_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax', random_state=42)
    archetype_model.fit(X_train, y_archetype_train)
    joblib.dump({"model": archetype_model, "features": features}, "archetype_classifier.pkl")
    print("Archetype Classifier saved as archetype_classifier.pkl")
    
    # --- Train and Save Anomaly Detector Autoencoders (PyTorch) ---
    print("Training Anomaly Detectors (PyTorch Autoencoders)...")
    autoencoders = {}
    scalers = {}
    archetype_ids = df['archetype'].unique()
    
    for archetype_id in archetype_ids:
        print(f"  - Training autoencoder for archetype {archetype_id}...")
        archetype_df = df[df['archetype'] == archetype_id][features]
        
        # Scale data to be between 0 and 1 for the Sigmoid activation in the autoencoder
        scaler = StandardScaler() 
        # Fit on training data only to prevent data leakage from test set
        scaler.fit(X_train[X_train.index.isin(archetype_df.index)])
        scaled_data = scaler.transform(archetype_df)
        
        # Convert to PyTorch Tensors
        dataset = TensorDataset(torch.tensor(scaled_data, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Initialize model, loss function, and optimizer
        input_dim = scaled_data.shape[1]
        encoding_dim = int(input_dim / 2)
        model = Autoencoder(input_dim, encoding_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            for data in dataloader:
                inputs, = data
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Save the trained model's state dictionary and the scaler
        torch.save(model.state_dict(), f"autoencoder_archetype_{archetype_id}.pth")
        scalers[archetype_id] = scaler
        
    joblib.dump({"scalers": scalers, "features": features}, "anomaly_artifacts.pkl")
    print("Anomaly Detector artifacts (scalers) saved.")
    print("--- Model Training Complete ---")

# ===================================================================
# Step 4: API Setup and Model Loading
# ===================================================================

# Check if models exist, if not, train them
model_files = ["pd_model.pkl", "quickscore_model.pkl", "archetype_classifier.pkl", "anomaly_artifacts.pkl"]
if not all(os.path.exists(p) for p in model_files):
    print("One or more models not found. Training from scratch...")
    raw_df = load_and_preprocess_data()
    train_all_models(raw_df)

# Load all the models and their metadata
print("Loading all trained models for the API...")
pd_bundle = joblib.load("pd_model.pkl")
pd_model, pd_features = pd_bundle["model"], pd_bundle["features"]

quickscore_bundle = joblib.load("quickscore_model.pkl")
quickscore_model, quickscore_features = quickscore_bundle["model"], quickscore_bundle["features"]

archetype_bundle = joblib.load("archetype_classifier.pkl")
archetype_model, archetype_features = archetype_bundle["model"], archetype_bundle["features"]

anomaly_artifacts = joblib.load("anomaly_artifacts.pkl")
scalers, anomaly_features = anomaly_artifacts["scalers"], anomaly_artifacts["features"]

# Load the PyTorch autoencoder models
autoencoders = {}
for archetype_id in scalers.keys():
    input_dim = len(anomaly_features)
    encoding_dim = int(input_dim / 2)
    model = Autoencoder(input_dim, encoding_dim)
    model.load_state_dict(torch.load(f"autoencoder_archetype_{archetype_id}.pth"))
    model.eval() # Set model to evaluation mode
    autoencoders[archetype_id] = model

# --- Initialize FastAPI App ---
app = FastAPI(
    title="PixCARD Unified API ",
    description="A self-contained API that trains and serves credit risk models, including PyTorch autoencoders.",
    version="3.1.0"
)

# ===================================================================
# Step 5: API Schemas and Endpoints
# ===================================================================

# Define Pydantic Schemas
class SMEInput(BaseModel):
    # ... (Copy the full SMEInput schema here) ...
    monthly_revenue: float; monthly_expenses: float; net_profit_margin: float; interest_coverage_ratio: float;
    revenue_trend_3mo: float; profit_trend_3mo: float; cash_on_hand: float; receivables_total: float;
    payables_total: float; current_loan_balance: float; debt_to_equity_ratio: float; current_ratio: float;
    quick_ratio: float; cash_burn_rate: float; loan_utilization_rate: float; days_sales_outstanding: int;
    days_payables_outstanding: int; avg_balance_last_3mo: float; overdraft_count_3mo: int; return_tx_rate: float;
    business_age_years: int; employee_count: int; industry_code: int; market_position_score: int;
    owner_credit_score: int; owner_age: int; owner_experience_years: int; founder_has_default_history: int;
    on_time_payment_ratio: float; prev_loans_count: int; industry_default_rate: float;
    local_economic_indicator: float; pandemic_impact_flag: int; government_subsidy_received: int

class QuickScoreInput(BaseModel):
    owner_credit_score: int; on_time_payment_ratio: float; debt_to_equity_ratio: float; business_age_years: int

# --- API Endpoints ---
# The logic now uses the loaded, trained models.

@app.post("/screening/quickscore", tags=["1. Quick Screening"])
def quickscore(data: QuickScoreInput):
    """Provides a fast risk indicator using the trained Logistic Regression model."""
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prob = quickscore_model.predict_proba(input_df[quickscore_features])[:, 1].item()
        
        if prob < 0.05: return {"risk_indicator": "Low Risk"}
        elif prob < 0.20: return {"risk_indicator": "Medium Risk"}
        else: return {"risk_indicator": "High Risk"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decisioning/predict-pd", tags=["2. Credit Decisioning"])
def predict_pd(data: SMEInput):
    """Predicts Probability of Default using the main XGBoost model."""
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prob = pd_model.predict_proba(input_df[pd_features])[:, 1].item()
        return {"probability_of_default": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/classify-archetype", tags=["3. Customer Analytics"])
def classify_archetype(data: SMEInput):
    """Classifies an SME into a business archetype using a multiclass XGBoost model."""
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction_idx = archetype_model.predict(input_df[archetype_features])[0]
        # This map should ideally be saved and loaded, but hardcoded for this example.
        archetype_map = {3: "Stable Manufacturer", 1: "High-Growth Tech Startup", 2: "Struggling Main Street Retailer", 0: "Niche Service Consultant"}
        return {"predicted_archetype": archetype_map.get(int(prediction_idx), "Unknown")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/detect-anomaly", tags=["3. Customer Analytics"])
def detect_anomaly(data: SMEInput):
    """Detects anomalies using a trained PyTorch Autoencoder for the SME's predicted archetype."""
    try:
        input_df = pd.DataFrame([data.model_dump()])
        
        # Step 1: Classify archetype
        archetype_idx = int(archetype_model.predict(input_df[archetype_features])[0])
        
        # Step 2: Select correct autoencoder and scaler
        if archetype_idx not in autoencoders:
            return {"is_anomaly": True, "reason": f"No anomaly detector for archetype {archetype_idx}"}
            
        autoencoder = autoencoders[archetype_idx]
        scaler = scalers[archetype_idx]
        
        # Step 3: Scale data and convert to PyTorch tensor
        scaled_input_np = scaler.transform(input_df[anomaly_features])
        input_tensor = torch.tensor(scaled_input_np, dtype=torch.float32)
        
        # Step 4: Get reconstruction and calculate error
        with torch.no_grad(): # Disable gradient calculation for inference
            reconstructed_tensor = autoencoder(input_tensor)
        
        loss_fn = nn.MSELoss()
        reconstruction_error = loss_fn(reconstructed_tensor, input_tensor).item()
        
        # Step 5: Compare error to a predefined threshold (determined experimentally)
        threshold = 0.1
        
        return {
            "is_anomaly": reconstruction_error > threshold,
            "reconstruction_error": reconstruction_error,
            "threshold": threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    # The --reload flag automatically restarts the server when you save changes to the file.
    uvicorn.run("unified_credit_api:app", host="0.0.0.0", port=8000, reload=True)