import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Configuration: Expanded SME Archetypes ---
ARCHETYPES = {
    "Stable Manufacturer": {
        "weight": 0.35, "industry_code": 31, "market_position": "Leader",
        "industry_default_rate_params": (0.03, 0.01),
        "base_params": {"business_age": (15, 5), "employee_count": (70, 25), "owner_experience": (20, 8)},
        "health_score_params": (0.85, 0.08),
        "financial_multipliers": {"revenue_per_employee": 15000, "expense_ratio": 0.75, "cash_on_hand_months": 3.0},
        "collateral_type": "Real Estate / Equipment", "pandemic_impact_prob": 0.2
    },
    "High-Growth Tech Startup": {
        "weight": 0.15, "industry_code": 51, "market_position": "Mid",
        "industry_default_rate_params": (0.10, 0.04),
        "base_params": {"business_age": (3, 1.5), "employee_count": (25, 15), "owner_experience": (5, 3)},
        "health_score_params": (0.6, 0.2),
        "financial_multipliers": {"revenue_per_employee": 25000, "expense_ratio": 1.25, "cash_on_hand_months": 6.0},
        "collateral_type": "Intellectual Property / Cash", "pandemic_impact_prob": 0.1
    },
    "Struggling Main Street Retailer": {
        "weight": 0.20, "industry_code": 44, "market_position": "Niche",
        "industry_default_rate_params": (0.08, 0.03),
        "base_params": {"business_age": (8, 4), "employee_count": (10, 5), "owner_experience": (10, 5)},
        "health_score_params": (0.25, 0.1),
        "financial_multipliers": {"revenue_per_employee": 8000, "expense_ratio": 0.95, "cash_on_hand_months": 0.75},
        "collateral_type": "Inventory", "pandemic_impact_prob": 0.85
    },
    "Niche Service Consultant": {
        "weight": 0.30, "industry_code": 54, "market_position": "Niche",
        "industry_default_rate_params": (0.02, 0.01),
        "base_params": {"business_age": (7, 4), "employee_count": (4, 2), "owner_experience": (12, 6)},
        "health_score_params": (0.9, 0.05),
        "financial_multipliers": {"revenue_per_employee": 50000, "expense_ratio": 0.40, "cash_on_hand_months": 4.0},
        "collateral_type": "None", "pandemic_impact_prob": 0.3
    }
}

def generate_full_synthetic_data(n_samples=100_000):
    """
    Generates a comprehensive and realistic synthetic dataset for SME credit risk.
    """
    print(f"Generating {n_samples} SME records...")
    df = pd.DataFrame(index=range(n_samples))

    # --- Step 1: Assign Archetypes and Static Categorical Features ---
    archetype_names = list(ARCHETYPES.keys())
    archetype_weights = [ARCHETYPES[name]['weight'] for name in archetype_names]
    df['archetype'] = np.random.choice(archetype_names, size=n_samples, p=archetype_weights)

    df['industry_code'] = df['archetype'].map({name: p['industry_code'] for name, p in ARCHETYPES.items()})
    df['market_position_score'] = df['archetype'].map({name: p['market_position'] for name, p in ARCHETYPES.items()})

    # --- Step 2 & 3: Generate Base Features & Latent Health Score ---
    df['health_score'] = 0.0
    for name, params in tqdm(ARCHETYPES.items(), desc="Generating Archetype Data"):
        mask = df['archetype'] == name
        count = mask.sum()
        df.loc[mask, 'business_age_years'] = np.random.normal(params['base_params']['business_age'][0], params['base_params']['business_age'][1], count)
        df.loc[mask, 'employee_count'] = np.random.normal(params['base_params']['employee_count'][0], params['base_params']['employee_count'][1], count)
        df.loc[mask, 'owner_experience_years'] = np.random.normal(params['base_params']['owner_experience'][0], params['base_params']['owner_experience'][1], count)
        df.loc[mask, 'health_score'] = np.random.normal(params['health_score_params'][0], params['health_score_params'][1], count)
        df.loc[mask, 'industry_default_rate'] = np.random.normal(params['industry_default_rate_params'][0], params['industry_default_rate_params'][1], count)

    # Clean up and add correlated features
    df['business_age_years'] = df['business_age_years'].clip(1, 50)
    df['employee_count'] = df['employee_count'].clip(1).round()
    df['owner_experience_years'] = df['owner_experience_years'].clip(0, 60)
    df['health_score'] = df['health_score'].clip(0.01, 0.99)
    df['industry_default_rate'] = df['industry_default_rate'].clip(0.005, 0.25)
    df['owner_age'] = (df['owner_experience_years'] + np.random.uniform(22, 35, n_samples)).clip(20, 80).round()
    df['prev_loans_count'] = (df['business_age_years'] * np.random.uniform(0.5, 1.5, n_samples)).round().clip(0, 50)

    # --- Step 4: Generate Correlated Financial & Behavioral Features ---
    # Revenue & Expenses
    rev_per_emp = df['archetype'].map({name: p['financial_multipliers']['revenue_per_employee'] for name, p in ARCHETYPES.items()})
    df['monthly_revenue'] = df['employee_count'] * rev_per_emp * np.random.normal(1, 0.15, n_samples)
    expense_ratios = df['archetype'].map({name: p['financial_multipliers']['expense_ratio'] for name, p in ARCHETYPES.items()})
    df['monthly_expenses'] = df['monthly_revenue'] * expense_ratios * np.random.normal(1, 0.05, n_samples)

    # Cash & Bank Behavior
    cash_multipliers = df['archetype'].map({name: p['financial_multipliers']['cash_on_hand_months'] for name, p in ARCHETYPES.items()})
    df['cash_on_hand'] = df['monthly_revenue'] * cash_multipliers * df['health_score']
    df['avg_balance_last_3mo'] = df['cash_on_hand'] * np.random.normal(1.2, 0.3, n_samples)
    df['overdraft_count_3mo'] = np.random.poisson(lam = ((1 - df['health_score'])**2) * 5)
    df['return_tx_rate'] = ((1 - df['health_score']) * 0.05 * np.random.uniform(0.5, 1.5, n_samples)).clip(0, 0.1)

    # Invoices, Receivables & Payables
    df['days_sales_outstanding'] = 30 + (1 - df['health_score']) * np.random.uniform(30, 90, n_samples)
    df['days_payables_outstanding'] = df['days_sales_outstanding'] + (df['health_score'] - 0.7) * 30
    df['receivables_total'] = (df['days_sales_outstanding'] / 30) * df['monthly_revenue']
    df['payables_total'] = (df['days_payables_outstanding'] / 30) * df['monthly_expenses']

    # Credit History & Owner Data
    df['on_time_payment_ratio'] = (df['health_score']**1.5).clip(0.4, 0.99) * np.random.normal(1, 0.02, n_samples)
    df['founder_has_default_history'] = (np.random.rand(n_samples) > df['health_score']**0.5).astype(int)
    df['owner_credit_score'] = 300 + (df['health_score'] * 550) + np.random.normal(0, 25, n_samples)

    # Loan, Collateral & LGD
    df['credit_limit'] = (df['monthly_revenue'] * 3 * df['health_score']**2).clip(5000, 1_000_000)
    df['loan_utilization_rate'] = ((1 - df['health_score']) * np.random.uniform(0.3, 1, n_samples)).clip(0.05, 1.0)
    df['current_loan_balance'] = df['credit_limit'] * df['loan_utilization_rate']
    df['collateral_type'] = df['archetype'].map({name: p['collateral_type'] for name, p in ARCHETYPES.items()})
    lgd_map = {"Real Estate / Equipment": 0.3, "Intellectual Property / Cash": 0.6, "Inventory": 0.8, "None": 0.95}
    base_lgd = df['collateral_type'].map(lgd_map)
    df['loss_given_default'] = np.random.normal(base_lgd, 0.1, n_samples).clip(0.05, 1.0)

    # --- Step 5: Calculate Derived Ratios (Now including the missing ones) ---
    epsilon = 1e-6
    df['net_profit_margin'] = (df['monthly_revenue'] - df['monthly_expenses']) / (df['monthly_revenue'] + epsilon)
    df['expense_to_revenue_ratio'] = df['monthly_expenses'] / (df['monthly_revenue'] + epsilon)
    df['cash_burn_rate'] = (df['monthly_expenses'] - df['monthly_revenue']).clip(0) # Only positive if expenses > revenue
    df['revenue_trend_3mo'] = (df['health_score'] - 0.5) * np.random.uniform(0.05, 0.2, n_samples)
    df.loc[df['archetype'] == 'High-Growth Tech Startup', 'revenue_trend_3mo'] *= 5

    # More complex ratios
    df['current_assets'] = df['cash_on_hand'] + df['receivables_total']
    df['current_liabilities'] = df['payables_total'] + (df['current_loan_balance'] * np.random.uniform(0.1, 0.4, n_samples))
    df['current_ratio'] = df['current_assets'] / (df['current_liabilities'] + epsilon)
    df['quick_ratio'] = (df['cash_on_hand'] + df['receivables_total']) / (df['current_liabilities'] + epsilon) # As requested
    df['debt_to_equity_ratio'] = (1 / (df['health_score'] + epsilon)) * np.random.normal(1, 0.2, n_samples) - 1
    df['interest_coverage_ratio'] = (df['net_profit_margin'] * 10 * df['health_score']).clip(-2, 50)
    
    # --- Step 6: Add External Factors and Simulation Flags ---
    df['local_economic_indicator'] = np.random.normal(1.0, 0.2, n_samples) + (df['health_score']-0.5)*0.1
    pandemic_probs = df['archetype'].map({name: p['pandemic_impact_prob'] for name, p in ARCHETYPES.items()})
    df['pandemic_impact_flag'] = (np.random.rand(n_samples) < pandemic_probs).astype(int)
    # If impacted by pandemic, reduce health score and give some a subsidy
    impact_mask = df['pandemic_impact_flag'] == 1
    df.loc[impact_mask, 'health_score'] *= np.random.uniform(0.5, 0.9, impact_mask.sum())
    df['government_subsidy_received'] = ((impact_mask) & (np.random.rand(n_samples) < 0.4)).astype(int)
    df['fraud_alert_flag'] = (np.random.rand(n_samples) < 0.005).astype(int)

    # --- Step 7: Generate Final Target Variable (Default Flag) ---
    logit_score = -2.5 + (-5 * df['health_score']) + (2 * (1 - df['on_time_payment_ratio'])) \
                  + (1.5 * df['founder_has_default_history']) + (1 * df['debt_to_equity_ratio'].clip(-1, 5)) \
                  - (0.5 * df['pandemic_impact_flag']) + (0.3 * df['government_subsidy_received']) \
                  + np.random.normal(0, 0.5, n_samples)
    prob_default = 1 / (1 + np.exp(-logit_score))
    df['default_in_12_mo'] = (np.random.rand(n_samples) < prob_default).astype(int)
    df['num_defaults_last_12mo'] = (df['default_in_12_mo'] * np.random.randint(1, 4, n_samples)).astype(int)

    # --- Final Cleanup ---
    df = df.drop(columns=['health_score', 'current_assets', 'current_liabilities'])
    # Reorder columns to match your original list as closely as possible
    final_ordered_columns = [
        # Key IDs & Targets
        'archetype', 'default_in_12_mo', 'loss_given_default', 'credit_limit',
        # Income Statement
        'monthly_revenue', 'monthly_expenses', 'net_profit_margin', 'interest_coverage_ratio',
        'revenue_trend_3mo', 'profit_trend_3mo', # profit_trend is similar to revenue_trend here
        # Balance Sheet & Cash Flow
        'cash_on_hand', 'receivables_total', 'payables_total', 'current_loan_balance',
        'debt_to_equity_ratio', 'current_ratio', 'quick_ratio', 'cash_burn_rate',
        # Transactional / Behavioral
        'loan_utilization_rate', 'days_sales_outstanding', 'days_payables_outstanding',
        'avg_balance_last_3mo', 'overdraft_count_3mo', 'return_tx_rate',
        # Firmographic / KYB
        'business_age_years', 'employee_count', 'industry_code', 'market_position_score',
        # Owner / KYC
        'owner_credit_score', 'owner_age', 'owner_experience_years', 'founder_has_default_history',
        # Credit History
        'on_time_payment_ratio', 'prev_loans_count', 'num_defaults_last_12mo',
        # External & Simulation Flags
        'industry_default_rate', 'local_economic_indicator', 'pandemic_impact_flag',
        'government_subsidy_received', 'fraud_alert_flag'
    ]
    # Add any missing columns with placeholder data to ensure they exist
    for col in final_ordered_columns:
        if col not in df.columns:
            df[col] = np.nan # Or generate plausible data
    
    df = df.reindex(columns=final_ordered_columns).round(4)
    df['profit_trend_3mo'] = df['revenue_trend_3mo'] * np.random.normal(1, 0.3, n_samples) # Add related profit trend
    
    print(f"\nDataset generation complete. Shape: {df.shape}")
    print(f"Default Rate: {df['default_in_12_mo'].mean():.2%}")
    return df

# --- Generate and Inspect ---
if __name__ == '__main__':
    sme_data = generate_full_synthetic_data(n_samples=100_000)
    print("\n--- Dataset Head ---")
    print(sme_data.head())
    print("\n--- Info on a few key new columns ---")
    print(sme_data[['quick_ratio', 'cash_burn_rate', 'market_position_score', 'pandemic_impact_flag']].describe(include='all'))
    sme_data.to_csv('sme_synthetic_credit_data_FULL.csv', index=False)
    print("\nFull dataset saved to 'sme_synthetic_credit_data_FULL.csv'")