import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

# --- Configuration: Define SME Archetypes ---
# This is the core of the logic. We define the parameters for our business personas.
ARCHETYPES = {
    "Stable Manufacturer": {
        "weight": 0.35,
        "base_params": {
            "business_age": (15, 5),  # (mean, std)
            "employee_count": (70, 25),
            "owner_experience": (20, 8)
        },
        "health_score_params": (0.85, 0.08), # High and stable health
        "financial_multipliers": {
            "revenue_per_employee": 15000,
            "expense_ratio": 0.75,
            "cash_on_hand_months": 3.0
        },
        "collateral_type": "Real Estate / Equipment"
    },
    "High-Growth Tech Startup": {
        "weight": 0.15,
        "base_params": {
            "business_age": (3, 1.5),
            "employee_count": (25, 15),
            "owner_experience": (5, 3)
        },
        "health_score_params": (0.6, 0.2), # Medium health, high volatility
        "financial_multipliers": {
            "revenue_per_employee": 25000,
            "expense_ratio": 1.25, # Cash burning
            "cash_on_hand_months": 6.0 # Higher cash reserves from funding
        },
        "collateral_type": "Intellectual Property / Cash"
    },
    "Struggling Main Street Retailer": {
        "weight": 0.20,
        "base_params": {
            "business_age": (8, 4),
            "employee_count": (10, 5),
            "owner_experience": (10, 5)
        },
        "health_score_params": (0.25, 0.1), # Low health
        "financial_multipliers": {
            "revenue_per_employee": 8000,
            "expense_ratio": 0.95, # Barely profitable or losing money
            "cash_on_hand_months": 0.75
        },
        "collateral_type": "Inventory"
    },
    "Niche Service Consultant": {
        "weight": 0.30,
        "base_params": {
            "business_age": (7, 4),
            "employee_count": (4, 2),
            "owner_experience": (12, 6)
        },
        "health_score_params": (0.9, 0.05), # Very high and stable health
        "financial_multipliers": {
            "revenue_per_employee": 50000, # High value per employee
            "expense_ratio": 0.40, # High profit margin
            "cash_on_hand_months": 4.0
        },
        "collateral_type": "None"
    }
}

def generate_synthetic_data(n_samples=100_000):
    """
    Generates a high-quality synthetic dataset for SME credit risk modeling.
    """
    print(f"Generating {n_samples} SME records...")
    df = pd.DataFrame(index=range(n_samples))

    # --- Step 1: Assign Archetypes ---
    archetype_names = list(ARCHETYPES.keys())
    archetype_weights = [ARCHETYPES[name]['weight'] for name in archetype_names]
    df['archetype'] = np.random.choice(archetype_names, size=n_samples, p=archetype_weights)

    # --- Step 2 & 3: Generate Base Features & Latent Health Score per Archetype ---
    df['health_score'] = 0.0
    for name, params in ARCHETYPES.items():
        mask = df['archetype'] == name
        count = mask.sum()
        
        # Base Features
        df.loc[mask, 'business_age_years'] = np.random.normal(params['base_params']['business_age'][0], params['base_params']['business_age'][1], count)
        df.loc[mask, 'employee_count'] = np.random.normal(params['base_params']['employee_count'][0], params['base_params']['employee_count'][1], count)
        df.loc[mask, 'owner_experience_years'] = np.random.normal(params['base_params']['owner_experience'][0], params['base_params']['owner_experience'][1], count)
        
        # Latent Health Score (The Secret Sauce for Correlation)
        df.loc[mask, 'health_score'] = np.random.normal(params['health_score_params'][0], params['health_score_params'][1], count)
    
    # Clip base features to be logical
    df['business_age_years'] = df['business_age_years'].clip(1, 50)
    df['employee_count'] = df['employee_count'].clip(1).round()
    df['owner_experience_years'] = df['owner_experience_years'].clip(1, 60)
    df['health_score'] = df['health_score'].clip(0.01, 0.99)
    df['owner_age'] = df['owner_experience_years'] + np.random.uniform(22, 35, n_samples)
    df['owner_age'] = df['owner_age'].clip(20, 80).round()
    
    # --- Step 4: Generate Correlated Financial & Behavioral Features ---
    print("Generating correlated financial features...")
    
    # Revenue is a function of employees, archetype, and a trend component
    rev_per_emp = df['archetype'].map({name: p['financial_multipliers']['revenue_per_employee'] for name, p in ARCHETYPES.items()})
    df['monthly_revenue'] = df['employee_count'] * rev_per_emp * np.random.normal(1, 0.15, n_samples)
    df['revenue_trend_3mo'] = (df['health_score'] - 0.5) * np.random.uniform(0.05, 0.2, n_samples) # Trend correlated with health
    df.loc[df['archetype'] == 'High-Growth Tech Startup', 'revenue_trend_3mo'] *= 5 # Exaggerate for startups

    # Expenses are a direct function of revenue and archetype expense ratio
    expense_ratios = df['archetype'].map({name: p['financial_multipliers']['expense_ratio'] for name, p in ARCHETYPES.items()})
    df['monthly_expenses'] = df['monthly_revenue'] * expense_ratios * np.random.normal(1, 0.05, n_samples)

    # Cash & Bank Behavior is driven by health score
    cash_multipliers = df['archetype'].map({name: p['financial_multipliers']['cash_on_hand_months'] for name, p in ARCHETYPES.items()})
    df['cash_on_hand'] = df['monthly_revenue'] * cash_multipliers * df['health_score']
    df['avg_balance_last_3mo'] = df['cash_on_hand'] * np.random.normal(1.2, 0.3, n_samples)
    df['overdraft_count_3mo'] = np.random.poisson(lam = ((1 - df['health_score'])**2) * 5)

    # Payments & Invoices
    df['days_sales_outstanding'] = 30 + (1 - df['health_score']) * np.random.uniform(30, 90, n_samples)
    df['days_payables_outstanding'] = df['days_sales_outstanding'] + (df['health_score'] - 0.7) * 30

    # Credit History
    df['on_time_payment_ratio'] = (df['health_score']**1.5).clip(0.4, 0.99) * np.random.normal(1, 0.02, n_samples)
    df['founder_has_default_history'] = (np.random.rand(n_samples) > df['health_score']**0.5).astype(int)
    df['owner_credit_score'] = 300 + (df['health_score'] * 550) + np.random.normal(0, 25, n_samples)
    
    # --- Step 5: Generate Loan, Collateral & LGD Features ---
    print("Generating loan and collateral features...")
    df['credit_limit'] = (df['monthly_revenue'] * 3 * df['health_score']**2).clip(5000, 1_000_000)
    df['loan_utilization_rate'] = ((1 - df['health_score']) * np.random.uniform(0.3, 1, n_samples)).clip(0.05, 1.0)
    df['current_loan_balance'] = df['credit_limit'] * df['loan_utilization_rate']
    
    # LGD is a function of collateral
    df['collateral_type'] = df['archetype'].map({name: p['collateral_type'] for name, p in ARCHETYPES.items()})
    lgd_map = {
        "Real Estate / Equipment": np.random.normal(0.3, 0.1, n_samples),
        "Intellectual Property / Cash": np.random.normal(0.6, 0.15, n_samples),
        "Inventory": np.random.normal(0.8, 0.1, n_samples),
        "None": np.random.normal(0.95, 0.05, n_samples)
    }
    df['loss_given_default'] = df['collateral_type'].map(lambda x: float(lgd_map[x][0])).clip(0.05, 1.0)

    df['collateral_value'] = df['current_loan_balance'] * (1 - df['loss_given_default']) * np.random.normal(1.1, 0.2, n_samples)
    df['collateral_value'] = df['collateral_value'].clip(0)
    
    # --- Step 6: Calculate Derived Ratios (Crucial for logical consistency) ---
    print("Calculating derived financial ratios...")
    epsilon = 1e-6 # To avoid division by zero
    df['net_profit_margin'] = (df['monthly_revenue'] - df['monthly_expenses']) / (df['monthly_revenue'] + epsilon)
    df['expense_to_revenue_ratio'] = df['monthly_expenses'] / (df['monthly_revenue'] + epsilon)
    df['debt_to_equity_ratio'] = (1 / (df['health_score'] + epsilon)) * np.random.normal(1, 0.2, n_samples) - 1
    df['current_ratio'] = df['health_score'] * 3 + np.random.normal(0, 0.2, n_samples)
    df['interest_coverage_ratio'] = (df['net_profit_margin'] * 10 * df['health_score']).clip(-2, 50)
    
    # --- Step 7: Generate Final Target Variable (Default Flag) ---
    # Create a logit score based on key risk drivers, then convert to probability
    print("Generating final default target...")
    logit_score = -2.5 + \
                  (-5 * df['health_score']) + \
                  (2 * (1 - df['on_time_payment_ratio'])) + \
                  (1.5 * df['founder_has_default_history']) + \
                  (1 * df['debt_to_equity_ratio'].clip(-1, 5)) + \
                  np.random.normal(0, 0.5, n_samples)

    prob_default = 1 / (1 + np.exp(-logit_score))
    df['default_in_12_mo'] = (np.random.rand(n_samples) < prob_default).astype(int)

    # Add final miscellaneous features
    df['num_defaults_last_12mo'] = (df['default_in_12_mo'] * np.random.randint(1, 4, n_samples)).astype(int)
    df['industry_code'] = df['archetype'].astype('category').cat.codes + 1000 # Example NAICS codes
    df['fraud_alert_flag'] = (np.random.rand(n_samples) < 0.005).astype(int) # Rare event
    
    print("Cleaning up and finalizing dataset...")
    # Drop intermediate columns and reorder for clarity
    df = df.drop(columns=['health_score'])
    final_columns = [
        'archetype', 'default_in_12_mo', 'loss_given_default', 'credit_limit', # Targets & Key Components
        'monthly_revenue', 'monthly_expenses', 'net_profit_margin', 'cash_on_hand', 'current_loan_balance',
        'receivables_total', 'payables_total', 'loan_utilization_rate', 'days_sales_outstanding',
        'days_payables_outstanding', 'business_age_years', 'employee_count', 'industry_code',
        'revenue_trend_3mo', 'debt_to_equity_ratio', 'current_ratio', 'on_time_payment_ratio',
        'owner_credit_score', 'owner_age', 'owner_experience_years', 'founder_has_default_history',
        'collateral_type', 'collateral_value', 'num_defaults_last_12mo', 'overdraft_count_3mo', 'fraud_alert_flag'
    ]
    # Fill any missing columns from the original list with plausible random data
    for col in final_columns:
        if col not in df.columns:
            df[col] = np.random.randn(n_samples) * 1000 # Placeholder

    df = df.reindex(columns=final_columns)
    
    print(f"\nDataset generation complete. Shape: {df.shape}")
    print(f"Default Rate: {df['default_in_12_mo'].mean():.2%}")
    return df


if __name__ == '__main__':
    # --- Generate and Inspect the Dataset ---
    sme_data = generate_synthetic_data(n_samples=100_000)

    print("\n--- Dataset Head ---")
    print(sme_data.head())

    print("\n--- Dataset Info ---")
    sme_data.info()
    
    print("\n--- Archetype Value Counts ---")
    print(sme_data['archetype'].value_counts())

    # Save the dataset to a file
    sme_data.to_csv('sme_synthetic_credit_data.csv', index=False)
    print("\nDataset saved to 'sme_synthetic_credit_data.csv'")