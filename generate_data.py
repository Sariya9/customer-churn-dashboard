import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'CustomerID': range(1, n+1),
    'Age': np.random.randint(18, 70, n),
    'Tenure_Months': np.random.randint(1, 72, n),
    'Monthly_Charges': np.round(np.random.uniform(20, 120, n), 2),
    'Total_Charges': np.round(np.random.uniform(100, 8000, n), 2),
    'Num_Products': np.random.randint(1, 5, n),
    'Support_Calls': np.random.randint(0, 10, n),
    'Payment_Delay_Days': np.random.randint(0, 30, n),
    'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.5, 0.3, 0.2]),
    'Internet_Service': np.random.choice(['DSL', 'Fibre', 'None'], n, p=[0.4, 0.4, 0.2]),
    'Gender': np.random.choice(['Male', 'Female'], n),
})

# Strong deterministic churn signal
contract_score = df['Contract_Type'].map({'Month-to-Month': 1.0, 'One Year': 0.3, 'Two Year': 0.0})
tenure_score   = 1 - (df['Tenure_Months'] / 72)
support_score  = df['Support_Calls'] / 10
delay_score    = df['Payment_Delay_Days'] / 30
charge_score   = df['Monthly_Charges'] / 120

churn_score = (
    0.35 * contract_score +
    0.25 * tenure_score +
    0.20 * support_score +
    0.12 * delay_score +
    0.08 * charge_score
)

# Add small noise to keep it realistic
noise = np.random.normal(0, 0.05, n)
churn_prob = np.clip(churn_score + noise, 0, 1)
df['Churned'] = (churn_prob > 0.45).astype(int)

df.to_csv('/Users/sariyalura/Documents/ChurnDashboard/customer_data.csv', index=False)
print(f"Dataset created: {n} customers, {df['Churned'].sum()} churned ({df['Churned'].mean()*100:.1f}%)")
