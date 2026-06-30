import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# Load data
df = pd.read_csv('customer_data.csv')

# Encode categorical columns into numbers
df['Contract_Enc'] = LabelEncoder().fit_transform(df['Contract_Type'])
df['Internet_Enc'] = LabelEncoder().fit_transform(df['Internet_Service'])
df['Gender_Enc']   = LabelEncoder().fit_transform(df['Gender'])

# Define features and target
features = ['Age', 'Tenure_Months', 'Monthly_Charges', 'Total_Charges',
            'Num_Products', 'Support_Calls', 'Payment_Delay_Days',
            'Contract_Enc', 'Internet_Enc', 'Gender_Enc']

X = df[features]
y = df['Churned']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train SVM
svm = SVC(probability=True, kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

# Evaluate both models
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
svm_auc = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:,1])

print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"SVM AUC:           {svm_auc:.4f}")

# Feature importance
import pandas as pd
importance = pd.Series(rf.feature_importances_, index=features)
print("\nTop 5 Churn Drivers:")
print(importance.sort_values(ascending=False).head())