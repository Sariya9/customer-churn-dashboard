import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import anthropic

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Risk & Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Customer Risk & Churn Dashboard")
st.markdown("*Predictive analytics for customer retention — powered by Random Forest, SVM & LLM insights*")

# ── Load & preprocess data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("customer_data.csv")
    return df

@st.cache_resource
def train_models(df):
    le_contract = LabelEncoder()
    le_internet = LabelEncoder()
    le_gender   = LabelEncoder()

    df = df.copy()
    df['Contract_Enc']  = le_contract.fit_transform(df['Contract_Type'])
    df['Internet_Enc']  = le_internet.fit_transform(df['Internet_Service'])
    df['Gender_Enc']    = le_gender.fit_transform(df['Gender'])

    features = ['Age','Tenure_Months','Monthly_Charges','Total_Charges',
                'Num_Products','Support_Calls','Payment_Delay_Days',
                'Contract_Enc','Internet_Enc','Gender_Enc']

    X = df[features]
    y = df['Churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    rf.fit(X_train, y_train)
    svm.fit(X_train_scaled, y_train)

    rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    svm_auc = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:,1])

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    return rf, svm, scaler, rf_auc, svm_auc, importances, features, X_test, X_test_scaled, y_test

df = load_data()
rf, svm, scaler, rf_auc, svm_auc, importances, features, X_test, X_test_scaled, y_test = train_models(df)

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
contract_filter  = st.sidebar.multiselect("Contract Type", df['Contract_Type'].unique(), default=list(df['Contract_Type'].unique()))
internet_filter  = st.sidebar.multiselect("Internet Service", df['Internet_Service'].unique(), default=list(df['Internet_Service'].unique()))
tenure_range     = st.sidebar.slider("Tenure (Months)", 1, 72, (1, 72))
model_choice     = st.sidebar.radio("Prediction Model", ["Random Forest", "SVM"])

filtered = df[
    (df['Contract_Type'].isin(contract_filter)) &
    (df['Internet_Service'].isin(internet_filter)) &
    (df['Tenure_Months'].between(*tenure_range))
]

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown("### 📈 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers",    f"{len(filtered):,}")
k2.metric("Churned",            f"{filtered['Churned'].sum():,}", delta=f"{filtered['Churned'].mean()*100:.1f}% rate", delta_color="inverse")
k3.metric("Avg Monthly Charge", f"${filtered['Monthly_Charges'].mean():.2f}")
k4.metric("RF AUC Score",       f"{rf_auc:.4f}")
k5.metric("SVM AUC Score",      f"{svm_auc:.4f}")

st.divider()

# ── Charts Row 1 ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Churn Analysis")
c1, c2, c3 = st.columns(3)

with c1:
    churn_counts = filtered['Churned'].value_counts().reset_index()
    churn_counts.columns = ['Status', 'Count']
    churn_counts['Status'] = churn_counts['Status'].map({0: 'Retained', 1: 'Churned'})
    fig = px.pie(churn_counts, values='Count', names='Status',
                 color_discrete_map={'Retained':'#888888','Churned':'#e74c3c'},
                 title="Churn Distribution",
                 color_discrete_sequence=['#888888','#e74c3c'])
    st.plotly_chart(fig, use_container_width=True)

with c2:
    churn_contract = filtered.groupby('Contract_Type')['Churned'].mean().reset_index()
    churn_contract.columns = ['Contract', 'Churn Rate']
    fig = px.bar(churn_contract, x='Contract', y='Churn Rate',
                 color='Churn Rate', color_continuous_scale=[[0,'#f0f0f0'],[0.5,'#888888'],[1,'#e74c3c']],
                 title="Churn Rate by Contract Type")
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

with c3:
    fig = px.histogram(filtered, x='Tenure_Months', color=filtered['Churned'].map({0:'Retained',1:'Churned'}),
                       barmode='overlay', title="Tenure Distribution by Churn",
                       color_discrete_map={'Retained':'#888888','Churned':'#e74c3c'})
    st.plotly_chart(fig, use_container_width=True)

# ── Charts Row 2 ─────────────────────────────────────────────────────────────
c4, c5 = st.columns(2)

with c4:
    fig = px.bar(importances.reset_index(), x='index', y=importances.values,
                 title="Feature Importance (Random Forest)",
                 labels={'index':'Feature','y':'Importance'},
                 color=importances.values, color_continuous_scale=[[0,'#f0f0f0'],[0.5,'#888888'],[1,'#111111']])
    st.plotly_chart(fig, use_container_width=True)

with c5:
    fig = px.scatter(filtered, x='Tenure_Months', y='Monthly_Charges',
                     color=filtered['Churned'].map({0:'Retained',1:'Churned'}),
                     title="Monthly Charges vs Tenure",
                     color_discrete_map={'Retained':'#888888','Churned':'#e74c3c'},
                     opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Individual Customer Risk Predictor ────────────────────────────────────────
st.markdown("### 🎯 Individual Customer Risk Predictor")
st.markdown("Enter customer details to predict churn risk and get AI-generated retention insights.")

col1, col2, col3 = st.columns(3)
with col1:
    age              = st.slider("Age", 18, 70, 35)
    tenure           = st.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges  = st.slider("Monthly Charges ($)", 20, 120, 65)
with col2:
    total_charges    = st.slider("Total Charges ($)", 100, 8000, 800)
    num_products     = st.slider("Number of Products", 1, 4, 2)
    support_calls    = st.slider("Support Calls", 0, 10, 3)
with col3:
    payment_delay    = st.slider("Payment Delay (Days)", 0, 30, 5)
    contract_type    = st.selectbox("Contract Type", ['Month-to-Month','One Year','Two Year'])
    internet_service = st.selectbox("Internet Service", ['DSL','Fibre','None'])
    gender           = st.selectbox("Gender", ['Male','Female'])

contract_enc  = {'Month-to-Month':0,'One Year':1,'Two Year':2}[contract_type]
internet_enc  = {'DSL':0,'Fibre':1,'None':2}[internet_service]
gender_enc    = {'Female':0,'Male':1}[gender]

input_data = [[age, tenure, monthly_charges, total_charges,
               num_products, support_calls, payment_delay,
               contract_enc, internet_enc, gender_enc]]

if model_choice == "Random Forest":
    churn_prob = rf.predict_proba(input_data)[0][1]
else:
    input_scaled = scaler.transform(input_data)
    churn_prob = svm.predict_proba(input_scaled)[0][1]
risk_level = "🔴 High Risk" if churn_prob > 0.6 else "🟡 Medium Risk" if churn_prob > 0.35 else "🟢 Low Risk"

r1, r2 = st.columns([1,2])
with r1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#e74c3c" if churn_prob > 0.6 else "#e74c3c" if churn_prob > 0.35 else "#888888"},
            'steps': [
                {'range': [0, 35],  'color': '#f0f0f0'},
                {'range': [35, 60], 'color': '#f5f5f5'},
                {'range': [60, 100],'color': '#ffe5e5'},
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Risk Level: {risk_level}**")

with r2:
    st.markdown("#### 🤖 AI Retention Insight")
    api_key = st.text_input("Enter Anthropic API Key", type="password", placeholder="sk-ant-...")

    if st.button("Generate AI Insight"):
        if not api_key:
            st.warning("Please enter your Anthropic API key.")
        else:
            with st.spinner("Generating insight..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    prompt = f"""You are a customer retention analyst. A customer has the following profile:
- Age: {age}, Gender: {gender}
- Tenure: {tenure} months, Contract: {contract_type}
- Monthly Charges: ${monthly_charges}, Total Charges: ${total_charges}
- Products: {num_products}, Support Calls: {support_calls}, Payment Delay: {payment_delay} days
- Internet Service: {internet_service}
- Churn Probability: {churn_prob*100:.1f}% ({risk_level})

In 3-4 sentences: identify the top 2 churn risk drivers for this customer and recommend one specific, actionable retention strategy. Be concise and practical."""

                    message = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=300,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.success(message.content[0].text)
                except Exception as e:
                    st.error(f"API error: {e}")

st.divider()

# ── Model Performance ─────────────────────────────────────────────────────────
st.markdown("### 🧪 Model Performance")
m1, m2 = st.columns(2)

with m1:
    st.markdown("**Random Forest**")
    rf_preds = rf.predict(X_test)
    report = classification_report(y_test, rf_preds, output_dict=True)
    perf_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(perf_df[['precision','recall','f1-score']].iloc[:2], use_container_width=True)
    st.metric("AUC Score", f"{rf_auc:.4f}")

with m2:
    st.markdown("**SVM**")
    svm_preds = svm.predict(X_test_scaled)
    report2 = classification_report(y_test, svm_preds, output_dict=True)
    perf_df2 = pd.DataFrame(report2).transpose().round(3)
    st.dataframe(perf_df2[['precision','recall','f1-score']].iloc[:2], use_container_width=True)
    st.metric("AUC Score", f"{svm_auc:.4f}")

st.divider()
st.markdown("*Built by Sariya Akhter Lura — AI & Data Analytics Research, MIT Melbourne*")
