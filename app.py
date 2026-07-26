import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# ---------- LOAD DATA (cached so it doesn't reload every interaction) ----------
@st.cache_data
def load_data():
    conn = sqlite3.connect("churn.db")
    df = pd.read_sql("""
        SELECT c.*, b.tenure, b.MonthlyCharges, b.TotalCharges, b.ChurnFlag
        FROM customers c
        JOIN billing b ON c.customerID = b.customerID
    """, conn)
    conn.close()
    return df

@st.cache_resource
def train_model(df):
    features_df = df.drop(columns=['customerID'])
    features_df = pd.get_dummies(features_df, drop_first=True)
    X = features_df.drop(columns=['ChurnFlag'])
    y = features_df['ChurnFlag']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, X, auc

df = load_data()
model, X, auc = train_model(df)

# Score every customer
df['churn_risk_score'] = model.predict_proba(X)[:, 1]

# ---------- HEADER ----------
st.title("📊 Customer Churn Analysis Dashboard")
st.caption(f"Random Forest model — ROC-AUC: {auc:.3f}")

# ---------- KEY METRICS ----------
col1, col2, col3, col4 = st.columns(4)
total_customers = len(df)
churn_rate = df['ChurnFlag'].mean() * 100
avg_monthly_churned = df[df['ChurnFlag'] == 1]['MonthlyCharges'].mean()
annual_revenue_lost = df[df['ChurnFlag'] == 1]['MonthlyCharges'].sum() * 12

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churn Rate", f"{churn_rate:.1f}%")
col3.metric("Avg Monthly Charge (Churned)", f"${avg_monthly_churned:.2f}")
col4.metric("Est. Annual Revenue Lost", f"${annual_revenue_lost:,.0f}")

st.divider()

# ---------- FILTERS (SIDEBAR) ----------
st.sidebar.header("Filters")
contract_filter = st.sidebar.multiselect(
    "Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique()
)
internet_filter = st.sidebar.multiselect(
    "Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique()
)

filtered_df = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter))
]

# ---------- CHARTS ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = filtered_df.groupby('Contract')['ChurnFlag'].mean().reset_index()
    contract_churn['ChurnFlag'] *= 100
    fig = px.bar(contract_churn, x='Contract', y='ChurnFlag',
                 labels={'ChurnFlag': 'Churn Rate (%)'}, color='Contract')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Churn Rate by Internet Service")
    internet_churn = filtered_df.groupby('InternetService')['ChurnFlag'].mean().reset_index()
    internet_churn['ChurnFlag'] *= 100
    fig = px.bar(internet_churn, x='InternetService', y='ChurnFlag',
                 labels={'ChurnFlag': 'Churn Rate (%)'}, color='InternetService')
    st.plotly_chart(fig, use_container_width=True)

# ---------- HIGH-RISK CUSTOMER TABLE ----------
st.subheader("🎯 Top At-Risk Customers")
top_n = st.slider("Show top N highest-risk customers", 10, 200, 50)

risk_table = filtered_df.sort_values('churn_risk_score', ascending=False).head(top_n)
st.dataframe(
    risk_table[['customerID', 'Contract', 'InternetService', 'PaymentMethod',
                'tenure', 'MonthlyCharges', 'churn_risk_score']],
    use_container_width=True
)

st.caption("Built as a customer churn analytics project — SQL + Python + scikit-learn + Streamlit")