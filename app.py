import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pulp

st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")

st.title("💰 Smart Expense Categorization & Savings Optimizer")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("User Inputs")

uploaded_file = st.sidebar.file_uploader("Upload Bank Statement (CSV)", type=["csv"])

user_name = st.sidebar.text_input("Account Holder Name (for Self Transfer)")
landlord_name = st.sidebar.text_input("Landlord Name (for Rent)")

savings_target = st.sidebar.number_input("Monthly Savings Target (₹)", min_value=0, value=2000)

# ===============================
# LOAD TRAIN DATA (MODEL)
# ===============================
@st.cache_resource
def train_model():
    train_df = pd.read_csv("financial_transaction_train1.csv")

    X_train = train_df["Transaction_Text"].astype(str)
    y_train = train_df["Label"].astype(str)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=2000)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

model = train_model()

# ===============================
# PROCESS USER FILE
# ===============================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # Clean amounts
    df['Withdrawal Amt.'] = df['Withdrawal Amt.'].fillna(0)
    df['Deposit Amt.'] = df['Deposit Amt.'].fillna(0)

    # Standardize
    def get_amount_type(row):
        if row['Deposit Amt.'] > 0:
            return row['Deposit Amt.'], 'Credit'
        else:
            return row['Withdrawal Amt.'], 'Debit'

    df[['Amount', 'Type']] = df.apply(get_amount_type, axis=1, result_type='expand')

    # ML Prediction
    df['Category'] = model.predict(df['Narration'].astype(str))

    # Overrides
    if user_name:
        df.loc[df['Narration'].str.upper().str.contains(user_name.upper(), na=False), 'Category'] = 'Self Transfer'

    if landlord_name:
        df.loc[df['Narration'].str.upper().str.contains(landlord_name.upper(), na=False), 'Category'] = 'Rent'

    # Filter Debit
    expenses = df[df['Type'] == 'Debit'].copy()

    st.subheader("🤖 Categorized Expenses")
    st.dataframe(expenses[['Date', 'Narration', 'Amount', 'Category']])

    # ===============================
    # DASHBOARD
    # ===============================
    st.subheader("📊 Expense Breakdown")

    category_summary = expenses.groupby('Category')['Amount'].sum().sort_values(ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Category-wise Spend")
        st.bar_chart(category_summary)

    with col2:
        fig, ax = plt.subplots()
        category_summary.plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    # ===============================
    # OPTIMIZATION
    # ===============================
    st.subheader("🎯 Savings Optimization")

    # Monthly avg
    expenses['Date'] = pd.to_datetime(expenses['Date'], errors='coerce', dayfirst=True)
    expenses['Month'] = expenses['Date'].dt.to_period('M')

    num_months = expenses['Month'].nunique()
    if num_months == 0:
        num_months = 1

    monthly_avg = expenses.groupby('Category')['Amount'].sum() / num_months

    categories = monthly_avg.index.tolist()

    prob = pulp.LpProblem("Savings", pulp.LpMinimize)
    cuts = pulp.LpVariable.dicts("Cut", categories, lowBound=0)

    # Objective
    prob += pulp.lpSum([cuts[c] for c in categories])

    # Constraint
    prob += pulp.lpSum([cuts[c] for c in categories]) == savings_target

    for c in categories:
        prob += cuts[c] <= monthly_avg[c] * 0.4

    prob.solve()

    results = []
    for c in categories:
        if cuts[c].value() > 0:
            results.append({
                "Category": c,
                "Current Avg": monthly_avg[c],
                "Cut": cuts[c].value(),
                "New Budget": monthly_avg[c] - cuts[c].value()
            })

    st.dataframe(pd.DataFrame(results))

else:
    st.info("👆 Upload your bank statement to get started")	