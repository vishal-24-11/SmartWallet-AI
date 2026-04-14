import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pulp

st.set_page_config(page_title="Expense Dashboard", layout="wide")

st.title("💰 Smart Expense Analyzer & Savings Optimizer")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload Bank Statement", type=["csv"])

user_name = st.sidebar.text_input("Account Holder Name (Self Transfer)")
landlord_name = st.sidebar.text_input("Landlord Name (Rent)")

savings_target = st.sidebar.number_input("Monthly Savings Target (₹)", min_value=0, value=2000)

# ===============================
# MODEL TRAINING
# ===============================
@st.cache_resource
def train_model():
    df = pd.read_csv("financial_transaction_train1.csv")

    X = df["Transaction_Text"].astype(str)
    y = df["Label"].astype(str)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=2000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model

model = train_model()

# ===============================
# MAIN LOGIC
# ===============================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    # Clean amounts
    df['Withdrawal Amt.'] = df['Withdrawal Amt.'].fillna(0)
    df['Deposit Amt.'] = df['Deposit Amt.'].fillna(0)

    def get_amount(row):
        if row['Deposit Amt.'] > 0:
            return row['Deposit Amt.'], "Credit"
        else:
            return row['Withdrawal Amt.'], "Debit"

    df[['Amount', 'Type']] = df.apply(get_amount, axis=1, result_type='expand')

    # ML Prediction
    df['Category'] = model.predict(df['Narration'].astype(str))

    # Overrides
    if user_name:
        df.loc[df['Narration'].str.upper().str.contains(user_name.upper(), na=False), 'Category'] = 'Self Transfer'

    if landlord_name:
        df.loc[df['Narration'].str.upper().str.contains(landlord_name.upper(), na=False), 'Category'] = 'Rent'

    # Only expenses
    expenses = df[df['Type'] == 'Debit'].copy()

    # ===============================
    # DATE + MONTH
    # ===============================
    expenses['Date'] = pd.to_datetime(expenses['Date'], errors='coerce', dayfirst=True)
    expenses['Month'] = expenses['Date'].dt.strftime('%b %Y')

    # ===============================
    # MONTH FILTER
    # ===============================
    st.sidebar.subheader("📅 Filter by Month")

    all_months = sorted(expenses['Month'].dropna().unique())

    selected_months = st.sidebar.multiselect(
        "Select Month(s)",
        options=all_months,
        default=all_months
    )

    # Apply filter
    if selected_months:
        filtered_expenses = expenses[expenses['Month'].isin(selected_months)]
    else:
        filtered_expenses = expenses.copy()

    st.subheader("📊 Filtered Expenses")
    st.write(f"Showing data for: {', '.join(selected_months)}")
    st.dataframe(filtered_expenses[['Date', 'Narration', 'Amount', 'Category']])

    # ===============================
    # DASHBOARD
    # ===============================
    st.subheader("📈 Expense Dashboard")

    category_summary = filtered_expenses.groupby('Category')['Amount'].sum().sort_values(ascending=False)

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

    num_months = filtered_expenses['Month'].nunique()
    if num_months == 0:
        num_months = 1

    monthly_avg = filtered_expenses.groupby('Category')['Amount'].sum() / num_months

    categories = monthly_avg.index.tolist()

    prob = pulp.LpProblem("Savings", pulp.LpMinimize)
    cuts = pulp.LpVariable.dicts("Cut", categories, lowBound=0)

    prob += pulp.lpSum([cuts[c] for c in categories])
    prob += pulp.lpSum([cuts[c] for c in categories]) == savings_target

    for c in categories:
        prob += cuts[c] <= monthly_avg[c] * 0.4

    prob.solve()

    results = []
    for c in categories:
        if cuts[c].value() and cuts[c].value() > 0:
            results.append({
                "Category": c,
                "Current Avg": round(monthly_avg[c], 2),
                "Suggested Cut": round(cuts[c].value(), 2),
                "New Budget": round(monthly_avg[c] - cuts[c].value(), 2)
            })

    st.dataframe(pd.DataFrame(results))

else:
    st.info("👆 Upload your bank statement to begin")
