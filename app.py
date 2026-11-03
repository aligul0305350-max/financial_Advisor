import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import csv

from load_data import load_and_clean
from eda_numpy import compute_summary
from nlp_assistant import ask_question, load_context

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="💰 Finance AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Custom CSS for Sidebar & Form
# ------------------------------
st.markdown(
    """
<style>
   /* Global dark theme */
.main { background-color: #0b0f19; }
.stApp { background-color: #0b0f19; color: #e0e0e0; }

/* ---------- Sidebar Button Nav ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b25 100%);
    border-right: 1px solid #2a2a3c;
}
.sidebar-logo {
    font-size: 2rem;
    font-weight: 800;
    color: #a78bfa;
    text-align: center;
    padding: 1.8rem 0 1rem 0;
    letter-spacing: 1.5px;
    text-shadow: 0 0 20px rgba(167,139,250,0.3);
}
div[role="radiogroup"] {
    margin: 1.5rem 0.8rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}
div[role="radiogroup"] label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.9rem 1.2rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 500;
    color: #b0b0c0;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.25s ease;
    cursor: pointer;
    backdrop-filter: blur(10px);
}
div[role="radiogroup"] label:hover {
    background: rgba(167,139,250,0.12);
    border-color: rgba(167,139,250,0.3);
    color: #ffffff;
    transform: translateX(4px);
}
div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
    border-color: #8b5cf6;
    color: white;
    font-weight: 600;
    box-shadow: 0 8px 20px -8px rgba(124,58,237,0.6);
    transform: scale(1.02);
}
div[role="radiogroup"] label[data-checked="true"] p {
    color: white !important;
}

/* Metric cards */
.metric-card {
    background: #161b25;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #2a2a3c;
}
.metric-card h3 {
    font-size: 0.9rem;
    color: #a0a0b0;
    margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
}

/* Form card */
.form-card {
    background: #161b25;
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #2a2a3c;
    margin-top: 1rem;
}
.form-card h2 {
    font-size: 1.3rem;
    margin-bottom: 1.5rem;
    color: #a78bfa;
}
.stTextInput>div>div>input, .stDateInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 0.6rem 0.8rem !important;
}
.stTextInput>div>div>input:focus, .stDateInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stNumberInput>div>div>input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.3) !important;
}
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    border: none;
    transition: all 0.2s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.5);
}

/* Answer card */
.answer-card {
    background: #161b25;
    border-left: 4px solid #7c3aed;
    padding: 1.5rem;
    border-radius: 12px;
    margin-top: 1rem;
}
.answer-card h4 {
    color: #a78bfa;
    margin-bottom: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------
# Cached Data Loader
# ------------------------------
@st.cache_data(ttl=10)
def load_data(filepath):
    return load_and_clean(filepath)


FILEPATH = "pakistan_statement_30k_enhanced.csv"
df, expenses, income = load_data(FILEPATH)
summary = compute_summary(expenses, income)
nlp_context = load_context(FILEPATH)

# ------------------------------
# Sidebar with custom navigation
# ------------------------------
st.sidebar.markdown('<div class="sidebar-logo">💸 FinAI</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["📋 Transactions", "📊 Charts", "🤖 AI Assistant"],
    label_visibility="collapsed",
)

# Date filter for transactions
if page == "📋 Transactions":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗓️ Filter by Date")
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "", (min_date, max_date), label_visibility="collapsed"
    )
    if len(date_range) == 2:
        start, end = date_range
        mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
        filtered_df = df[mask]
    else:
        filtered_df = df
    st.sidebar.caption(f"Showing {len(filtered_df)} transactions")

# ------------------------------
# Page 1: Transactions
# ------------------------------
if page == "📋 Transactions":
    st.header("💳 All Transactions")

    # Metrics
    total_credit = filtered_df[filtered_df["Type"] == "Credit"]["Amount"].sum()
    total_debit = filtered_df[filtered_df["Type"] == "Debit"]["Amount"].sum()
    net = total_credit - total_debit

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><h3>Total Income</h3><div class="metric-value">PKR {:,.2f}</div></div>'.format(
                total_credit
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><h3>Total Expenses</h3><div class="metric-value">PKR {:,.2f}</div></div>'.format(
                total_debit
            ),
            unsafe_allow_html=True,
        )
    with col3:
        arrow = "⬆️" if net >= 0 else "⬇️"
        st.markdown(
            f'<div class="metric-card"><h3>Net</h3><div class="metric-value">PKR {net:,.2f} {arrow}</div></div>',
            unsafe_allow_html=True,
        )

    # Transaction table
    st.dataframe(
        filtered_df[["Date", "Description", "Amount", "Type", "Category"]].sort_values(
            "Date", ascending=False
        ),
        use_container_width=True,
        height=380,
    )

    # ------------------------------
    # Enhanced Add New Transaction Form
    # ------------------------------
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("<h2>➕ Add New Transaction</h2>", unsafe_allow_html=True)

    with st.form("new_txn", clear_on_submit=True):
        # Two rows of fields
        row1_col1, row1_col2, row1_col3 = st.columns([1.2, 2, 1])
        with row1_col1:
            txn_date = st.date_input("📅 Date", datetime.today())
        with row1_col2:
            txn_desc = st.text_input(
                "📝 Description", placeholder="e.g. Imtiaz Super Market"
            )
        with row1_col3:
            txn_amount = st.number_input(
                "💰 Amount", min_value=0.0, step=10.0, format="%.2f"
            )

        row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])
        with row2_col1:
            txn_type = st.selectbox("🔀 Type", ["Debit", "Credit"])
        with row2_col2:
            txn_cat = st.selectbox("📂 Category", sorted(df["Category"].unique()))
        # empty column for spacing
        row2_col3.markdown("")

        submitted = st.form_submit_button("💾 Save Transaction")
        if submitted:
            new_row = pd.DataFrame(
                [
                    {
                        "Date": txn_date.strftime("%Y-%m-%d"),
                        "Description": txn_desc,
                        "Amount": txn_amount,
                        "Type": txn_type,
                        "Category": txn_cat,
                    }
                ]
            )
            new_row.to_csv(
                FILEPATH, mode="a", index=False, header=False, quoting=csv.QUOTE_ALL
            )
            st.success("✅ Transaction saved successfully!")
            # Clear cache and rerun to refresh data
            load_data.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "📊 Charts":
    st.header("📊 Financial Insights")

    # Date filter for charts
    st.markdown("### 🗓️ Filter Charts by Date")
    col_start, col_end = st.columns(2)
    with col_start:
        chart_start = st.date_input("Start Date", value=df["Date"].min().date())
    with col_end:
        chart_end = st.date_input("End Date", value=df["Date"].max().date())

    chart_mask = (expenses["Date"].dt.date >= chart_start) & (
        expenses["Date"].dt.date <= chart_end
    )
    filtered_expenses = expenses[chart_mask]

    if filtered_expenses.empty:
        st.warning("Is date range mein koi expense nahi hai.")
    else:
        monthly_exp = filtered_expenses.groupby("Month")["Amount"].sum()
        cat_sum = filtered_expenses.groupby("Category")["Amount"].sum()
        top_merchants = filtered_expenses["Description"].value_counts().nlargest(10)

        # ---------- Custom color palette ----------
        unique_colors = [
            "#00bfa5",
            "#ff6f61",
            "#ffa600",
            "#665191",
            "#a05195",
            "#d45087",
            "#f95d6a",
            "#ff7c43",
            "#ffa600",
            "#003f5c",
        ]

        # ---------- Row 1: Two charts side by side ----------
        row1_left, row1_right = st.columns(2, gap="large")
        with row1_left:
            with st.container(border=True):
                st.subheader("📈 Monthly Spending Trend")
                monthly_exp_df = monthly_exp.reset_index()
                monthly_exp_df.columns = ["Month", "Amount"]
                monthly_exp_df["Month"] = monthly_exp_df["Month"].astype(str)
                fig = px.line(
                    monthly_exp_df,
                    x="Month",
                    y="Amount",
                    markers=True,
                    color_discrete_sequence=[unique_colors[0]],
                )
                fig.update_traces(
                    line=dict(width=3), marker=dict(size=10, color=unique_colors[0])
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#333"),
                )
                st.plotly_chart(fig, use_container_width=True)

        with row1_right:
            with st.container(border=True):
                st.subheader("🍩 Category Distribution")
                cat_sum_df = cat_sum.sort_values(ascending=False).reset_index()
                cat_sum_df.columns = ["Category", "Amount"]
                fig2 = px.pie(
                    cat_sum_df,
                    values="Amount",
                    names="Category",
                    hole=0.7,
                    color_discrete_sequence=unique_colors[1:],
                )
                fig2.update_traces(
                    textinfo="percent+label",
                    textfont_color="white",
                    marker=dict(line=dict(color="#0b0f19", width=2)),
                )
                fig2.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ---------- Row 2: Full-width chart ----------
        st.markdown("---")
        with st.container(border=True):
            st.subheader("🏪 Top 10 Frequent Merchants")
            top_merchants_df = top_merchants.reset_index()
            top_merchants_df.columns = ["Merchant", "Count"]
            fig3 = px.bar(
                top_merchants_df,
                y="Merchant",
                x="Count",
                orientation="h",
                color="Count",
                color_continuous_scale=unique_colors[:6],
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                coloraxis_showscale=False,
            )
            fig3.update_traces(marker=dict(line=dict(width=0)))
            st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Page 3: AI Assistant
# ------------------------------
elif page == "🤖 AI Assistant":
    st.header("🤖 Ask Your Finance AI")
    st.caption(
        "Ask any Question Regarding your Data"
    )

    prompt = st.text_area(
        "", placeholder="Ask Question", height=100, key="prompt_input"
    )
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        ask_btn = st.button("Send Prompt", use_container_width=True)

    if ask_btn and prompt.strip():
        with st.spinner("thinking..."):
            try:
                answer = ask_question(prompt, nlp_context)
                st.markdown(
                    f"""
                <div class="answer-card">
                    <h4>🧠 Answer</h4>
                    <p style="font-size:1.05rem; color:#cfcfcf;">{answer}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error: {e}")
    elif ask_btn:
        st.warning("type any Question!")
