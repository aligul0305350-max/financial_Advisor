import os
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

from load_data import load_and_clean
from eda_numpy import compute_summary

# Groq API key .env se uthao
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in .env file")

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

# Data Context
def load_context(filepath="pakistan_statement_30k_enhanced.csv"):
    df, expenses, income = load_and_clean(filepath)
    summary = compute_summary(expenses, income)

    top_cat = summary['category_sum'].sort_values(ascending=False).head(10).to_dict()
    monthly_exp = summary['monthly_expenses'].to_string()
    top_merchants = expenses['Description'].value_counts().head(5).to_dict()
    sample_expenses = expenses[['Date','Description','Amount','Category']].head(5).to_csv(index=False)

    # Monthly category breakdown
    monthly_cat = expenses.groupby(['Month', 'Category'])['Amount'].sum().unstack(fill_value=0)
    monthly_cat_str = monthly_cat.to_string()

    # Monthly income
    monthly_income = income.groupby('Month')['Amount'].sum()
    monthly_income_str = monthly_income.to_string()

    # ********** NEW: Daily expenses (total per day) **********
    daily_expenses = expenses.groupby('Date')['Amount'].sum()
    daily_expenses_str = daily_expenses.to_string()
    # ********************************************************

    context = {
        "total_income": summary['total_income'],
        "total_expenses": summary['total_expenses'],
        "net_savings": summary['net_savings'],
        "avg_daily_expense": summary['avg_daily_expense'],
        "avg_monthly_expense": summary['avg_monthly_expense'],
        "top_categories": top_cat,
        "monthly_expenses_table": monthly_exp,
        "monthly_category_breakdown": monthly_cat_str,
        "monthly_income_table": monthly_income_str,
        "daily_expenses_table": daily_expenses_str,
        "top_merchants_frequency": top_merchants,
        "sample_transactions_csv": sample_expenses
    }
    return context

def build_question_prompt(question, context):
    prompt = f"""
You are a personal finance assistant. Only use the financial data provided below.

- Total Income: PKR {context['total_income']:,.2f}
- Total Expenses: PKR {context['total_expenses']:,.2f}
- Net Savings: PKR {context['net_savings']:,.2f}
- Average Daily Expense: PKR {context['avg_daily_expense']:.2f}
- Average Monthly Expense: PKR {context['avg_monthly_expense']:.2f}
- Top Spending Categories (overall): {context['top_categories']}
- Monthly Expenses (total per month):
{context['monthly_expenses_table']}
- Monthly Category Breakdown (expenses per category per month):
{context['monthly_category_breakdown']}
- Monthly Income (credit) per month:
{context['monthly_income_table']}
- Daily Expenses (total per date):
{context['daily_expenses_table']}
- Top 5 Most Frequent Merchants: {context['top_merchants_frequency']}
- Sample Transactions:
{context['sample_transactions_csv']}

Instructions:
- Answer the user's question using ONLY the data above.
- Do NOT use outside knowledge.
- If data insufficient, say "Mere paas iska jawab dene ke liye kaafi data nahi hai."
- When asked about a specific date's expenses, find the date in the Daily Expenses table and give the exact amount.
- **Answer in Roman Urdu (Urdu written in English script) only.** Keep it short and helpful.

User Question: {question}

Answer in Roman Urdu:
"""
    return prompt

# ------------------------------------------------------------
# API call helper (only question remains)
# ------------------------------------------------------------
def ask_question(question, context):
    prompt = build_question_prompt(question, context)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

# ------------------------------------------------------------
# Interactive Loop
# ------------------------------------------------------------
if __name__ == "__main__":
    print("📊 Personal Finance Assistant (Roman Urdu)")
    print("Data load ho raha hai...")
    context = load_context()
    print("Tayar hai! Sawaal puchhein.\n")

    print("Kya karna chahte hain?")
    print("  - Koi bhi sawaal Roman Urdu mein likhein")
    print("  - 'exit' likhein → band karne ke liye\n")

    while True:
        user_input = input("❓ Sawaal: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Allah Hafiz! 😊")
            break

        # Only one path – ask question
        print("\n🤔 Jawab generate ho raha hai...\n")
        answer = ask_question(user_input, context)
        print(answer)
        print("─" * 50)