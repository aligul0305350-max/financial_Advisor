import numpy as np
import pandas as pd
from load_data import load_and_clean

def compute_summary(expenses_df, income_df):
    """
    Uses numpy to compute:
    - Total income, total expenses, net savings
    - Average daily spend
    - Category-wise total and average
    - Monthly breakdown
    Returns a dictionary of computed metrics.
    """
    # Convert Amount columns to numpy arrays for speed
    income_amounts = income_df['Amount'].to_numpy()
    expense_amounts = expenses_df['Amount'].to_numpy()
    
    total_income = np.sum(income_amounts)
    total_expenses = np.sum(expense_amounts)
    net_savings = total_income - total_expenses
    
    avg_daily_expense = np.mean(expenses_df.groupby('Date')['Amount'].sum())

    # Category-wise aggregates using numpy (via groupby mean and sum)
    cat_groups = expenses_df.groupby('Category')['Amount']
    cat_sum = cat_groups.sum()
    cat_avg = cat_groups.mean()
    
    # Monthly aggregates (already grouped)
    monthly_groups = expenses_df.groupby('Month')['Amount'].sum()
    avg_monthly_expense = np.mean(monthly_groups)
    
    results = {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'net_savings': net_savings,
        'avg_daily_expense': avg_daily_expense,
        'avg_monthly_expense': avg_monthly_expense,
        'category_sum': cat_sum,
        'category_avg': cat_avg,
        'monthly_expenses': monthly_groups
    }
    return results

def print_summary(results):
    print("="*50)
    print("FINANCIAL SUMMARY")
    print("="*50)
    print(f"Total Income:       PKR {results['total_income']:,.2f}")
    print(f"Total Expenses:     PKR {results['total_expenses']:,.2f}")
    print(f"Net Savings:        PKR {results['net_savings']:,.2f}")
    print(f"Avg Daily Spend:    PKR {results['avg_daily_expense']:.2f}")
    print(f"Avg Monthly Spend:  PKR {results['avg_monthly_expense']:.2f}")
    print("\nCategory-wise Spending:")
    print(results['category_sum'].sort_values(ascending=False))
    print("\nMonthly Spending Trend:")
    print(results['monthly_expenses'])

if __name__ == "__main__":
    df, expenses, income = load_and_clean("pakistan_statement_30k_enhanced.csv")
    summary = compute_summary(expenses, income)
    print_summary(summary)