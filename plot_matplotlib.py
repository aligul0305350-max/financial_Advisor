import matplotlib.pyplot as plt
import pandas as pd
from load_data import load_and_clean
from eda_numpy import compute_summary

def save_all_plots(expenses, summary):
    """
    Creates and saves 3 plots:
    1. Monthly spending trend (line)
    2. Category distribution (donut)
    3. Top 10 frequent merchants (horizontal bar)
    """
    # 1. Monthly spending line chart
    monthly = summary['monthly_expenses']
    plt.figure(figsize=(10,5))
    monthly.plot(kind='line', marker='o', color='crimson', linewidth=2)
    plt.title('Monthly Spending Trend', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Amount (PKR)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('monthly_spending_trend.png')
    plt.close()
    print("Saved: monthly_spending_trend.png")

    # 2. Category donut chart
    cat_sum = summary['category_sum'].sort_values(ascending=False)
    plt.figure(figsize=(8,8))
    plt.pie(cat_sum, labels=cat_sum.index, autopct='%1.1f%%', startangle=140, 
            pctdistance=0.85, wedgeprops={'edgecolor':'white'})
    # Draw donut hole
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)
    plt.title('Expense Distribution by Category', fontsize=14)
    plt.tight_layout()
    plt.savefig('category_donut.png')
    plt.close()
    print("Saved: category_donut.png")

    # 3. Top 10 merchants bar chart
    top_merchants = expenses['Description'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    plt.barh(top_merchants.index, top_merchants.values, color='teal')
    plt.title('Top 10 Most Frequent Merchants', fontsize=14)
    plt.xlabel('Number of Transactions')
    plt.gca().invert_yaxis()  # highest at top
    plt.tight_layout()
    plt.savefig('top_merchants.png')
    plt.close()
    print("Saved: top_merchants.png")

if __name__ == "__main__":
    df, expenses, income = load_and_clean("pakistan_statement_30k_enhanced.csv")
    summary = compute_summary(expenses, income)
    save_all_plots(expenses, summary)
    print("All charts saved successfully!")