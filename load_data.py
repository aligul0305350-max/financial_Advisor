import pandas as pd

def load_and_clean(filepath):
    # Load CSV
    # df = pd.read_csv(filepath)
    df = pd.read_csv(
        filepath,
        on_bad_lines='skip',   # kharab lines ignore karega
        quoting=1,             # csv.QUOTE_ALL
        encoding='utf-8'
    )
    
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date (optional, already sorted)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Separate into expenses (Debit) and income (Credit)
    expenses = df[df['Type'] == 'Debit'].copy()
    income = df[df['Type'] == 'Credit'].copy()
    
    # Add a month period for grouping (e.g., 2024-01)
    df['Month'] = df['Date'].dt.to_period('M')
    expenses['Month'] = expenses['Date'].dt.to_period('M')
    income['Month'] = income['Date'].dt.to_period('M')
    
    print("Data loaded successfully!")
    print(f"Total rows: {len(df)}, Expenses: {len(expenses)}, Income: {len(income)}")
    return df, expenses, income

if __name__ == "__main__":
    df, exp, inc = load_and_clean("pakistan_statement_30k_enhanced.csv")
    print(df.head())