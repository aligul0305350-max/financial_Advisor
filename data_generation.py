import random
import pandas as pd
from datetime import datetime, timedelta
import calendar

# Pakistani merchants
merchants = {
    'Food': [
        'Foodpanda', 'Eat Mubarak', 'Cheezious', 'KFC Pakistan', 'McDonalds Pakistan',
        'Butt Karahi', 'Gourmet Bakery', 'Lal Qila Restaurant', 'Student Biryani', 'Local Dhaba'
    ],
    'Travel': [
        'Careem', 'Bykea', 'Uber Pakistan', 'Metro Bus Recharge', 'Petrol Pump PSO', 'Pakistan Railways'
    ],
    'Shopping': [
        'Daraz.pk', 'Al-Fatah', 'Imtiaz Super Market', 'Hyperstar', 'Shophive', 'J.', 'Khaadi',
        'Utility Stores Corporation'
    ],
    'Entertainment': [
        'Cinepax', 'Universal Cinemas', 'Netflix Pakistan', 'Spotify Pakistan', 'BookMe'
    ],
    'Utilities': [
        'KE Electric', 'IESCO', 'LESCO', 'SNGPL Gas', 'PTCL Broadband', 'Jazz Recharge',
        'Telenor Recharge', 'Zong Recharge', 'Ufone Bill'
    ],
    'Rent': ['Monthly Rent Transfer', 'Rent Payment'],
    'Healthcare': [
        'Shifa International Hospital', 'Aga Khan Hospital', 'Sehat Kahani', 'Oladoc',
        'D.Watson Pharmacy', 'Servaid Pharmacy'
    ],
    'EMI': [
        'HBL Car Loan EMI', 'UBL Personal Loan EMI', 'Meezan Bank EMI', 'Faysal Bank EMI'
    ],
    'Investment': [
        'PSX Stock Purchase', 'KASB Securities', 'JS Global', 'Mutual Fund SIP', 'National Savings'
    ],
    'Others': [
        'ATM Cash Withdrawal', 'Easypaisa Transfer', 'JazzCash Transfer', '1Link Transfer',
        'Branch Deposit', 'Bank Charges'
    ]
}

salary_day = 1
salary_amount = 30000
end_date = datetime.today().date()
start_date = end_date - timedelta(days=365)

transactions = []

# Salary credit for each month
for month_offset in range(12):
    month = start_date.month + month_offset
    year = start_date.year + ((start_date.month + month_offset - 1) // 12)
    if month > 12:
        month -= 12
        year += 1
    last_day = calendar.monthrange(year, month)[1]
    pay_date = datetime(year, month, min(salary_day, last_day)).date()
    if pay_date <= end_date:
        transactions.append({
            'Date': pay_date,
            'Description': 'SALARY CREDIT',
            'Amount': salary_amount,
            'Type': 'Credit',
            'Category': 'Salary'
        })

# Daily expenses
current_date = start_date
while current_date <= end_date:
    if current_date.day == salary_day:
        current_date += timedelta(days=1)
        continue

    num_txns = random.randint(2, 4)
    for _ in range(num_txns):
        category = random.choices(
            list(merchants.keys()),
            weights=[25, 12, 20, 5, 15, 5, 5, 5, 3, 5]
        )[0]
        merchant = random.choice(merchants[category])

        if category == 'Food':
            amount = round(random.uniform(100, 1000), 2)
        elif category == 'Travel':
            amount = round(random.uniform(50, 800), 2)
        elif category == 'Shopping':
            amount = round(random.uniform(200, 5000), 2)
        elif category == 'Entertainment':
            amount = round(random.uniform(100, 1500), 2)
        elif category == 'Utilities':
            amount = round(random.uniform(300, 4000), 2)
        elif category == 'Rent':
            amount = 8000
        elif category == 'Healthcare':
            amount = round(random.uniform(200, 2500), 2)
        elif category == 'EMI':
            amount = random.choice([2000, 3000, 5000])
        elif category == 'Investment':
            amount = round(random.uniform(500, 5000), 2)
        else:
            amount = round(random.uniform(100, 2000), 2)

        transactions.append({
            'Date': current_date,
            'Description': merchant,
            'Amount': amount,
            'Type': 'Debit',
            'Category': category
        })

    current_date += timedelta(days=1)

# ============================================================
# Pehle DataFrame banao, phir usme anomalies add karo
# ============================================================
df = pd.DataFrame(transactions)
df = df.sort_values(['Date']).reset_index(drop=True)

# Pick 4 random unique dates from existing transactions
all_dates = df['Date'].unique()
anomaly_dates = random.sample(list(all_dates), k=4)

# Add anomalies
anomaly_rows = []
for i, adate in enumerate(anomaly_dates):
    anomaly_rows.append({
        'Date': adate,
        'Description': f'UNUSUAL TXN {i+1}',
        'Amount': random.choice([15000, 25000, 40000]),
        'Type': 'Debit',
        'Category': 'Others'  # Will be detected as anomaly later
    })

# Append anomalies to existing DataFrame
df = pd.concat([df, pd.DataFrame(anomaly_rows)], ignore_index=True)
df = df.sort_values(['Date']).reset_index(drop=True)

# Save final CSV
df.to_csv('pakistan_statement_30k_enhanced.csv', index=False)

print("✅ Enhanced dataset generated successfully!")
print(f"Total transactions: {len(df)}")
total_credit = df[df['Type']=='Credit']['Amount'].sum()
total_debit = df[df['Type']=='Debit']['Amount'].sum()
print(f"Total salary credits: {total_credit} PKR")
print(f"Total debits: {total_debit:.2f} PKR")
print(f"Anomalies added: {df[df['Description'].str.startswith('UNUSUAL')].shape[0]}")
print(df.head())