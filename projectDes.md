# 🧰 Libraries & Tools (Tech Stack)

| Tool / Library | Kyun Use Hoga |
|----------------|----------------|
| **Python 3.9+** | Poori programming ke liye |
| **pandas** | Bank CSV ko load, clean, group, aggregate karne ke liye |
| **numpy** | Numerical operations, moving averages, threshold calculations |
| **matplotlib** | Pie chart, line graph, bar chart – saari visualizations |
| **seaborn (optional)** | Agar thoda stylish plots chahiye toh |
| **scikit-learn** | 4 alag kaam: |
| — `TfidfVectorizer` | Transaction descriptions ko text features mein badalne |
| — `RandomForestClassifier` | Auto-categorize transactions (Food, Rent, etc.) |
| — `IsolationForest` | Fraud / abnormal transactions detect karne |
| — `RandomForestRegressor` ya `LinearRegression` | Agle mahine ki spending predict karne |
| **Groq Python SDK** | `pip install groq` → LLM call karna (llama3-70b-8192 model) |
| **Streamlit (optional)** | Frontend banane ke liye taaki log upload karein aur result dekhein |
| **datetime** | Date parsing, month extraction |
| **Jupyter Notebook / VS Code** | Code likhne aur test karne ke liye |

---

# Groq API Key

* *******************34

# Libraries Useage in this Project:

- **pip install pandas**
- **pip install numpy**
- **pip install matplotlib**
- **pip install faker**
- **pip install scikit-learn**
- **pip install groq**
- **pip install python-dotenv**
- **pip install plotly**

# how to run this project after installing all the packages:
1. `streamlit run app.py`

# Project Description:

**Project Summary: Personal Finance Categorization & Analysis (Pakistan PKR Bank Statement)**

This project analyzes a bank statement CSV containing debit/credit transactions with descriptions, amounts, and categories. It loads data, computes financial summaries, generates plots, and trains an ML model to automatically categorize new transactions by merchant description.

---

### 1. Data Format (CSV)
- **Columns:** `Date`, `Description`, `Amount`, `Type`, `Category`
- **`Date`:** Transaction date (e.g., `2025-07-01`)
- **`Description`:** Merchant or transaction label (e.g., `SALARY CREDIT`, `Imtiaz Super Market`)
- **`Amount`:** Numeric value (positive for all)
- **`Type`:** `Credit` (income) or `Debit` (expense)
- **`Category`:** Pre-assigned category (e.g., `Salary`, `Shopping`); used as target for ML model.

---

### 2. File‑by‑File Breakdown

#### **`load_data.py` – Data Loading & Preprocessing**
- Function `load_and_clean(filepath)`  
  - Reads CSV using pandas  
  - Converts `Date` to datetime, sorts by date, resets index  
  - Separates DataFrame into `expenses` (Type == 'Debit') and `income` (Type == 'Credit')  
  - Adds a `Month` column (period like `2025-07`) for monthly grouping  
  - Returns three DataFrames: `df` (full), `expenses`, `income`

#### **`eda_numpy.py` – Numerical Summary & Aggregations**
- Function `compute_summary(expenses_df, income_df)`  
  - Uses **NumPy** arrays for fast aggregation  
  - Computes:  
    - Total income, total expenses, net savings  
    - Average daily spend (mean of daily totals)  
    - Average monthly expense  
    - Category‑wise total and average spending  
    - Monthly expense totals  
  - Returns a dictionary with all metrics  
- Function `print_summary(results)` prints the summary in a formatted way.

#### **`plot_matplotlib.py` – Visualisations**
- Function `save_all_plots(expenses, summary)`  
  Generates and saves three PNG charts:
  1. **Monthly Spending Trend** – Line chart of `summary['monthly_expenses']`
  2. **Category Donut Chart** – Distribution of expenses by category (pie with centre hole)
  3. **Top 10 Frequent Merchants** – Horizontal bar chart of most recurring descriptions (by transaction count, not amount)

#### **`ml_classifier.py` – Machine Learning Categorisation**
- **Goal:** Predict `Category` from `Description` for expense transactions.
- **Pipeline:** TF‑IDF vectorizer + RandomForestClassifier (using scikit‑learn)
- Key functions:  
  - `preprocess_text(series)` – lowercases, strips, removes extra spaces  
  - `build_pipeline()` – creates scikit‑learn `Pipeline`  
  - `train_category_classifier(df, tune_hyperparams=False)`  
    - Uses only expense rows (Type == 'Debit')  
    - Splits data into train/test (80/20, stratified)  
    - Optionally performs GridSearchCV over hyperparameters  
    - Trains the pipeline, evaluates accuracy, prints classification report and 5‑fold cross‑validation score  
    - Returns the trained pipeline and accuracy  
  - `predict_category(description, pipeline)` – returns predicted category and confidence (probability) for a single new description.

---

### 3. Dependencies (Python Libraries)
- `pandas`, `numpy`, `matplotlib`  
- `scikit-learn` (sklearn): `TfidfVectorizer`, `RandomForestClassifier`, `train_test_split`, `cross_val_score`, `GridSearchCV`, `Pipeline`, `classification_report`, `accuracy_score`  
- `joblib` (imported but not used in snippet, likely for later model persistence)

---

### 4. Execution Flow (if all scripts are run in sequence)
1. **Load & Clean** (`load_data.py`) → `df`, `expenses`, `income`  
2. **Compute Summary** (`eda_numpy.py`) → `summary` dict  
3. **Generate Plots** (`plot_matplotlib.py`) → three PNG files  
4. **Train ML Model** (`ml_classifier.py`) → pipeline ready to predict categories for new merchants like "KFC Phase 6 DHA" → "Food" etc.

Each file is also self‑contained with a `if __name__ == "__main__":` block that runs the complete pipeline using a hardcoded CSV name `"pakistan_statement_30k_enhanced.csv"`.

---

### 5. Key Outputs
- Console: total income/expenses, net savings, average daily/monthly spend, category‑wise spending, monthly trend, ML accuracy and classification report, predictions for sample new descriptions.
- Saved files: `monthly_spending_trend.png`, `category_donut.png`, `top_merchants.png`

---

### 6. Example Usage (for context)
The data includes entries like `SALARY CREDIT` (income) and `Imtiaz Super Market` (expense, Shopping). The ML model learns to map merchant names like `KFC Phase 6 DHA` → `Food`, `Shell Petrol Pump` → `Transport`, `Careem Ride` → `Transport`, etc., achieving around 90‑95%+ accuracy on test data.