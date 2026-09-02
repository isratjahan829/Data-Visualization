# Lab Assignment 2 — Data Cleaning, Integration, and Exploration

PGDDS 103: Data Analysis & Data Visualizations — FreshCart Grocers churn case.

## Files
- `FreshCart_Lab2_Data_Cleaning_Integration_Exploration.ipynb` — the solution notebook, already executed (all outputs saved). Runs as-is in Google Colab.
- `data/freshcart_users.csv` — shopper demographics, plan tier, tenure, total orders, churn status
- `data/freshcart_refunds.csv` — logs of orders with missing/damaged items
- `freshcart_clean.csv` — the cleaned, merged dataset produced by the notebook
- `chart_1_churn_averages.png` … `chart_4_items_vs_order_rate.png` — the charts the notebook renders

## Running in Google Colab
1. Open the notebook in Colab (File → Upload notebook, or open it from GitHub).
2. Upload `freshcart_users.csv` and `freshcart_refunds.csv` into the Colab file panel (`/content/`).
   The setup cell finds them automatically; if they are missing it writes the assignment's sample data
   so the notebook still runs end to end.
3. Runtime → Run all.

## Steps covered
1. Cleaning — drop duplicate rows, median-impute `Age`
2. Anomaly correction — negative `Total_Orders` → 0; `Months_Active == 0` with `Total_Orders > 0` → 1
3. Variable creation — `Monthly_Order_Rate`, `Tenure_Cohort` (0-6 / 7-12 / 13+ months)
4. Integration — left join the refunds log on `ShopperID`, fill `Missing_Items_Reported` NaN with 0
5. Exploration — averages by `Churn`, cross-tabulation of `Tenure_Cohort` vs `Churn`
6. Validation — assertions for zero missing values and zero duplicate `ShopperID` entries
7. Visualizations — churn averages, tenure cohort vs churn, per-shopper order rate, missing items vs order rate

---

# Ecommerce Purchases Exercise (Cust_Purch_FakeData)

A 30,000-row fake customer-purchase dataset with 20 columns. The exercise asks 20
questions about the customers; the notebook answers each one with pandas.

## Files
- `Cust_Purch_Data_Exercise_Solutions.ipynb` — solution notebook, already executed (all outputs saved)
- `data/Cust_Purch_FakeData.csv` — the dataset
- `chart_customers_per_weekday.png`, `chart_spending_distribution.png` — the two bonus charts

## Running
Run the notebook from the repository root, or open it in Colab and upload
`Cust_Purch_FakeData.csv` next to it — the first cell looks in `data/`, the
working directory, and `/content/`.

## Answers at a glance
| # | Question | Answer |
|---|---|---|
| 3 | Entries / columns | 30,000 rows × 20 columns |
| 4 | Age max / min / mean | 65 / 18 / 41.55 |
| 5 | Three most common first names | Willie (130), Francis (124), Eula (86) |
| 6 | Shared phone number | (263) 382-8004 — Lilly Tyler & Peter Cain |
| 7 | Structural Engineers | 87 |
| 8 | Male Structural Engineers | 43 |
| 9 | Female Structural Engineers in AB | 4 |
| 10 | Spending max / min / mean | 100.00 / 0.00 / 49.99 CAD |
| 11 | Spent nothing | 2 customers (Bruce Bryan, Flora Clark) |
| 12 | Spent 100 CAD or more | 3 customers |
| 13 | Emails on card 5020000000000230 | 2 |
| 14 | Cards expiring in 2019 | 2,684 |
| 15 | Visa users | 1,721 |
| 16 | Spent 100 CAD on Visa | Gregory Brown |
| 17 | Two most common professions | Preschool Teacher (112), Distribution Manager (107) |
| 18 | Top 5 email providers | gmail.com, me.com, outlook.com, live.com, hotmail.com |
| 19 | "am.edu" email | Yes — Loretta Fletcher |
| 20 | Busiest weekday | Saturday (4,376 customers) |
