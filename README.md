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
