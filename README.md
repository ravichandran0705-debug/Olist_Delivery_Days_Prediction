# Olist Delivery Time Prediction

Data profiling and machine learning project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), predicting how many days a delivery will take **using only information known at the time of purchase**.

## Contents

| File | Description |
|---|---|
| [`data_audit.ipynb`](./data_audit.ipynb) | Systematic data profiling and quality assessment of all 9 raw Olist tables |
| [`delivery_days_prediction.ipynb`](./delivery_days_prediction.ipynb) | Feature engineering, EDA, and model comparison to predict delivery time |

## 1. Data Audit

Profiles the `orders`, `customers`, `order_items`, `products`, `sellers`, `order_payments`, `order_reviews`, `geolocation`, and `category_translation` tables. For each table it checks:

- Structure (rows, columns, dtypes)
- Missing values and duplicates
- Candidate primary keys and uniqueness
- Descriptive/categorical statistics
- Logical consistency (e.g. date ordering, valid ranges)
- Cross-table relationship coverage (foreign key matches)

**Key findings:**
- `orders`: 99,441 rows; `order_id` and `customer_id` are unique; ~97% of orders are `delivered`; 8 delivered orders are missing a delivery timestamp.
- `customers`: 99,441 rows, no missing values or duplicates; `customer_unique_id` distinguishes repeat customers from per-order `customer_id`s.
- `order_items`: 112,650 rows; composite key is (`order_id`, `order_item_id`); covers 98,666 orders.
- `products`: 32,951 rows; ~1.85% missing category/description metadata; a small number missing physical dimensions.
- `sellers`: 3,095 rows, no missing values, clean primary key.
- `order_payments`: 103,886 rows; ~74% of payments are by credit card.
- `order_reviews`: 99,224 rows; review title/message are frequently missing (~88% / ~59%).
- `geolocation`: ~1M rows keyed by ZIP-code prefix, with a meaningful number of exact duplicates and a few coordinate outliers.
- `category_translation`: 71 Portuguese→English category mappings, no duplicates.

These findings drove the cleaning and feature-engineering decisions in the modeling notebook.

## 2. Delivery Time Prediction

**Goal:** Predict `delivery_days` (time between order purchase and delivery to the customer) using only features available at or before order placement.

**Approach:**
- Engineer features at the order, product, seller, customer, payment, and geolocation levels (item counts, price/freight aggregates, product weight/volume, seller–customer haversine distance and same-state flag, payment type/installments, purchase-time features).
- Explicitly exclude any timestamp or status field only known *after* delivery (`order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date`, `order_status`) to avoid data leakage.
- Split data 70% train / 15% cross-validation / 15% test.
- Drop highly correlated features (|r| ≥ 0.7) after EDA.
- Compare regularized linear models (Ridge, Lasso, Elastic Net) against tree-based ensembles (Random Forest, XGBoost) to test whether the relationship between features and delivery time is linear.

**Results (cross-validation set):**

| Model | MAE (days) | RMSE (days) | R² |
|---|---|---|---|
| OLS | unstable (ill-conditioned) | — | — |
| Ridge | 5.09 | 7.79 | 0.269 |
| Lasso | 5.09 | 7.79 | 0.268 |
| Elastic Net | 5.09 | 7.79 | 0.269 |
| Random Forest | 4.57 | 7.16 | 0.382 |
| **XGBoost** | **4.49** | **7.14** | **0.387** |

XGBoost was the best-performing model and generalized reasonably well to the held-out test set (MAE 4.51, RMSE 7.54, R² 0.369), confirming a non-linear relationship between the engineered features and delivery time that tree-based models capture better than linear ones.

## Data

The notebooks expect the raw Olist CSVs (`olist_customers_dataset.csv`, `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `olist_geolocation_dataset.csv`, `product_category_name_translation.csv`), available from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Update the file paths at the top of each notebook to point to your local copy of the data before running.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, seaborn/matplotlib.
