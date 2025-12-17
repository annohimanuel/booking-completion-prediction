# Drivers of Booking Completion (Airbnb-style Conversion Prediction)

Predict whether a customer will **complete a booking** using behavioral and trip context features. This project builds an end-to-end classification workflow (EDA → statistical testing → feature engineering → model training → tuning → explainability) to support **conversion targeting** (ads, perks, follow-ups).

---

## Problem Statement

Online travel/booking platforms lose revenue when users abandon bookings.  
Goal: **predict booking completion** (`booking_complete`) so marketing and product teams can prioritize high-intent customers.

**Target**
- `booking_complete` (binary)
  - `1` = booking completed
  - `0` = not completed

**Dataset**
- 50,000 rows × 14 columns
- No missing values in provided dataset

---

## Key Questions This Project Answers

### Business Questions
- Which customers are most likely to complete a booking so we can **target incentives efficiently**?
- What are the **strongest behavioral drivers** of completion (timing, route popularity, preferences)?
- How should we handle **class imbalance** (completion is minority class) so the model doesn’t ignore converters?

### Analytical Questions
- Which variables show statistically significant relationships with booking completion?
- Which modeling approach gives the best trade-off between **precision vs recall**, given conversion targeting needs?
- Which features drive predictions globally and locally (model interpretability)?

---

## Workflow

### 1) Exploratory Data Analysis (EDA)
- Checked data shape, dtypes, missingness
- Summary statistics
- Analyzed conversion rates by:
  - `trip_type`
  - `flight_day`
- Identified class imbalance (~4x more non-completions than completions)

### 2) Statistical Association Testing
- Numeric vs binary target: **Point-biserial correlation**
  - Dropped `flight_hour` due to non-significant relationship (p > 0.05)
- Categorical vs target: **Chi-square tests**
  - Confirmed categorical variables have significant relationships with completion

### 3) Feature Engineering
High-cardinality features:
- `route` (799 unique values)
- `booking_origin` (104 unique values)

Approach:
- Frequency encoding:
  - `route_freq`
  - `booking_origin_freq`
- One-hot encoding for low-cardinality categoricals:
  - `sales_channel`, `trip_type`, `flight_day`

### 4) Modeling
Train/test split:
- 80/20, stratified on `booking_complete`

Models trained:
- Logistic Regression (`class_weight="balanced"`)
- Random Forest (`class_weight="balanced"`)
- XGBoost with `scale_pos_weight` to address class imbalance

Primary evaluation focus:
- **F1 score** (balances precision and recall)
- Also reported accuracy, precision, recall

### 5) Hyperparameter Tuning
- GridSearchCV on XGBoost (5-fold CV, scoring = F1)
- Tuned:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`

### 6) Explainability
- Feature importance (XGBoost)
- SHAP summary plot for global drivers of conversion

Top drivers identified:
- `route_freq`, `purchase_lead`, `length_of_stay`
- Preference indicators (extra baggage, preferred seat, meals) contributed modestly

### 7) Model Export
- Saved best tuned model:
  - `xgb_booking_model.pkl`

---

## Results (Test Set)

Baseline models:
- Logistic Regression: F1 ≈ **0.340**
- Random Forest: F1 ≈ **0.188** (high accuracy but poor recall due to imbalance)
- XGBoost (baseline): F1 ≈ **0.440** (best)

Tuned XGBoost:
- F1 ≈ **0.434**
- Recall ≈ **0.722**
- Precision ≈ **0.310**

Note: Hyperparameter tuning slightly reduced F1 compared to baseline XGBoost, indicating baseline settings were already near-optimal for this dataset and metric.

---

Author : Imanuel Annoh
