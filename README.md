# 🛒 E-Commerce Customer Churn Analysis & Prediction

> **Senior Data Science Project** — End-to-end ML pipeline with an interactive Streamlit Business Intelligence dashboard.

---

## 📋 Project Overview

Customer churn is one of the most critical business metrics in e-commerce. This project delivers a **full data science workflow** — from raw CSV ingestion and exploratory data analysis through to a trained Random Forest classifier and an interactive, multi-page Streamlit dashboard for business stakeholders.

### Key Capabilities
| Capability | Description |
|---|---|
| **Data Pipeline** | Automated CSV ingestion, missing-value imputation, type coercion |
| **EDA** | Distribution analysis, segment profiling, churn driver identification |
| **ML Model** | Random Forest Classifier (200 trees, balanced class weights) |
| **Business Dashboard** | 4-page Streamlit app with Plotly visualisations |
| **Live Predictor** | Real-time churn probability from user-input customer attributes |

---

## 📁 Project Structure

```
E-Commerce-Churn-Project/
│
├── SwapnilSinghaBiswas_ECommerce-Customer-Churn-Analysis.py                     # Main application (pipeline + dashboard)
├── requirements.txt                                                             # Python dependency versions
├── README.md                                                                    # This file
├── SwapnilSinghaBiswas_ProjectReport.docx                                       # Full project report (Word document)
└── E-commerce_Customer_Churn_2026.csv                                           # Source dataset
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
python -m streamlit run SwapnilSinghaBiswas_ECommerce-Customer-Churn-Analysis.py
```
The app will open automatically in your default browser at `http://localhost:8501`.

Backend / Architecture: Single-script Streamlit execution (SwapnilSinghaBiswas_ECommerce-Customer-Churn.py). Predictions are calculated natively in-memory via Scikit-Learn without requiring an external REST API endpoint.

---

## 📊 Dashboard Pages

### Page 1 — Executive Overview
- **Top 5 KPIs**: Overall Churn Rate %, Average Customer LTV, Average Satisfaction Score, Average Churn Risk Score, Revenue at Risk
- Interactive filters: Customer Segment & Country
- Churn rate breakdowns by segment, country, subscription type, and engagement level

### Page 2 — Drivers & Risk Analysis
- Top 10 Churn Reasons (Funnel chart)
- Churn Category distribution
- Support Tickets vs. Churn (Box plot + binned bar chart)
- Satisfaction Level impact on churn (line chart + scatter)
- Random Forest Feature Importance ranking
- Model Performance: Accuracy, ROC-AUC, Precision, Recall, Confusion Matrix
- Total Spend & LTV distribution by churn status

### Page 3 — Strategic Action Plan
- High / Medium / Low risk segment snapshot with LTV
- Churn Risk Score distribution histogram
- Prevention Method effectiveness (retention rate by method)
- 6 data-driven business retention strategies
- Estimated Retention ROI metrics

### Page 4 — Live Churn Predictor
- Interactive form: 20+ customer attributes
- Real-time Random Forest prediction
- Probability gauge chart
- Colour-coded result card (High Risk / Low Risk)
- Personalised recommended action list based on risk tier

---

## 🤖 Machine Learning

### Model: Random Forest Classifier
| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 12 |
| `min_samples_leaf` | 5 |
| `class_weight` | balanced |
| `test_size` | 20% |
| `random_state` | 42 |

### Key Features Used
- `churn_risk_score`, `churn_probability`, `retention_probability`
- `satisfaction_score`, `customer_health_score`
- `support_tickets`, `returns_count`
- `total_spend_usd`, `customer_lifetime_value_usd`, `avg_order_value_usd`
- `days_since_last_purchase`, `customer_age_months`, `loyalty_score`
- `feedback_score`, `referral_count`, `marketing_conversions`
- Encoded: `subscription_type`, `customer_segment`, `engagement_level`, `social_media_engagement`

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
streamlit
plotly
python-docx
flask
joblib
requests
matplotlib
seaborn
```

---

## 📂 Dataset

**Source**: [E-Commerce Customer Churn Analysis and Prediction — Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

**File**: `E-commerce_Customer_Churn_2026.csv`

**Key Columns**:
| Column | Description |
|---|---|
| `churn_flag` | Binary target (1 = Churned, 0 = Retained) |
| `customer_segment` | Consumer / Small Business / Enterprise / Premium |
| `churn_risk_score` | Composite risk score 0–100 |
| `satisfaction_score` | Customer satisfaction 1–5 |
| `support_tickets` | Number of support tickets raised |
| `total_spend_usd` | Cumulative customer spend |
| `customer_lifetime_value_usd` | Predicted or actual LTV |
| `days_since_last_purchase` | Recency metric |
| `churn_reason` | Textual reason for churning |
| `prevention_method` | Retention action applied |

---

## 📄 License

This project is for educational and analytical purposes. Dataset credit: Ankit Verma on Kaggle.

---

*Built with Python · Streamlit · Plotly · scikit-learn*
