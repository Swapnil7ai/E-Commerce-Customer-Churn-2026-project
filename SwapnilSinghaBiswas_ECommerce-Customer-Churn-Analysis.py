"""
E-Commerce Customer Churn Analysis & Prediction
================================================
Senior Data Science project — Single-file Streamlit application.
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Customer Churn Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e8eaf0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12151f 100%);
        border-right: 1px solid #2a2d3e;
    }
    section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

    /* KPI metric cards */
    .kpi-card {
        background: #1e2130;
        border: 1px solid #2e3150;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #8892a4;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 40px;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-sub { font-size: 12px; color: #6b7485; margin-top: 6px; }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #a5b4fc;
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Page title */
    .page-title {
        font-size: 32px;
        font-weight: 800;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 28px;
    }

    /* Strategy cards */
    .strategy-card {
        background: #1a1f35;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .strategy-title {
        font-size: 15px;
        font-weight: 700;
        color: #a5b4fc;
        margin-bottom: 6px;
    }
    .strategy-body { font-size: 13px; color: #94a3b8; line-height: 1.6; }

    /* Prediction result */
    .pred-high {
        background: #3b0a0a;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .pred-low {
        background: #052e16;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .pred-label { font-size: 14px; color: #94a3b8; }
    .pred-value { font-size: 48px; font-weight: 900; }

    /* Divider */
    hr { border-color: #2a2d3e; }

    /* Streamlit element overrides */
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3150;
        border-radius: 10px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_data():
    df = pd.read_csv("E-commerce_Customer_Churn_2026.csv")

    # Fill missing / 'None' string values
    df["subscription_type"] = df["subscription_type"].replace("None", np.nan).fillna("Unknown")
    df["prevention_method"] = df["prevention_method"].replace("None", np.nan).fillna("Unknown")
    df["social_media_engagement"] = df["social_media_engagement"].replace("None", np.nan).fillna("Unknown")
    df["churn_reason"] = df["churn_reason"].fillna("Unknown")
    df["churn_category"] = df["churn_category"].fillna("Unknown")
    df["last_support_channel"] = df["last_support_channel"].fillna("Unknown")

    # Ensure numeric
    numeric_cols = [
        "churn_risk_score", "churn_probability", "satisfaction_score",
        "support_tickets", "total_spend_usd", "days_since_last_purchase",
        "customer_lifetime_value_usd", "avg_order_value_usd", "num_purchases",
        "customer_age_months", "returns_count", "loyalty_score",
        "feedback_score", "referral_count", "marketing_conversions",
        "marketing_emails_sent", "marketing_emails_opened",
        "marketing_emails_clicked", "support_interactions",
        "customer_retention_cost_usd", "recovery_revenue_usd",
        "churn_revenue_impact_usd", "customer_health_score",
        "retention_probability"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Binary target
    df["churn_flag"] = df["churn_flag"].astype(int)
    return df

# ─────────────────────────────────────────────
# MACHINE LEARNING PIPELINE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Random Forest model…")
def train_model(df: pd.DataFrame):
    FEATURES = [
        "churn_risk_score", "satisfaction_score", "support_tickets",
        "total_spend_usd", "days_since_last_purchase",
        "customer_lifetime_value_usd", "avg_order_value_usd",
        "num_purchases", "customer_age_months", "returns_count",
        "loyalty_score", "feedback_score", "referral_count",
        "marketing_conversions", "churn_probability",
        "customer_health_score", "retention_probability"
    ]
    # Encode categorical helpers
    cat_cols = ["subscription_type", "customer_segment", "engagement_level",
                "social_media_engagement"]
    encoders = {}
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
        FEATURES.append(col + "_enc")

    X = df_enc[FEATURES].fillna(0)
    y = df_enc["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_proba) * 100, 2),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "feature_names": FEATURES,
        "feature_importances": clf.feature_importances_,
    }
    return clf, encoders, FEATURES, metrics

# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_churn(clf, encoders, feature_names, input_dict: dict):
    df_input = pd.DataFrame([input_dict])
    cat_cols = ["subscription_type", "customer_segment", "engagement_level",
                "social_media_engagement"]
    for col in cat_cols:
        le = encoders[col]
        val = str(input_dict.get(col, "Unknown"))
        if val in le.classes_:
            df_input[col + "_enc"] = le.transform([val])[0]
        else:
            df_input[col + "_enc"] = 0
    X = df_input[feature_names].fillna(0)
    proba = clf.predict_proba(X)[0][1]
    label = "High Risk" if proba >= 0.5 else "Low Risk"
    return round(proba * 100, 1), label

# ─────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL (cached)
# ─────────────────────────────────────────────
df = load_data()
clf, encoders, feature_names, metrics = train_model(df)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION & FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Churn Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Executive Overview",
         "🔍 Drivers & Risk Analysis",
         "🎯 Strategic Action Plan",
         "⚡ Live Churn Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Global Filters")
    segments = ["All"] + sorted(df["customer_segment"].dropna().unique().tolist())
    sel_segment = st.selectbox("Customer Segment", segments)
    countries = ["All"] + sorted(df["customer_country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", countries)
    st.markdown("---")
    st.markdown(
        f"**Model Accuracy:** `{metrics['accuracy']}%`  \n"
        f"**ROC-AUC Score:** `{metrics['roc_auc']}%`"
    )
    st.markdown("---")
    st.caption("Data: E-Commerce Churn 2026 · Model: Random Forest")

# Apply filters
dff = df.copy()
if sel_segment != "All":
    dff = dff[dff["customer_segment"] == sel_segment]
if sel_country != "All":
    dff = dff[dff["customer_country"] == sel_country]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1e2130",
    plot_bgcolor="#1e2130",
    font=dict(color="#c9d1e0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="#252840", bordercolor="#3a3d5e", borderwidth=1)
)
COLORS = px.colors.qualitative.Vivid

# ═══════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown('<div class="page-title">📊 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">High-level churn health metrics across your customer base</div>', unsafe_allow_html=True)

    # KPIs
    total = len(dff)
    churned = dff["churn_flag"].sum()
    churn_rate = round(churned / total * 100, 2) if total > 0 else 0
    avg_clv = round(dff["customer_lifetime_value_usd"].mean(), 2)
    avg_sat = round(dff["satisfaction_score"].mean(), 2)
    avg_risk = round(dff["churn_risk_score"].mean(), 1)
    total_rev_at_risk = round(dff[dff["churn_flag"] == 1]["churn_revenue_impact_usd"].sum(), 0)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        color = "#ef4444" if churn_rate > 30 else "#f59e0b" if churn_rate > 15 else "#22c55e"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Overall Churn Rate</div>
            <div class="kpi-value" style="color:{color}">{churn_rate}%</div>
            <div class="kpi-sub">{churned:,} of {total:,} customers</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Customer LTV</div>
            <div class="kpi-value" style="color:#60a5fa">${avg_clv:,.0f}</div>
            <div class="kpi-sub">Lifetime Value (USD)</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        sat_color = "#22c55e" if avg_sat >= 3.5 else "#f59e0b" if avg_sat >= 2.5 else "#ef4444"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Satisfaction Score</div>
            <div class="kpi-value" style="color:{sat_color}">{avg_sat}/5</div>
            <div class="kpi-sub">Customer satisfaction index</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        risk_color = "#ef4444" if avg_risk > 50 else "#f59e0b" if avg_risk > 30 else "#22c55e"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Churn Risk Score</div>
            <div class="kpi-value" style="color:{risk_color}">{avg_risk}</div>
            <div class="kpi-sub">Scale 0–100</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Revenue at Risk</div>
            <div class="kpi-value" style="color:#fb923c">${total_rev_at_risk:,.0f}</div>
            <div class="kpi-sub">From churned customers</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Churn by Segment & Country
    st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        seg_data = (dff.groupby("customer_segment")["churn_flag"]
                    .agg(["sum", "count"])
                    .rename(columns={"sum": "Churned", "count": "Total"})
                    .reset_index())
        seg_data["Churn Rate %"] = (seg_data["Churned"] / seg_data["Total"] * 100).round(1)
        fig = px.bar(seg_data, x="customer_segment", y="Churn Rate %",
                     color="Churn Rate %", color_continuous_scale="RdYlGn_r",
                     title="Churn Rate by Customer Segment",
                     labels={"customer_segment": "Segment"},
                     text="Churn Rate %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_countries = (dff.groupby("customer_country")["churn_flag"]
                         .agg(["sum", "count"])
                         .rename(columns={"sum": "Churned", "count": "Total"})
                         .reset_index()
                         .assign(**{"Churn Rate %": lambda x: (x["Churned"] / x["Total"] * 100).round(1)})
                         .nlargest(12, "Total"))
        fig2 = px.bar(top_countries, x="Churn Rate %", y="customer_country",
                      orientation="h", color="Churn Rate %",
                      color_continuous_scale="RdYlBu_r",
                      title="Top Countries by Churn Rate",
                      labels={"customer_country": "Country"})
        fig2.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Subscription type & Churn trend
    c3, c4 = st.columns(2)

    with c3:
        sub_churn = (dff.groupby("subscription_type")["churn_flag"]
                     .agg(["sum", "count"])
                     .rename(columns={"sum": "Churned", "count": "Total"})
                     .assign(**{"Churn Rate %": lambda x: (x["Churned"] / x["Total"] * 100).round(1)})
                     .reset_index())
        fig3 = px.pie(sub_churn, names="subscription_type", values="Churned",
                      title="Churned Customers by Subscription Type",
                      color_discrete_sequence=COLORS, hole=0.45)
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        eng_churn = (dff.groupby(["engagement_level", "churn_flag"])
                     .size().reset_index(name="Count"))
        eng_churn["Churn Status"] = eng_churn["churn_flag"].map({0: "Retained", 1: "Churned"})
        fig4 = px.bar(eng_churn, x="engagement_level", y="Count",
                      color="Churn Status", barmode="group",
                      title="Engagement Level vs Churn",
                      color_discrete_map={"Churned": "#ef4444", "Retained": "#22c55e"})
        fig4.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════
# PAGE 2 — DRIVERS & RISK ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🔍 Drivers & Risk Analysis":
    st.markdown('<div class="page-title">🔍 Drivers & Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand why customers churn and who is at risk</div>', unsafe_allow_html=True)

    # ── Churn Reason Breakdown
    st.markdown('<div class="section-header">Churn Reason Analysis</div>', unsafe_allow_html=True)
    churned_df = dff[dff["churn_flag"] == 1]

    c1, c2 = st.columns([3, 2])
    with c1:
        reason_counts = (churned_df["churn_reason"]
                         .value_counts()
                         .reset_index()
                         .rename(columns={"index": "Reason", "churn_reason": "Count"}))
        reason_counts.columns = ["Reason", "Count"]
        fig = px.funnel(reason_counts.head(10), x="Count", y="Reason",
                        title="Top 10 Churn Reasons (Funnel)",
                        color_discrete_sequence=["#6366f1"])
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cat_counts = (churned_df["churn_category"]
                      .value_counts()
                      .reset_index())
        cat_counts.columns = ["Category", "Count"]
        fig2 = px.pie(cat_counts, names="Category", values="Count",
                      title="Churn Category Distribution",
                      color_discrete_sequence=COLORS, hole=0.4)
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Support Tickets vs Churn
    st.markdown('<div class="section-header">Support Tickets vs. Churn</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.box(dff, x="churn_flag", y="support_tickets",
                      color="churn_flag",
                      labels={"churn_flag": "Churned (1=Yes)", "support_tickets": "Support Tickets"},
                      title="Support Tickets Distribution by Churn Status",
                      color_discrete_map={0: "#22c55e", 1: "#ef4444"})
        fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        fig3.update_xaxes(tickvals=[0, 1], ticktext=["Retained", "Churned"])
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        ticket_bins = pd.cut(dff["support_tickets"], bins=[0, 2, 5, 10, 20, 50],
                             labels=["0–2", "3–5", "6–10", "11–20", "20+"],
                             right=True)
        dff_copy = dff.copy()
        dff_copy["ticket_range"] = ticket_bins
        ticket_churn = (dff_copy.groupby("ticket_range")["churn_flag"]
                        .mean().reset_index()
                        .rename(columns={"churn_flag": "Churn Rate"}))
        ticket_churn["Churn Rate %"] = (ticket_churn["Churn Rate"] * 100).round(1)
        fig4 = px.bar(ticket_churn, x="ticket_range", y="Churn Rate %",
                      title="Churn Rate by Support Ticket Volume",
                      color="Churn Rate %", color_continuous_scale="Reds",
                      text="Churn Rate %")
        fig4.update_traces(texttemplate="%{text}%", textposition="outside")
        fig4.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                           xaxis_title="Ticket Volume Range")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Satisfaction Impact
    st.markdown('<div class="section-header">Satisfaction Level Impact</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        sat_churn = (dff.groupby("satisfaction_level")["churn_flag"]
                     .mean().reset_index()
                     .rename(columns={"churn_flag": "Churn Rate"}))
        sat_churn["Churn Rate %"] = (sat_churn["Churn Rate"] * 100).round(1)
        order_map = ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"]
        sat_churn["satisfaction_level"] = pd.Categorical(
            sat_churn["satisfaction_level"], categories=order_map, ordered=True
        )
        sat_churn = sat_churn.sort_values("satisfaction_level")
        fig5 = px.line(sat_churn, x="satisfaction_level", y="Churn Rate %",
                       markers=True, title="Churn Rate by Satisfaction Level",
                       color_discrete_sequence=["#f59e0b"])
        fig5.update_traces(line=dict(width=3), marker=dict(size=10))
        fig5.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        fig6 = px.scatter(
            dff.sample(min(1000, len(dff)), random_state=42),
            x="satisfaction_score", y="churn_risk_score",
            color=dff.sample(min(1000, len(dff)), random_state=42)["churn_flag"].map({0: "Retained", 1: "Churned"}),
            color_discrete_map={"Churned": "#ef4444", "Retained": "#22c55e"},
            title="Satisfaction Score vs. Churn Risk Score",
            opacity=0.7,
            labels={"satisfaction_score": "Satisfaction Score",
                    "churn_risk_score": "Churn Risk Score"}
        )
        fig6.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig6, use_container_width=True)

    # ── Feature Importance
    st.markdown('<div class="section-header">ML Model — Feature Importance</div>', unsafe_allow_html=True)
    fi_df = (pd.DataFrame({
        "Feature": metrics["feature_names"],
        "Importance": metrics["feature_importances"]
    }).sort_values("Importance", ascending=False).head(15))
    fig7 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                  title="Top 15 Churn Predictors (Random Forest)",
                  color="Importance", color_continuous_scale="Purples")
    fig7.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig7, use_container_width=True)

    # ── Model Performance
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    report = metrics["report"]
    m1.metric("Accuracy", f"{metrics['accuracy']}%")
    m2.metric("ROC-AUC Score", f"{metrics['roc_auc']}%")
    m3.metric("Precision (Churn)", f"{round(report['1']['precision']*100,1)}%")
    m4.metric("Recall (Churn)", f"{round(report['1']['recall']*100,1)}%")

    # Confusion matrix
    cm = metrics["conf_matrix"]
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Predicted", y="Actual"),
                       x=["Retained", "Churned"], y=["Retained", "Churned"],
                       title="Confusion Matrix", color_continuous_scale="Blues")
    fig_cm.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Spend & LTV analysis
    st.markdown('<div class="section-header">Spend & Lifetime Value Analysis</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)

    with c7:
        fig8 = px.histogram(dff, x="total_spend_usd", color=dff["churn_flag"].map({0: "Retained", 1: "Churned"}),
                            nbins=40, barmode="overlay", opacity=0.7,
                            title="Total Spend Distribution by Churn Status",
                            color_discrete_map={"Churned": "#ef4444", "Retained": "#22c55e"},
                            labels={"total_spend_usd": "Total Spend (USD)"})
        fig8.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig8, use_container_width=True)

    with c8:
        fig9 = px.violin(dff, y="customer_lifetime_value_usd",
                         x=dff["churn_flag"].map({0: "Retained", 1: "Churned"}),
                         color=dff["churn_flag"].map({0: "Retained", 1: "Churned"}),
                         box=True, points="outliers",
                         title="Customer LTV by Churn Status",
                         color_discrete_map={"Churned": "#ef4444", "Retained": "#22c55e"})
        fig9.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                           xaxis_title="Churn Status",
                           yaxis_title="Lifetime Value (USD)")
        st.plotly_chart(fig9, use_container_width=True)

# ═══════════════════════════════════════════════
# PAGE 3 — STRATEGIC ACTION PLAN
# ═══════════════════════════════════════════════
elif page == "🎯 Strategic Action Plan":
    st.markdown('<div class="page-title">🎯 Strategic Action Plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Data-driven retention strategies for high-risk customer segments</div>', unsafe_allow_html=True)

    # Risk segment summary
    st.markdown('<div class="section-header">High-Risk Segment Snapshot</div>', unsafe_allow_html=True)

    high_risk = dff[dff["churn_risk_score"] >= 70]
    med_risk = dff[(dff["churn_risk_score"] >= 40) & (dff["churn_risk_score"] < 70)]
    low_risk = dff[dff["churn_risk_score"] < 40]

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🔴 High Risk</div>
            <div class="kpi-value" style="color:#ef4444">{len(high_risk):,}</div>
            <div class="kpi-sub">Score ≥ 70 · Avg LTV: ${high_risk['customer_lifetime_value_usd'].mean():,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🟡 Medium Risk</div>
            <div class="kpi-value" style="color:#f59e0b">{len(med_risk):,}</div>
            <div class="kpi-sub">Score 40–69 · Avg LTV: ${med_risk['customer_lifetime_value_usd'].mean():,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🟢 Low Risk</div>
            <div class="kpi-value" style="color:#22c55e">{len(low_risk):,}</div>
            <div class="kpi-sub">Score < 40 · Avg LTV: ${low_risk['customer_lifetime_value_usd'].mean():,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk score distribution
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(dff, x="churn_risk_score", nbins=50,
                            color=dff["churn_flag"].map({0: "Retained", 1: "Churned"}),
                            barmode="overlay", opacity=0.8,
                            title="Churn Risk Score Distribution",
                            color_discrete_map={"Churned": "#ef4444", "Retained": "#22c55e"})
        fig1.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        risk_cat = (dff.groupby("churn_risk_category")["churn_flag"]
                    .agg(["sum", "count"])
                    .rename(columns={"sum": "Churned", "count": "Total"})
                    .assign(**{"Churn Rate %": lambda x: (x["Churned"] / x["Total"] * 100).round(1)})
                    .reset_index())
        fig2 = px.bar(risk_cat, x="churn_risk_category", y="Churn Rate %",
                      color="Churn Rate %", color_continuous_scale="Reds",
                      title="Churn Rate by Risk Category", text="Churn Rate %")
        fig2.update_traces(texttemplate="%{text}%", textposition="outside")
        fig2.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Prevention method effectiveness
    st.markdown('<div class="section-header">Prevention Method Effectiveness</div>', unsafe_allow_html=True)
    prev = (dff[dff["prevention_method"] != "Unknown"]
            .groupby("prevention_method")["churn_flag"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "Churned", "count": "Applied"})
            .assign(**{"Retention Rate %": lambda x: ((1 - x["Churned"] / x["Applied"]) * 100).round(1)})
            .sort_values("Retention Rate %", ascending=False)
            .reset_index())
    fig3 = px.bar(prev, x="prevention_method", y="Retention Rate %",
                  color="Retention Rate %", color_continuous_scale="Greens",
                  title="Retention Rate by Prevention Method Applied",
                  text="Retention Rate %")
    fig3.update_traces(texttemplate="%{text}%", textposition="outside")
    fig3.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                       xaxis_title="Prevention Method")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Strategic Recommendations
    st.markdown('<div class="section-header">Recommended Retention Strategies</div>', unsafe_allow_html=True)

    strategies = [
        {
            "icon": "🔴",
            "title": "High-Risk Segment: Immediate Intervention Program",
            "body": (
                "Target customers with churn_risk_score ≥ 70 with a dedicated win-back program. "
                "Deploy proactive outreach via phone or dedicated account managers within 48 hours. "
                "Offer personalised discounts (10–20%) or free service upgrades. "
                "Flag these accounts in CRM for weekly check-ins and assign a Customer Success partner."
            )
        },
        {
            "icon": "😞",
            "title": "Low Satisfaction Customers: Experience Recovery",
            "body": (
                "For customers with satisfaction_score ≤ 2 (Very Dissatisfied / Dissatisfied), "
                "trigger automated 'We're Sorry' campaigns with tangible resolution offers. "
                "Deploy NPS micro-surveys post-resolution to measure recovery. "
                "Implement 24-hour SLA commitments for this cohort's support tickets."
            )
        },
        {
            "icon": "📞",
            "title": "High Support-Ticket Customers: Proactive Support Escalation",
            "body": (
                "Customers with support_tickets > 10 signal chronic product or service pain. "
                "Escalate to Tier-2 support automatically. Conduct root-cause analysis on top complaint types "
                "(Shipping Issues, Poor Product Quality, Account Issues) and build self-service resolution flows. "
                "Assign a dedicated support liaison for enterprise and small business segments."
            )
        },
        {
            "icon": "💰",
            "title": "Price-Sensitive Churners: Value Reinforcement",
            "body": (
                "'Too Expensive' and 'Better Price Elsewhere' are top churn drivers. "
                "Introduce tiered loyalty rewards that increase perceived value without margin erosion. "
                "Deploy 'Value Statement' emails highlighting ROI and features used. "
                "Offer annual plan discounts (15–25%) 60 days before contract renewal."
            )
        },
        {
            "icon": "📧",
            "title": "Low-Engagement Customers: Re-engagement Drip Campaigns",
            "body": (
                "Customers with Low engagement_level and days_since_last_purchase > 90 need automated "
                "re-engagement journeys. Use personalised product recommendations, flash sales, and "
                "'We miss you' email sequences. A/B test subject lines and send-times to optimise open rates. "
                "Consider push notifications for mobile app users."
            )
        },
        {
            "icon": "⭐",
            "title": "Loyalty Programme Expansion",
            "body": (
                "Analysis shows 'Loyalty Points' retention method achieves the highest retention rates. "
                "Expand the loyalty programme to all subscription tiers. Introduce milestone rewards "
                "(e.g., 1-year anniversary gifts, spend threshold bonuses). Gamify the experience with "
                "tier progression (Silver → Gold → Platinum) to increase emotional commitment."
            )
        },
    ]

    for s in strategies:
        st.markdown(f"""
        <div class="strategy-card">
            <div class="strategy-title">{s['icon']} {s['title']}</div>
            <div class="strategy-body">{s['body']}</div>
        </div>""", unsafe_allow_html=True)

    # ── ROI of Retention
    st.markdown('<div class="section-header">Estimated Retention ROI</div>', unsafe_allow_html=True)
    avg_rcost = dff["customer_retention_cost_usd"].mean()
    avg_recovery = dff["recovery_revenue_usd"].mean()
    avg_impact = dff[dff["churn_flag"] == 1]["churn_revenue_impact_usd"].mean()
    total_impact = dff[dff["churn_flag"] == 1]["churn_revenue_impact_usd"].sum()

    roi1, roi2, roi3, roi4 = st.columns(4)
    roi1.metric("Avg Retention Cost / Customer", f"${avg_rcost:,.2f}")
    roi2.metric("Avg Recovery Revenue / Customer", f"${avg_recovery:,.2f}")
    roi3.metric("Avg Revenue Impact (Churned)", f"${avg_impact:,.2f}")
    roi4.metric("Total Revenue Impact (Churned)", f"${total_impact:,.0f}")

# ═══════════════════════════════════════════════
# PAGE 4 — LIVE CHURN PREDICTOR
# ═══════════════════════════════════════════════
elif page == "⚡ Live Churn Predictor":
    st.markdown('<div class="page-title">⚡ Live Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter customer attributes to predict real-time churn probability using the trained Random Forest model</div>', unsafe_allow_html=True)

    with st.form("churn_form"):
        st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            f_segment = st.selectbox("Customer Segment",
                                     ["Consumer", "Small Business", "Enterprise", "Premium"])
            f_subscription = st.selectbox("Subscription Type",
                                          ["Basic", "Premium", "Enterprise", "Unknown"])
            f_engagement = st.selectbox("Engagement Level", ["Low", "Medium", "High"])
            f_social = st.selectbox("Social Media Engagement", ["None", "Low", "Medium", "High", "Unknown"])

        with fc2:
            f_age = st.slider("Customer Age (Months)", 1, 120, 24)
            f_satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, 3)
            f_tickets = st.slider("Support Tickets", 0, 50, 3)
            f_returns = st.slider("Returns Count", 0, 30, 2)

        with fc3:
            f_spend = st.number_input("Total Spend (USD)", 0.0, 200000.0, 5000.0, step=100.0)
            f_clv = st.number_input("Customer Lifetime Value (USD)", 0.0, 200000.0, 6000.0, step=100.0)
            f_aov = st.number_input("Avg Order Value (USD)", 0.0, 50000.0, 300.0, step=10.0)
            f_purchases = st.slider("Number of Purchases", 1, 200, 15)

        st.markdown('<div class="section-header">Behavioural Signals</div>', unsafe_allow_html=True)
        fb1, fb2, fb3 = st.columns(3)

        with fb1:
            f_days_last = st.slider("Days Since Last Purchase", 0, 365, 30)
            f_loyalty = st.slider("Loyalty Score", 0.0, 100.0, 50.0)
            f_feedback = st.slider("Feedback Score (1–10)", 1, 10, 5)

        with fb2:
            f_risk = st.slider("Churn Risk Score (0–100)", 0, 100, 40)
            f_churn_prob = st.slider("Churn Probability (0–1)", 0.0, 1.0, 0.4, step=0.01)
            f_health = st.slider("Customer Health Score (0–100)", 0, 100, 65)

        with fb3:
            f_retention_prob = st.slider("Retention Probability (0–1)", 0.0, 1.0, 0.6, step=0.01)
            f_referrals = st.slider("Referral Count", 0, 30, 3)
            f_mkt_conv = st.slider("Marketing Conversions", 0, 20, 2)

        submitted = st.form_submit_button("🔮 Predict Churn Probability", use_container_width=True)

    if submitted:
        input_data = {
            "churn_risk_score": f_risk,
            "satisfaction_score": f_satisfaction,
            "support_tickets": f_tickets,
            "total_spend_usd": f_spend,
            "days_since_last_purchase": f_days_last,
            "customer_lifetime_value_usd": f_clv,
            "avg_order_value_usd": f_aov,
            "num_purchases": f_purchases,
            "customer_age_months": f_age,
            "returns_count": f_returns,
            "loyalty_score": f_loyalty,
            "feedback_score": f_feedback,
            "referral_count": f_referrals,
            "marketing_conversions": f_mkt_conv,
            "churn_probability": f_churn_prob,
            "customer_health_score": f_health,
            "retention_probability": f_retention_prob,
            "subscription_type": f_subscription,
            "customer_segment": f_segment,
            "engagement_level": f_engagement,
            "social_media_engagement": f_social,
        }
        proba, label = predict_churn(clf, encoders, feature_names, input_data)

        st.markdown("<br>", unsafe_allow_html=True)
        pr1, pr2, pr3 = st.columns([1, 2, 1])
        with pr2:
            if label == "High Risk":
                st.markdown(f"""
                <div class="pred-high">
                    <div class="pred-label">Churn Probability</div>
                    <div class="pred-value" style="color:#ef4444">{proba}%</div>
                    <div class="pred-label" style="color:#fca5a5; margin-top:8px">⚠️ HIGH CHURN RISK — Immediate action recommended</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-low">
                    <div class="pred-label">Churn Probability</div>
                    <div class="pred-value" style="color:#22c55e">{proba}%</div>
                    <div class="pred-label" style="color:#86efac; margin-top:8px">✅ LOW CHURN RISK — Customer is likely to stay</div>
                </div>""", unsafe_allow_html=True)

        # Gauge chart
        st.markdown("<br>", unsafe_allow_html=True)
        gauge_color = "#ef4444" if proba >= 70 else "#f59e0b" if proba >= 40 else "#22c55e"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba,
            delta={"reference": 50, "valueformat": ".1f",
                   "prefix": "vs 50%: ", "suffix": "%"},
            title={"text": "Churn Probability Gauge", "font": {"size": 16, "color": "#c9d1e0"}},
            number={"suffix": "%", "font": {"size": 40, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892a4"},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 40], "color": "#052e16"},
                    {"range": [40, 70], "color": "#78350f"},
                    {"range": [70, 100], "color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": proba,
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#1e2130", font=dict(color="#c9d1e0"),
            height=320, margin=dict(l=30, r=30, t=60, b=20)
        )
        g1, g2, g3 = st.columns([1, 2, 1])
        with g2:
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommended action
        st.markdown('<div class="section-header">Recommended Action</div>', unsafe_allow_html=True)
        if proba >= 70:
            recs = [
                "🚨 **Urgent**: Assign a dedicated Customer Success Manager immediately.",
                "💰 Offer a personalised retention discount or free upgrade within 24 hours.",
                "📞 Schedule a direct call — do not rely on email for this customer tier.",
                "🎯 Review all open support tickets and fast-track resolution.",
                "📊 Add to CRM 'At-Risk' watchlist for weekly executive review.",
            ]
        elif proba >= 40:
            recs = [
                "⚠️ Monitor closely — trigger a proactive check-in email within 3 days.",
                "🎁 Enrol customer in loyalty rewards acceleration programme.",
                "📧 Send personalised 'What matters to you' NPS survey.",
                "🔔 Set up automated alert if satisfaction score drops below 3.",
                "📦 Review recent order/return history for unresolved friction points.",
            ]
        else:
            recs = [
                "✅ Customer health is strong — focus on deepening the relationship.",
                "⭐ Consider inviting them to a referral or ambassador programme.",
                "📰 Share premium feature updates and insider product roadmap previews.",
                "🎂 Acknowledge loyalty milestones (anniversary, spend thresholds).",
                "🔄 Upsell opportunity — explore higher-tier subscription upgrade path.",
            ]
        for r in recs:
            st.markdown(f"- {r}")
