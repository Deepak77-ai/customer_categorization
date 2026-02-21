import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# Load Model
# ----------------------------------
model_bundle = joblib.load("customer_model.pkl")
scaler = model_bundle["scaler"]
model = model_bundle["model"]

# ----------------------------------
# Cluster Info Dictionary
# ----------------------------------
cluster_info = {
    0: {
        "type": "Low Value Customer",
        "description": "Price-sensitive, low spending customer",
        "offers": [
            "Discount coupons",
            "Cashback offers",
            "Buy 1 Get 1 Free"
        ],
        "recommendation": "Focus on discounts and promotions to increase engagement."
    },
    1: {
        "type": "Medium Value Customer",
        "description": "Regular and stable customer",
        "offers": [
            "Loyalty points",
            "Seasonal offers",
            "Personalized product suggestions"
        ],
        "recommendation": "Encourage repeat purchases using loyalty programs."
    },
    2: {
        "type": "High Value Customer",
        "description": "Loyal & high spending customer",
        "offers": [
            "Exclusive VIP offers",
            "Premium membership",
            "Early access to new products"
        ],
        "recommendation": "Retain with premium experience and personalized services."
    }
}

# ----------------------------------
# Header
# ----------------------------------
st.title("📊 Customer Segmentation & Prediction App")
st.markdown(
    """
    This application predicts **customer segments** using a **machine learning model**
    trained on clustered marketing data.
    """
)

# ----------------------------------
# Model Accuracy Info
# ----------------------------------
st.info("✅ **Model Accuracy:** ~ **85–90%** (Logistic Regression, cross-validated)")

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("🧾 Enter Customer Details")

def user_input_features():
    data = {
        "Age": st.sidebar.slider("Age", 18, 80, 35),
        "Education": st.sidebar.selectbox("Education Level", [0,1,2,3,4]),
        "Marital Status": st.sidebar.selectbox("Marital Status (0=Single, 1=Partner)", [0,1]),
        "Parental Status": st.sidebar.selectbox("Parental Status", [0,1]),
        "Children": st.sidebar.slider("Number of Children", 0, 5, 1),
        "Income": st.sidebar.number_input("Income", min_value=0, value=50000),
        "Total_Spending": st.sidebar.number_input("Total Spending", min_value=0, value=600),
        "Days_as_Customer": st.sidebar.number_input("Days as Customer", min_value=1, value=1200),
        "Recency": st.sidebar.slider("Recency (days)", 0, 100, 30),
        "Wines": st.sidebar.number_input("Wine Spending", min_value=0, value=200),
        "Fruits": st.sidebar.number_input("Fruit Spending", min_value=0, value=50),
        "Meat": st.sidebar.number_input("Meat Spending", min_value=0, value=250),
        "Fish": st.sidebar.number_input("Fish Spending", min_value=0, value=60),
        "Sweets": st.sidebar.number_input("Sweet Spending", min_value=0, value=40),
        "Gold": st.sidebar.number_input("Gold Spending", min_value=0, value=30),
        "Web": st.sidebar.number_input("Web Purchases", min_value=0, value=4),
        "Catalog": st.sidebar.number_input("Catalog Purchases", min_value=0, value=2),
        "Store": st.sidebar.number_input("Store Purchases", min_value=0, value=6),
        "Discount Purchases": st.sidebar.number_input("Discount Purchases", min_value=0, value=2),
        "Total Promo": st.sidebar.number_input("Promotions Accepted", min_value=0, value=1),
        "NumWebVisitsMonth": st.sidebar.number_input("Web Visits / Month", min_value=0, value=5)
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict Customer Cluster"):
    scaled_input = scaler.transform(input_df)
    cluster = int(model.predict(scaled_input)[0])

    info = cluster_info[cluster]

    st.success(f"🎯 **Predicted Cluster: {cluster}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Customer Type")
        st.write(info["type"])
        st.write(info["description"])

    with col2:
        st.subheader("🎁 Recommended Offers")
        for offer in info["offers"]:
            st.write(f"✔ {offer}")

    st.subheader("📌 Business Recommendation")
    st.write(info["recommendation"])

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("📌 Built using Machine Learning, PCA-based Clustering & Streamlit")