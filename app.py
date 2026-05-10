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

    👈 **Fill in the customer details in the sidebar**, then click **Predict Customer Cluster** to see results.
    """
)

# ----------------------------------
# Model Accuracy Info
# ----------------------------------
st.info("✅ **Model Accuracy:** ~ **85–90%** (Logistic Regression, cross-validated)")

# ----------------------------------
# Feature Guide Expander
# ----------------------------------
with st.expander("📖 How to Fill the Form — Field Guide", expanded=False):
    st.markdown("""
    ### 👤 People / Demographics
    | Field | What to Enter |
    |-------|--------------|
    | **Age** | Customer's current age (18–80) |
    | **Education Level** | 0 = Basic, 1 = High School, 2 = Graduation, 3 = Master, 4 = PhD |
    | **Marital Status** | 0 = Single / Divorced / Widowed, 1 = Married / Together (has partner) |
    | **Parental Status** | 0 = No children at home, 1 = Has children or teenagers at home |
    | **Number of Children** | Total kids + teens living in the household (0–5) |
    | **Income** | Customer's yearly household income in ₹ or $ (e.g., 50000) |

    ### 🛍️ Purchase Behavior
    | Field | What to Enter |
    |-------|--------------|
    | **Total Spending** | Total amount spent across all product categories in the last 2 years |
    | **Days as Customer** | Number of days since the customer first enrolled/registered |
    | **Recency (days)** | How many days ago the customer made their **last purchase** (lower = more recent) |

    ### 🍷 Product Spending (last 2 years)
    | Field | What to Enter |
    |-------|--------------|
    | **Wine Spending** | Total amount spent on wine products |
    | **Fruit Spending** | Total amount spent on fruits |
    | **Meat Spending** | Total amount spent on meat products |
    | **Fish Spending** | Total amount spent on fish products |
    | **Sweet Spending** | Total amount spent on sweet products |
    | **Gold Spending** | Total amount spent on gold / luxury products |

    ### 🛒 Purchase Channels
    | Field | What to Enter |
    |-------|--------------|
    | **Web Purchases** | Number of purchases made through the company's **website** |
    | **Catalog Purchases** | Number of purchases made using a **printed catalog** |
    | **Store Purchases** | Number of purchases made **in physical stores** |
    | **Discount Purchases** | Number of purchases made **using a discount/deal** |

    ### 📣 Promotions & Visits
    | Field | What to Enter |
    |-------|--------------|
    | **Promotions Accepted** | Total number of marketing campaigns the customer accepted (0–5+) |
    | **Web Visits / Month** | How many times the customer visited the website in the **last month** |
    """)

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("🧾 Enter Customer Details")
st.sidebar.markdown("_Fill in each field. Hover over field names for hints._")

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Demographics")

def user_input_features():

    # --- Demographics ---
    age = st.sidebar.slider(
        "Age",
        min_value=18, max_value=80, value=35,
        help="Customer's current age in years. Range: 18–80."
    )

    education = st.sidebar.selectbox(
        "Education Level",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "0 — Basic / No formal education",
            1: "1 — High School",
            2: "2 — Graduation (Bachelor's)",
            3: "3 — Master's Degree",
            4: "4 — PhD / Doctorate"
        }[x],
        help="Customer's highest education level. Select the closest match."
    )

    marital = st.sidebar.selectbox(
        "Marital Status",
        options=[0, 1],
        format_func=lambda x: {
            0: "0 — Single / Divorced / Widowed (no partner)",
            1: "1 — Married / In a Relationship (has partner)"
        }[x],
        help="0 = Customer lives alone (single, divorced, widowed). 1 = Customer lives with a partner."
    )

    parental = st.sidebar.selectbox(
        "Parental Status",
        options=[0, 1],
        format_func=lambda x: {
            0: "0 — No children at home",
            1: "1 — Has children / teenagers at home"
        }[x],
        help="Does the customer have kids or teenagers living in their household?"
    )

    children = st.sidebar.slider(
        "Number of Children (Kids + Teens)",
        min_value=0, max_value=5, value=1,
        help="Total number of children AND teenagers living in the customer's home."
    )

    income = st.sidebar.number_input(
        "Annual Household Income (₹ / $)",
        min_value=0, value=50000, step=1000,
        help="Customer's total yearly household income. Example: 50000 means ₹50,000 or $50,000."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛍️ Purchase Behavior")

    total_spending = st.sidebar.number_input(
        "Total Spending (last 2 years)",
        min_value=0, value=600, step=50,
        help="Total amount the customer has spent across ALL product categories over the last 2 years."
    )

    days_customer = st.sidebar.number_input(
        "Days as Customer",
        min_value=1, value=1200, step=10,
        help="Number of days since the customer first enrolled/registered. Example: 1200 = ~3.3 years as a customer."
    )

    recency = st.sidebar.slider(
        "Recency — Days Since Last Purchase",
        min_value=0, max_value=100, value=30,
        help="How many days ago the customer last made a purchase. Lower value = more recently active. 0 = bought today."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🍷 Product Category Spending (last 2 years)")

    wines = st.sidebar.number_input(
        "Wine Spending",
        min_value=0, value=200, step=10,
        help="Total amount spent on wine products in the last 2 years."
    )

    fruits = st.sidebar.number_input(
        "Fruit Spending",
        min_value=0, value=50, step=5,
        help="Total amount spent on fruit products in the last 2 years."
    )

    meat = st.sidebar.number_input(
        "Meat Spending",
        min_value=0, value=250, step=10,
        help="Total amount spent on meat products in the last 2 years."
    )

    fish = st.sidebar.number_input(
        "Fish Spending",
        min_value=0, value=60, step=5,
        help="Total amount spent on fish products in the last 2 years."
    )

    sweets = st.sidebar.number_input(
        "Sweet Spending",
        min_value=0, value=40, step=5,
        help="Total amount spent on sweet/candy products in the last 2 years."
    )

    gold = st.sidebar.number_input(
        "Gold / Luxury Spending",
        min_value=0, value=30, step=5,
        help="Total amount spent on gold or premium/luxury products in the last 2 years."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛒 Purchase Channels")

    web = st.sidebar.number_input(
        "Web Purchases",
        min_value=0, value=4, step=1,
        help="Total number of purchases made through the company's WEBSITE."
    )

    catalog = st.sidebar.number_input(
        "Catalog Purchases",
        min_value=0, value=2, step=1,
        help="Total number of purchases made by ordering from a printed CATALOG."
    )

    store = st.sidebar.number_input(
        "Store Purchases",
        min_value=0, value=6, step=1,
        help="Total number of purchases made directly at a PHYSICAL STORE."
    )

    discount_purchases = st.sidebar.number_input(
        "Discount / Deal Purchases",
        min_value=0, value=2, step=1,
        help="Number of purchases where a DISCOUNT or special deal was used."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📣 Promotions & Online Activity")

    total_promo = st.sidebar.number_input(
        "Total Promotions Accepted",
        min_value=0, value=1, step=1,
        help="How many marketing campaigns has this customer accepted in total? (Campaigns 1–5 + last campaign)"
    )

    web_visits = st.sidebar.number_input(
        "Website Visits per Month",
        min_value=0, value=5, step=1,
        help="How many times did the customer visit the company's website in the last month?"
    )

    data = {
        "Age": age,
        "Education": education,
        "Marital Status": marital,
        "Parental Status": parental,
        "Children": children,
        "Income": income,
        "Total_Spending": total_spending,
        "Days_as_Customer": days_customer,
        "Recency": recency,
        "Wines": wines,
        "Fruits": fruits,
        "Meat": meat,
        "Fish": fish,
        "Sweets": sweets,
        "Gold": gold,
        "Web": web,
        "Catalog": catalog,
        "Store": store,
        "Discount Purchases": discount_purchases,
        "Total Promo": total_promo,
        "NumWebVisitsMonth": web_visits
    }

    return pd.DataFrame([data])


input_df = user_input_features()

# ----------------------------------
# Input Summary Table
# ----------------------------------
st.subheader("📋 Customer Data Summary")
st.markdown("Review the inputs below before predicting:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Demographics**")
    st.table(input_df[["Age", "Education", "Marital Status", "Parental Status", "Children", "Income"]].T.rename(columns={0: "Value"}))

with col2:
    st.markdown("**🍷 Product Spending**")
    st.table(input_df[["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Total_Spending"]].T.rename(columns={0: "Value"}))

with col3:
    st.markdown("**🛒 Behavior & Channels**")
    st.table(input_df[["Recency", "Days_as_Customer", "Web", "Catalog", "Store", "Discount Purchases", "Total Promo", "NumWebVisitsMonth"]].T.rename(columns={0: "Value"}))

# ----------------------------------
# Prediction Button
# ----------------------------------
st.markdown("---")
if st.button("🔍 Predict Customer Cluster", use_container_width=True):
    scaled_input = scaler.transform(input_df)
    cluster = int(model.predict(scaled_input)[0])

    info = cluster_info[cluster]

    cluster_colors = {0: "🔴", 1: "🟡", 2: "🟢"}
    st.success(f"{cluster_colors[cluster]} **Predicted Cluster: {cluster} — {info['type']}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Customer Type")
        st.write(f"**{info['type']}**")
        st.write(info["description"])

    with col2:
        st.subheader("🎁 Recommended Offers")
        for offer in info["offers"]:
            st.write(f"✔ {offer}")

    st.subheader("📌 Business Recommendation")
    st.info(info["recommendation"])

    st.markdown("#### 🧠 Cluster Legend")
    st.markdown("""
    | Cluster | Type | Description |
    |---------|------|-------------|
    | 🔴 **0** | Low Value Customer | Price-sensitive, infrequent buyer, low spending |
    | 🟡 **1** | Medium Value Customer | Regular buyer, moderate spending, stable engagement |
    | 🟢 **2** | High Value Customer | Loyal, frequent, high-spending premium customer |
    """)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("📌 Built using Machine Learning, PCA-based Clustering & Streamlit  |  Hover over sidebar fields for input guidance")
