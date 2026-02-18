# ================================
# 1. Import Libraries
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# 2. Page Configuration
# ================================
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Machine Learning Prediction App")
st.write("This app predicts output based on user input features.")

# ================================
# 3. Load Saved Model & Scaler
# ================================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")   # Remove if not using scaling
    return model, scaler

model, scaler = load_model()

# ================================
# 4. Sidebar for Navigation
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# ================================
# 5. Home Page
# ================================
if page == "Home":
    st.header("📌 Project Overview")
    st.write("""
    This Machine Learning model was built using:
    - Data Cleaning
    - Feature Engineering
    - Feature Selection
    - Model Training
    
    Enter values in Prediction section to get results.
    """)

# ================================
# 6. Prediction Page
# ================================
elif page == "Prediction":
    st.header("🔍 Make a Prediction")

    # ---- Example Inputs (Change According to Your Dataset) ----
    feature1 = st.number_input("Feature 1", min_value=0.0)
    feature2 = st.number_input("Feature 2", min_value=0.0)
    feature3 = st.number_input("Feature 3", min_value=0.0)

    # Convert input to DataFrame
    input_data = pd.DataFrame(
        [[feature1, feature2, feature3]],
        columns=["feature1", "feature2", "feature3"]
    )

    if st.button("Predict"):
        try:
            # Scale input (if used during training)
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)

            st.success(f"Prediction Result: {prediction[0]}")

        except Exception as e:
            st.error(f"Error: {e}")

# ================================
# 7. Footer
# ================================
st.write("---")
st.write("Built with ❤️ using Streamlit")
