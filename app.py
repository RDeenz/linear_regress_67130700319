
# ===== app_319.py =====
# Streamlit app to predict Sales using trained Linear Regression model
# Student ID: 67130700319

import streamlit as st
import pandas as pd
import pickle

# --- Page setup ---
st.set_page_config(page_title="Sales Prediction App", layout="centered")
st.title("📊 Sales Prediction using Linear Regression")
st.write("Enter advertising budgets for each platform below:")

# --- Step 1: Load trained model ---
@st.cache_resource
def load_model():
    with open("model_reg_67130700319.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- Step 2: User Input ---
youtube = st.number_input("YouTube Budget", min_value=0.0, value=50.0, step=1.0)
tiktok = st.number_input("TikTok Budget", min_value=0.0, value=50.0, step=1.0)
instagram = st.number_input("Instagram Budget", min_value=0.0, value=50.0, step=1.0)

# --- Step 3: Predict ---
if st.button("Predict Sales"):
    new_data = pd.DataFrame({
        "youtube": [youtube],
        "tiktok": [tiktok],
        "instagram": [instagram]
    })
    prediction = model.predict(new_data)
    st.success(f"💰 **Estimated Sales:** {prediction[0]:.2f}")

st.markdown("---")
st.caption("Model: model_reg_67130700319.pkl  |  Built with scikit-learn & Streamlit")
