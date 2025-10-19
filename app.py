# ===== app_319.py =====
# Streamlit app with sliders for user input
# Student ID: 67130700319

import streamlit as st
import pandas as pd
import pickle

# --- Page setup ---
st.set_page_config(page_title="Sales Prediction App", layout="centered")
st.title("📊 Sales Prediction using Linear Regression")
st.write("Use the sliders below to adjust your advertising budgets:")

# --- Step 1: Load trained model ---
@st.cache_resource
def load_model():
    with open("model_reg_67130700319.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- Step 2: User Input with sliders ---
youtube = st.slider("📺 YouTube Budget", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
tiktok = st.slider("🎵 TikTok Budget", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
instagram = st.slider("📸 Instagram Budget", min_value=0.0, max_value=300.0, value=50.0, step=1.0)

# --- Step 3: Predict when button clicked ---
if st.button("🔮 Predict Sales"):
    new_data = pd.DataFrame({
        "youtube": [youtube],
        "tiktok": [tiktok],
        "instagram": [instagram]
    })
    prediction = model.predict(new_data)
    st.success(f"💰 **Estimated Sales:** {prediction[0]:.2f}")

st.markdown("---")
st.caption("Model: model_reg_67130700319.pkl  |  Built with scikit-learn & Streamlit")
