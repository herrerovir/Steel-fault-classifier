# Import required libraries
import sys
import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Dynamically resolve paths
APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from config import *

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open(MODEL_DIR / "xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# --- App UI ---
st.set_page_config(page_title = "Steel Fault Classifier", layout = "centered")

# Center title and description
st.markdown(
    """
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html = True,
)

# Title and description
st.markdown('<h1 class="centered-text">Steel Fault Classifier</h1>', unsafe_allow_html = True)
st.markdown('<p class="centered-text">Enter process parameters to predict the steel fault type (not all parameters are required).</p>', unsafe_allow_html = True)
st.write("")  # Spacer

# --- Feature Input Section ---

FEATURES = ["Pixels_Areas", "Length_of_Conveyer", "Steel_Type_A300", "Steel_Type_A400",
            "Steel_Plate_Thickness", "Empty_Index", "Square_Index", "Outside_Global_Index",
            "Orientation_Index", "Luminosity_Index", "X_Range", "Y_Range", "Defect_Area", 
            "Edge", "Outside_X_Range", "Log_Area", "Luminosity_Sum_Range", "Log_Area_Sigmoid"]

CLASS_NAMES = {0: "Pastry", 1: "Z_Scratch", 2: "K_Scratch", 3: "Stains", 4: "Dirtiness", 5: "Bumps", 6: "Other_Faults"}

# Divide FEATURES into 3 groups for columns
group_size = len(FEATURES) // 3
cols = st.columns(3)

input_data = []

for i, feature in enumerate(FEATURES):
    col_idx = i // group_size
    if col_idx > 2:
        col_idx = 2
    with cols[col_idx]:
        # Replace underscores with spaces and capitalize properly
        display_name = feature.replace("_", " ").title()
        value = st.number_input(
            display_name,
            value = 0.0,
            step = 0.1,
            format = "%.4f",
            key = feature
        )
        input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

st.write("")  # Spacer

# --- Prediction Section ---
if st.button("Predict Fault Type"):
    input_df = pd.DataFrame(input_array, columns=FEATURES)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    predicted_name = CLASS_NAMES.get(prediction, "Unknown")
    
    st.write("### Prediction")
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        predicted_prob = probabilities[prediction]
        st.markdown(
            f"""
            <div style="
                background-color:#74add1;
                padding: 1rem;
                border-radius: 0.4rem;
                font-weight: bold;
                color: white;
                margin-top: 1rem;
                text-align: center;
                font-size: 1.2rem;
            ">
                Predicted Fault Type: <strong>{predicted_name.replace("_", " ").title()}</strong><br>
                Probability: <strong>{predicted_prob:.2%}</strong>
            </div>
            """,
            unsafe_allow_html = True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background-color:#74add1;
                padding: 1rem;
                border-radius: 0.4rem;
                font-weight: bold;
                color: white;
                margin-top: 1rem;
                text-align: center;
                font-size: 1.2rem;
            ">
                Predicted Fault Type: <strong>{predicted_name.replace("_", " ").title()}</strong>
            </div>
            """,
            unsafe_allow_html = True
        )