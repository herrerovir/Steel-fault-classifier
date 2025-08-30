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

st.set_page_config(page_title = "Steel Fault Classifier", layout = "wide")

# Customize style
st.markdown(
    """
    <style>
    .centered-text {
        text-align: center;
    }
    .result-box {
        background-color: #74add1;
        padding: 1rem;
        border-radius: 0.4rem;
        font-weight: bold;
        color: white;
        margin-top: 1rem;
        text-align: center;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html = True,
)

# Title and description (centered)
st.markdown('<h1 class="centered-text">Steel Fault Classifier</h1>', unsafe_allow_html = True)
st.markdown('<p class="centered-text">Enter the process parameters to predict the steel fault type. Not all parameters are required.</p>', unsafe_allow_html = True)

# --- Sidebar for Example Values ---

with st.sidebar:
    st.subheader("Example Feature Values:")
    EXAMPLES = {
        "Defect_Area": "100", "Edge": "0.10", "Empty_Index": "0.10",
        "Length_of_Conveyer": "1000", "Log_Area": "1.10", "Log_Area_Sigmoid": "1.10",
        "Luminosity_Index": "0.10", "Luminosity_Sum_Range": "110000", "Orientation_Index": "0.10",
        "Outside_Global_Index": "0", "Outside_X_Range": "0.10", "Pixels_Areas": "200",
        "Steel_Plate_Thickness": "100", "Steel_Type_A300": "0", "Steel_Type_A400": "0",
        "Square_Index": "0.10", "X_Range": "10", "Y_Range": "10"
    }

    for feature, example in EXAMPLES.items():
        st.write(f"{feature.replace('_', ' ').title()}: **{example}**")
        
# --- Main Panel for Input Fields ---

st.subheader("Feature Values")

FEATURES = [
    "Pixels_Areas", "Length_of_Conveyer", "Steel_Type_A300", 
    "Steel_Type_A400", "Steel_Plate_Thickness", "Empty_Index", 
    "Square_Index", "Outside_Global_Index", "Orientation_Index", 
    "Luminosity_Index", "X_Range", "Y_Range", "Defect_Area", 
    "Edge", "Outside_X_Range", "Log_Area", "Luminosity_Sum_Range", 
    "Log_Area_Sigmoid" 
]

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Assign features to the columns (6 features per column)
input_data = []

# Column 1
with col1:
    for i in range(6):
        feature = FEATURES[i]
        display_name = feature.replace("_", " ").title()
        value = st.number_input(
            display_name,
            value = float(EXAMPLES[feature]),  # Set default to example values
            step = 0.1,
            format = "%.2f",
            key = feature
        )
        input_data.append(value)

# Column 2
with col2:
    for i in range(6, 12):
        feature = FEATURES[i]
        display_name = feature.replace("_", " ").title()
        value = st.number_input(
            display_name,
            value = float(EXAMPLES[feature]),
            step = 0.1,
            format = "%.2f",
            key = feature
        )
        input_data.append(value)

# Column 3
with col3:
    for i in range(12, len(FEATURES)):
        feature = FEATURES[i]
        display_name = feature.replace("_", " ").title()
        value = st.number_input(
            display_name,
            value = float(EXAMPLES[feature]),
            step = 0.1,
            format = "%.2f",
            key = feature
        )
        input_data.append(value)

# --- Prediction Section ---

input_array = np.array(input_data).reshape(1, -1)

CLASS_NAMES = {
    0: "Pastry", 1: "Z_Scratch", 2: "K_Scratch", 3: "Stains", 
    4: "Dirtiness", 5: "Bumps", 6: "Other_Faults"
}

if st.button("Predict Fault Type"):
    input_df = pd.DataFrame(input_array, columns=FEATURES)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    predicted_name = CLASS_NAMES.get(prediction, "Unknown")

    st.write("### Prediction Result")
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        predicted_prob = probabilities[prediction]
        st.markdown(
            f"""
            <div class="result-box">
                Predicted Fault Type: <strong>{predicted_name.replace("_", " ").title()}</strong><br>
                Probability: <strong>{predicted_prob:.2%}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-box">
                Predicted Fault Type: <strong>{predicted_name.replace("_", " ").title()}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )