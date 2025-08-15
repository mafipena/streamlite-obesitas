import streamlit as st
import pandas as pd
import numpy as np
import joblib

obes = joblib.load("xgboost_obesity_model.pkl")

st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .card {
            padding: 15px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<h1 style='text-align: center; color: #333;'>🏥 Obesity Prediction</h1>", unsafe_allow_html=True)
st.write("Isi data berikut untuk memprediksi tingkat risiko obesitas.")

# --- FORM INPUT ---
with st.form("obesity_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🧍 Age", min_value=1, max_value=120)
        weight = st.number_input("⚖ Weight (kg)", min_value=1.0)
        main_meals = st.slider("🍽 Main Meals (1-4)", 1, 4)
        physical_activity = st.slider("🏃 Physical Activity (0-3)", 0, 3)
        smoke = st.selectbox("🚬 Do you smoke?", ["No", "Yes"])
        high_calorie_food = st.selectbox("🍔 High Calorie Food?", ["No", "Yes"])
        snacking = st.selectbox("🍪 Snacking?", ["No", "Yes"])
        gender = st.selectbox("⚧ Gender", ["Male", "Female"])

    with col2:
        height = st.number_input("📏 Height (m)", min_value=0.5, max_value=2.5)
        veg_consumption = st.slider("🥦 Vegetable Consumption (1-3)", 1, 3)
        water_intake = st.slider("💧 Water Intake (1-3)", 1, 3)
        tech_usage = st.slider("💻 Tech Usage (0-2)", 0, 2)
        calories_monitor = st.selectbox("📊 Calories Monitor?", ["No", "Yes"])
        family_history = st.selectbox("👨‍👩‍👧 Family Obesity History?", ["No", "Yes"])
        alcohol = st.selectbox("🍷 Alcohol Consumption?", ["No", "Yes"])
        transportation = st.selectbox("🚶 Transport Type", ["Walking", "Bike", "Car", "Public Transport"])

    submit = st.form_submit_button("🔍 Predict")

# --- PREDICTION ---
if submit:
    # Contoh model prediksi dummy
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        result = "Underweight"
        color = "#5bc0de"
    elif bmi < 25:
        result = "Normal weight"
        color = "#5cb85c"
    elif bmi < 30:
        result = "Overweight"
        color = "#f0ad4e"
    else:
        result = "Obese"
        color = "#d9534f"

    st.markdown(f"""
        <div class="card" style="border-left: 8px solid {color};">
            <h3>📊 Prediction Result</h3>
            <p style="font-size:18px;">Your BMI: <b>{bmi:.2f}</b></p>
            <p style="font-size:20px; color:{color};"><b>{result}</b></p>
        </div>
    """, unsafe_allow_html=True)