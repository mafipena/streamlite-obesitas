import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model xgboost
model = joblib.load("xgboost_obesity_model.pkl")

# CSS tampilan
st.markdown("""
<style>
.stApp {
    font-family: 'Arial', sans-serif;
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: white;
}
h1 {
    text-align: center;
    color: white !important;
}
h2, h3, h4, h5, h6, p, label {
    color: white !important;
}
div.stButton > button {
    display: block;
    margin: 0 auto;
    background-color: #ff7f50;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.markdown("<h1>🔍 Prediksi Obesitas</h1>", unsafe_allow_html=True)

# --- FORM INPUT ---
with st.form("obesity_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🧍 Age", min_value=1, max_value=120)
        weight = st.number_input("⚖ Weight (kg)", min_value=1.0)
        main_meals = st.number_input("🍽 Main Meals (1-4)", min_value=1, max_value=4)
        physical_activity = st.number_input("🏃 Physical Activity (0-3)", min_value=0, max_value=3)
        smoke = st.selectbox("🚬 Do you smoke?", ["No", "Yes"])
        high_calorie_food = st.selectbox("🍔 High Calorie Food?", ["No", "Yes"])
        snacking = st.selectbox("🍪 Snacking?", ["No", "Sometimes", "Frequently", "Always"])
        gender = st.selectbox("⚧ Gender", ["Male", "Female"])

    with col2:
        height = st.number_input("📏 Height (m)", min_value=0.5, max_value=2.5)
        veg_consumption = st.selectbox("🥦 Vegetable Consumption", ["Low", "Medium", "High"])
        water_intake = st.selectbox("💧 Water Intake", ["Low", "Medium", "High"])
        tech_usage = st.selectbox("💻 Tech Usage", ["Low", "Medium", "High"])
        calories_monitor = st.selectbox("📊 Calories Monitor?", ["No", "Yes"])
        family_history = st.selectbox("👨‍👩‍👧 Family Obesity History?", ["No", "Yes"])
        alcohol = st.selectbox("🍷 Alcohol Consumption?", ["No", "Sometimes", "Frequently", "Always"])
        transportation = st.selectbox("🚶 Transport Type", ["Walking", "Bike", "Motorbike", "Automobile", "Public transportation"])

    submit = st.form_submit_button("🔍 Predict")

# --- PREDIKSI ---
if submit:
    # Encode data sesuai kebutuhan model
    data = pd.DataFrame([[
        age, weight, main_meals, physical_activity, smoke, high_calorie_food, snacking, gender,
        height, veg_consumption, water_intake, tech_usage, calories_monitor, family_history, alcohol, transportation
    ]], columns=[
        "Age", "Weight", "NCP", "FAF", "SMOKE", "FAVC", "CAEC", "Gender",
        "Height", "FCVC", "CH2O", "TUE", "SCC", "family_monitor",
        "CALC", "MTRANS"
    ])

    # Prediksi
    prediction = model.predict(data)[0]

    st.markdown(f"<h2 style='text-align:center;'>Hasil Prediksi: {prediction}</h2>", unsafe_allow_html=True)