import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# ==================== LOAD & TRAIN MODEL ==================== #
@st.cache_resource
def load_model():
    # Baca dataset
    df = pd.read_csv("Obesity prediction.csv")

    # Ganti sesuai nama kolom target di dataset kamu
    target_col = "NObeyesdad"

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label encoding untuk kolom kategorikal
    le_dict = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    # Encoding target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    return model, scaler, le_dict, le_target, list(X.columns)

model, scaler, le_dict, le_target, feature_cols = load_model()

# ==================== UI ==================== #
st.markdown("<h1 style='text-align: center; color: #333;'>🏥 Obesity Prediction</h1>", unsafe_allow_html=True)
st.write("Isi data berikut untuk memprediksi tingkat risiko obesitas.")

with st.form("obesity_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("🧍 Age", min_value=1, max_value=120)
        Weight = st.number_input("⚖ Weight (kg)", min_value=1.0)
        Main_meals = st.slider("🍽 Main Meals (1-4)", 1, 4)
        Physical_activity = st.slider("🏃 Physical Activity (0-3)", 0, 3)
        Smoking = st.selectbox("🚬 Do you smoke?", ["No", "Yes"])
        High_cal = st.selectbox("🍔 High Calorie Food?", ["No", "Yes"])
        Snacking = st.selectbox("🍪 Snacking?", ["No", "Yes"])
        Gender = st.selectbox("⚧ Gender", ["Male", "Female"])

    with col2:
        Height = st.number_input("📏 Height (m)", min_value=0.5, max_value=2.5)
        Veg_consume = st.slider("🥦 Vegetable Consumption (1-3)", 1, 3)
        Water = st.slider("💧 Water Intake (1-3)", 1, 3)
        Tech_usage = st.slider("💻 Tech Usage (0-2)", 0, 2)
        Calories_monitor = st.selectbox("📊 Calories Monitor?", ["No", "Yes"])
        Family_history = st.selectbox("👨‍👩‍👧 Family Obesity History?", ["No", "Yes"])
        Alcohol = st.selectbox("🍷 Alcohol Consumption?", ["No", "Yes"])
        Transport = st.selectbox("🚶 Transport Type", ["Walking", "Bike", "Car", "Public Transport"])

    submit = st.form_submit_button("🔍 Predict")

# ==================== PREDICTION ==================== #
if submit:
    # Buat dataframe dari input
    input_data = pd.DataFrame([[Age, Height, Weight, Veg_consume, Main_meals, Water, Tech_usage,
                                Smoking, Calories_monitor, High_cal, Family_history, Snacking,
                                Alcohol, Gender, Transport]],
                              columns=feature_cols)

    # Encode kategorikal
    for col in input_data.select_dtypes(include=['object']).columns:
        le = le_dict[col]
        input_data[col] = le.transform(input_data[col])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    pred = model.predict(input_scaled)
    pred_label = le_target.inverse_transform(pred)[0]

    st.success(f"📊 Prediction Result: *{pred_label}*")