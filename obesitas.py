import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# 1. Konfigurasi Halaman
# -----------------------------
st.set_page_config(page_title="Prediksi Obesitas", page_icon="🍔", layout="centered")

st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="color: #2E86C1;">🍔 Prediksi Tingkat Obesitas</h1>
    <p style="font-size: 18px; color: #555;">
        Aplikasi ini memprediksi tingkat obesitas berdasarkan data kebiasaan makan, aktivitas fisik, dan gaya hidup Anda.
    </p>
    <hr style="border: 1px solid #ccc;">
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 2. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity prediction.csv")
    return df

df = load_data()

# -----------------------------
# 3. Preprocessing Data
# -----------------------------
def preprocess_data(df):
    df = df.copy()
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    target_col = 'Obesity' if 'Obesity' in df.columns else 'Obesity'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y, label_encoders, target_col

X, y, label_encoders, target_col = preprocess_data(df)

# Standarisasi data numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menangani data tidak seimbang
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Training Model
# -----------------------------
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. Form Input User (2 Kolom)
# -----------------------------
st.subheader("📝 Masukkan Data Anda")

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

# -----------------------------
# 6. Prediksi
# -----------------------------
if submit:
    # Membuat dataframe input sesuai urutan kolom
    input_dict = {
        "Age": age,
        "Weight": weight,
        "Height": height,
        "MainMeals": main_meals,
        "PhysicalActivity": physical_activity,
        "Smoke": smoke,
        "HighCaloricFood": high_calorie_food,
        "Snacking": snacking,
        "Gender": gender,
        "VegConsumption": veg_consumption,
        "WaterIntake": water_intake,
        "TechUsage": tech_usage,
        "CaloriesMonitor": calories_monitor,
        "FamilyHistory": family_history,
        "Alcohol": alcohol,
        "Transportation": transportation
    }
    input_df = pd.DataFrame([input_dict])

    # Encode kolom kategori
    for col in label_encoders:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Standarisasi
    input_scaled = scaler.transform(input_df)

    # Prediksi
    pred = model.predict(input_scaled)[0]
    pred_label = label_encoders[target_col].inverse_transform([pred])[0]

    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #F4F6F7;">
        <h2>📊 Hasil Prediksi: <span style="color: #E74C3C;">{pred_label}</span></h2>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 7. Evaluasi Model
# -----------------------------
if st.checkbox("📈 Tampilkan evaluasi model"):
    y_pred_test = model.predict(X_test)
    report = classification_report(
        y_test, y_pred_test,
        target_names=label_encoders[target_col].classes_
    )
    st.text(report)