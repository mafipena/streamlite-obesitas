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
    X = df.drop('Obesity', axis=1)
    y = df['Obesity']
    return X, y, label_encoders

X, y, label_encoders = preprocess_data(df)

# Standarisasi data numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menangani data tidak seimbang dengan SMOTE
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
# 5. Form Input User
# -----------------------------
st.subheader("📝 Masukkan Data Anda")
user_data = {}
for col in X.columns:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        user_data[col] = st.selectbox(f"{col}:", options)
    else:
        user_data[col] = st.number_input(f"{col}:", value=0.0)

# -----------------------------
# 6. Prediksi
# -----------------------------
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame([user_data])
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    pred_label = label_encoders['ObesityCategory'].inverse_transform([pred])[0]
    
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
        target_names=label_encoders['ObesityCategory'].classes_
    )
    st.text(report)