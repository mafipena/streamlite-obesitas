import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Memuat model yang sudah disimpan
model = joblib.load('xgboost_obesity_model.pkl')

# Fungsi untuk preprocessing data
def preprocess_data(input_data):
    # Label encoding untuk fitur kategorikal
    label_encoder = LabelEncoder()

    # Fitur kategorikal yang perlu di-encode
    categorical_columns = ['Gender', 'family_history_with_overweight', 'smoke', 'high_calorie_food', 'snacking', 'calories_monitor', 'alcohol', 'transportation']
    
    for col in categorical_columns:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    # Standardisasi fitur numerik
    numerical_columns = ['Age', 'Height', 'Weight', 'veg_consumption', 'water_intake', 'tech_usage', 'main_meals', 'physical_activity']
    scaler = StandardScaler()
    input_data[numerical_columns] = scaler.fit_transform(input_data[numerical_columns])

    # Menggunakan SMOTE jika diperlukan (untuk mengatasi ketidakseimbangan kelas)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, _ = smote.fit_resample(input_data[numerical_columns], input_data['Obesity'])

    return X_resampled

# Fungsi untuk prediksi
def predict(input_data):
    # Melakukan preprocessing data terlebih dahulu
    processed_data = preprocess_data(input_data)
    # Melakukan prediksi dengan model
    prediction = model.predict(processed_data)
    return prediction

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
# 5. Form Input User (2 Kolom)
# -----------------------------
st.markdown("## 📝 Masukkan Data Anda")
with st.form("obesity_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🧍 Usia", min_value=1, max_value=120, value=25)
        weight = st.number_input("⚖ Berat Badan (kg)", min_value=1.0, value=60.0)
        main_meals = st.number_input("🍽 Makanan Utama (1-4)", min_value=1, max_value=4, value=3)
        physical_activity = st.selectbox("🏃 Aktivitas Fisik", ["Low", "Medium", "High"])
        smoke = st.selectbox("🚬 Apakah Anda Merokok?", ["Tidak", "Ya"])
        high_calorie_food = st.selectbox("🍔 Makanan Tinggi Kalori?", ["Tidak", "Ya"])
        snacking = st.selectbox("🍪 Camilan?", ["Tidak", "Terkadang", "Sering", "Selalu"])
        gender = st.selectbox("⚧ Jenis Kelamin", ["Laki-laki", "Perempuan"])
    with col2:
        height = st.number_input("📏 Tinggi Badan (m)", min_value=0.5, max_value=2.5, value=1.65)
        
        # Menggunakan keterangan untuk skala rendah, menengah, dan tinggi
        veg_consumption = st.selectbox("🥦 Konsumsi Sayuran (1-3)", [1, 2, 3])
        water_intake = st.selectbox("💧 Asupan Air", ["Low", "Medium", "High"])
        tech_usage = st.selectbox("💻 Penggunaan Teknologi", ["Low", "Medium", "High"])
        calories_monitor = st.selectbox("📊 Memantau Kalori?", ["Tidak", "Ya"])
        family_history = st.selectbox("👨‍👩‍👧 Riwayat Obesitas Keluarga?", ["Tidak", "Ya"])
        alcohol = st.selectbox("🍷 Konsumsi Alkohol?", ["Tidak", "Terkadang", "Sering", "Selalu"])
        transportation = st.selectbox("🚶 Jenis Transportasi", ["Berjalan", "Sepeda", "Motor", "Mobil", "Transportasi Umum"])

    submit = st.form_submit_button("🔍 Prediksi", use_container_width=True)

# -----------------------------
# 6. Menampilkan Hasil Prediksi
# -----------------------------
if submit:
    # Membuat dataframe untuk inputan
    input_data = pd.DataFrame([[gender, age, height, weight, family_history, main_meals, physical_activity, smoke,
                                high_calorie_food, snacking, veg_consumption, water_intake, tech_usage, calories_monitor, alcohol, transportation]],
                              columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'main_meals', 'physical_activity', 'smoke',
                                       'high_calorie_food', 'snacking', 'veg_consumption', 'water_intake', 'tech_usage', 'calories_monitor', 'alcohol', 'transportation'])
    
    # Pastikan kolom yang digunakan sesuai dengan input yang ada
    relevant_columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'main_meals', 'physical_activity', 'smoke',
                        'high_calorie_food', 'snacking', 'veg_consumption', 'water_intake', 'tech_usage', 'calories_monitor', 'alcohol', 'transportation']
    
    # Hanya ambil kolom yang relevan
    input_data = input_data[relevant_columns]

    # Prediksi
    prediction = predict(input_data)
    st.write(f'Prediksi Tingkat Obesitas: {prediction[0]}')
