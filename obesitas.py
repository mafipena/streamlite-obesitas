import streamlit as st
import pandas as pd
import joblib
import warnings

# Mengabaikan peringatan yang tidak perlu
warnings.filterwarnings('ignore')

# --- 1. Fungsi untuk Memuat Model dan Objek Pra-pemrosesan ---
# Menggunakan cache_resource agar model hanya dimuat sekali
@st.cache_resource
def load_model():
    """Memuat model dan objek pra-pemrosesan dari file .pkl."""
    try:
        model = joblib.load("xgboost_obesity_model.pkl")
        scaler = joblib.load("scaler.pkl")
        le_dict = joblib.load("le_dict.pkl")
        le_target = joblib.load("le_target.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
        return model, scaler, le_dict, le_target, feature_cols
    except FileNotFoundError as e:
        st.error(f"Error: File model atau preprocessing tidak ditemukan. Pastikan file .pkl berada di direktori yang sama. Detail: {e}")
        return None, None, None, None, None

# Muat semua objek
model, scaler, le_dict, le_target, feature_cols = load_model()

# Jika ada error saat memuat, hentikan aplikasi
if model is None:
    st.stop()

# --- 2. Halaman Aplikasi Streamlit ---
st.title("Prediksi Obesitas")
st.write("Aplikasi untuk memprediksi kategori obesitas berdasarkan data input.")

# --- 3. Input Pengguna di Sidebar ---
st.sidebar.header("Input Data Pengguna")

# Fungsi untuk mengambil input dari sidebar
def get_user_input():
    Gender = st.sidebar.selectbox("Jenis Kelamin", options=['Male', 'Female'])
    Age = st.sidebar.number_input("Umur", min_value=1.0, max_value=120.0, value=25.0)
    Height = st.sidebar.number_input("Tinggi (cm)", min_value=1.0, max_value=300.0, value=170.0)
    Weight = st.sidebar.number_input("Berat (kg)", min_value=1.0, max_value=500.0, value=70.0)
    family_history_with_overweight = st.sidebar.selectbox("Riwayat Keluarga Obesitas", options=['yes', 'no'])
    FAVC = st.sidebar.selectbox("Konsumsi Makanan Tinggi Kalori", options=['yes', 'no'])
    FCVC = st.sidebar.number_input("Frekuensi Konsumsi Sayur", min_value=1.0, max_value=3.0, value=2.0)
    NCP = st.sidebar.number_input("Jumlah Makanan Utama", min_value=1.0, max_value=4.0, value=3.0)
    CAEC = st.sidebar.selectbox("Konsumsi Makanan di Luar Makanan Utama", options=['Sometimes', 'Frequently', 'Always', 'no'])
    SMOKE = st.sidebar.selectbox("Merokok", options=['yes', 'no'])
    CH2O = st.sidebar.number_input("Konsumsi Air (liter)", min_value=1.0, max_value=3.0, value=2.0)
    SCC = st.sidebar.selectbox("Memantau Kalori", options=['yes', 'no'])
    FAF = st.sidebar.number_input("Frekuensi Aktivitas Fisik", min_value=0.0, max_value=3.0, value=1.0)
    TUE = st.sidebar.number_input("Waktu Penggunaan Gadget", min_value=0.0, max_value=2.0, value=0.0)
    CALC = st.sidebar.selectbox("Konsumsi Alkohol", options=['Sometimes', 'Frequently', 'Always', 'no'])
    MTRANS = st.sidebar.selectbox("Alat Transportasi", options=['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    data = {
        'Gender': Gender,
        'Age': Age,
        'Height': Height,
        'Weight': Weight,
        'family_history_with_overweight': family_history_with_overweight,
        'FAVC': FAVC,
        'FCVC': FCVC,
        'NCP': NCP,
        'CAEC': CAEC,
        'SMOKE': SMOKE,
        'CH2O': CH2O,
        'SCC': SCC,
        'FAF': FAF,
        'TUE': TUE,
        'CALC': CALC,
        'MTRANS': MTRANS
    }
    return pd.DataFrame(data, index=[0])

# Dapatkan data input
input_df = get_user_input()

# --- 4. Pra-pemrosesan Data dan Prediksi ---
if st.button("Prediksi"):
    st.write("---")
    st.subheader("Hasil Prediksi")
    
    try:
        # Menghitung BMI
        input_df['BMI'] = input_df['Weight'] / ((input_df['Height'] / 100) ** 2)
        
        # Urutkan kolom input agar sama dengan kolom fitur saat pelatihan
        # Ini penting untuk menghindari KeyError dan mismatch
        data = input_df[feature_cols]
        
        # Pra-pemrosesan kolom kategorikal menggunakan LabelEncoder yang telah dilatih
        for col, le in le_dict.items():
            if col in data.columns:
                data[col] = le.transform(data[col])
        
        # Normalisasi data menggunakan StandardScaler yang telah dilatih
        data_scaled = scaler.transform(data)
        
        # Melakukan prediksi
        prediction = model.predict(data_scaled)
        
        # Mengubah hasil prediksi numerik kembali ke label asli
        predicted_label = le_target.inverse_transform(prediction)[0]
        
        st.success(f"Kategori Obesitas yang Diprediksi: **{predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi. Pastikan semua input sudah benar. Detail error: {e}")