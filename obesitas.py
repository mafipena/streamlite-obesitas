import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Memuat model yang sudah disimpan
model = joblib.load('xgboost_obesity_model.pkl')

# Fungsi untuk preprocessing data
def preprocess_data(input_data):
    # Label encoding untuk fitur kategorikal
    label_encoder = LabelEncoder()

    # Fitur kategorikal yang perlu di-encode
    categorical_columns = ['Gender', 'family_history_with_overweight', 'CAEC', 'SCC', 'SMOKE', 'CALC', 'MTRANS']
    
    for col in categorical_columns:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    # Standardisasi fitur numerik
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH20', 'FAF', 'TUE']
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

# Elemen UI Streamlit
st.title('Prediksi Obesitas')

# Mengumpulkan input dari pengguna
gender = st.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
age = st.number_input('Usia', min_value=0, max_value=100, value=25)
height = st.number_input('Tinggi Badan (cm)', min_value=50, max_value=250, value=170)
weight = st.number_input('Berat Badan (kg)', min_value=10, max_value=200, value=70)

# Input kategori lainnya, seperti 'family_history_with_overweight'
family_history = st.selectbox('Riwayat Obesitas Keluarga', ['ya', 'tidak'])

# Input untuk fitur-fitur lain yang ada di dataset
FCVC = st.number_input('Frekuensi Konsumsi Sayuran', min_value=0, value=1)
NCP = st.number_input('Jumlah Makanan Utama per Hari', min_value=1, max_value=10, value=3)
CH20 = st.number_input('Jumlah Air yang Diminum per Hari (Liter)', min_value=0.1, max_value=10.0, value=2.0)
FAF = st.number_input('Frekuensi Aktivitas Fisik dalam Seminggu', min_value=0, max_value=7, value=3)
TUE = st.number_input('Waktu yang Dihabiskan untuk Teknologi per Hari (Jam)', min_value=0, max_value=24, value=2)

# Membuat dataframe untuk inputan
input_data = pd.DataFrame([[gender, age, height, weight, family_history, FCVC, NCP, CH20, FAF, TUE]],
                          columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FCVC', 'NCP', 'CH20', 'FAF', 'TUE'])

# Menampilkan hasil prediksi ketika tombol ditekan
if st.button('Prediksi'):
    prediction = predict(input_data)
    st.write(f'Prediksi Tingkat Obesitas: {prediction[0]}')
