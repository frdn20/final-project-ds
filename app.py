import streamlit as st
import pandas as pd
import pickle

# Muat pipeline model yang sudah disimpan
with open('income_prediction_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

st.title('Prediksi Pendapatan Tahunan')
st.write('Masukkan data individu untuk memprediksi apakah pendapatan mereka lebih dari $50K atau tidak.')

# Membuat form input di sidebar
st.sidebar.header('Input Fitur')

# Buat fungsi untuk menerima input dari pengguna
def user_input_features():
    age = st.sidebar.slider('Usia', 17, 90, 37)
    workclass = st.sidebar.selectbox('Kelas Pekerjaan', ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Other'])
    final_weight = st.sidebar.number_input('Final Weight', 12285, 1490400, 178142)
    education_num = st.sidebar.slider('Nomor Edukasi', 1, 16, 10)
    marital_status = st.sidebar.selectbox('Status Pernikahan', ['Married', 'Never-married', 'Divorced'])
    occupation = st.sidebar.selectbox('Pekerjaan', ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'])
    relationship = st.sidebar.selectbox('Hubungan', ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
    race = st.sidebar.selectbox('Ras', ['White', 'Non-White'])
    gender = st.sidebar.selectbox('Jenis Kelamin', ['Male', 'Female'])
    capital_gain = st.sidebar.slider('Capital Gain', 0, 1, 0) # Input sebagai biner
    capital_loss = st.sidebar.slider('Capital Loss', 0, 1, 0) # Input sebagai biner
    hours_per_week = st.sidebar.slider('Jam Kerja per Minggu', 1, 99, 40)
    native_country = st.sidebar.selectbox('Negara Asal', ['USA', 'Non-USA'])

    data = {
        'Age': age,
        'Workclass': workclass,
        'Final Weight': final_weight,
        'EducationNum': education_num,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Relationship': relationship,
        'Race': race,
        'Gender': gender,
        'Capital Gain': capital_gain,
        'capital loss': capital_loss,
        'Hours per Week': hours_per_week,
        'Native Country': native_country
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Dapatkan input dari pengguna
input_df = user_input_features()

# Tampilkan input pengguna dalam bentuk tabel
st.subheader('Data Input Pengguna')
st.write(input_df)

# Buat tombol untuk prediksi
if st.button('Prediksi'):
    # Lakukan prediksi
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi')
    income_result = '>50K' if prediction[0] == 1 else '<=50K'
    st.write(f"Prediksi Pendapatan: **{income_result}**")

    st.subheader('Probabilitas Prediksi')
    st.write(f"Probabilitas untuk <=50K: {prediction_proba[0][0]:.2f}")
    st.write(f"Probabilitas untuk >50K: {prediction_proba[0][1]:.2f}")