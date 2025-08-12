import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import cloudpickle
import xgboost  # Pastikan library ini di-import
import imblearn   # Pastikan library ini di-import

# --- Konfigurasi dan Pemuatan Model ---

# Setel judul dan ikon halaman
st.set_page_config(page_title="Income Prediction", page_icon="💰", layout="centered")

# Gunakan cache untuk memuat model agar lebih cepat pada interaksi berikutnya
@st.cache_resource
def load_model():
    """Memuat pipeline model yang sudah dilatih."""
    try:
        with open("income_prediction_pipeline.pkl", "rb") as f:
            model = cloudpickle.load(f)
        return model
    except FileNotFoundError:
        st.error("File model 'XGBoost_Model.pkl' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
        return None

XGBoost_Model = load_model()

# --- Definisi Variabel Global dan UI ---

# Template HTML untuk judul
html_temp = """
<div style="background-color:#024959;padding:10px;border-radius:10px;margin-bottom:20px;">
    <h1 style="color:#F2E3B3;text-align:center;">Income Category Prediction App</h1> 
    <h4 style="color:#F2E3B3;text-align:center;">Decision Support Tool</h4> 
</div>
"""

# Daftar fitur ini HARUS SAMA PERSIS dengan yang digunakan saat training model
# Diambil dari notebook 'Out [85]' atau 'Out [88]'
MODEL_COLUMNS = [
    'Capital Gain', 'capital loss', 'Race_White', 'Gender_Male', 'Native Country_USA',
    'Age', 'Final Weight', 'EducationNum', 'Hours per Week', 'Workclass_Federal-gov',
    'Workclass_Local-gov', 'Workclass_Never-worked', 'Workclass_Other', 'Workclass_Private',
    'Workclass_Self-emp-inc', 'Workclass_Self-emp-not-inc', 'Workclass_State-gov',
    'Workclass_Without-pay', 'Marital Status_Divorced', 'Marital Status_Married',
    'Marital Status_Never-married', 'Occupation_Adm-clerical', 'Occupation_Armed-Forces',
    'Occupation_Craft-repair', 'Occupation_Exec-managerial', 'Occupation_Farming-fishing',
    'Occupation_Handlers-cleaners', 'Occupation_Machine-op-inspct', 'Occupation_Other-service',
    'Occupation_Priv-house-serv', 'Occupation_Prof-specialty', 'Occupation_Protective-serv',
    'Occupation_Sales', 'Occupation_Tech-support', 'Occupation_Transport-moving',
    'Relationship_Not-in-family', 'Relationship_Other-relative', 'Relationship_Own-child',
    'Relationship_Unmarried', 'Relationship_Wife'
]

# Mapping dan list untuk dropdown, agar konsisten dengan data training
education_map = {
    1: 'Preschool', 2: '1st-4th', 3: '5th-6th', 4: '7th-8th', 5: '9th', 6: '10th', 
    7: '11th', 8: '12th', 9: 'HS-grad', 10: 'Some-college', 11: 'Assoc-acdm',
    12: 'Assoc-voc', 13: 'Bachelors', 14: 'Masters', 15: 'Prof-school', 16: 'Doctorate'
}
EDUCATION_LIST = list(education_map.values())

# Daftar ini dibuat berdasarkan analisis `value_counts()` dari notebook
WORKCLASS_LIST = ['Private', 'Self-emp-not-inc', 'Local-gov', 'Other', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
MARITAL_STATUS_LIST = ['Married', 'Never-married', 'Divorced']
OCCUPATION_LIST = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
RELATIONSHIP_LIST = ['Wife', 'Own-child', 'Not-in-family', 'Unmarried', 'Other-relative'] # Husband sudah di-drop
RACE_LIST = ['White', 'Non-White']
NATIVE_COUNTRY_LIST = ['United-States', 'Non-United-States']

# --- Fungsi Prediksi ---

def predict(user_inputs):
    """Menerima input user, memprosesnya, dan mengembalikan prediksi."""
    
    # 1. Buat dictionary kosong untuk menampung data yang akan diprediksi
    input_data = {col: 0 for col in MODEL_COLUMNS}

    # 2. Isi dictionary dengan nilai dari input user
    input_data['Age'] = user_inputs['Age']
    input_data['Final Weight'] = user_inputs['Final Weight']
    input_data['Hours per Week'] = user_inputs['Hours per Week']
    
    # Konversi 'Education' (string) ke 'EducationNum' (integer)
    education_reverse_map = {v: k for k, v in education_map.items()}
    input_data['EducationNum'] = education_reverse_map[user_inputs['Education']]
    
    # Konversi jawaban biner (Ya/Tidak)
    input_data['Capital Gain'] = 1 if user_inputs['Capital Gain?'] == "Yes" else 0
    input_data['capital loss'] = 1 if user_inputs['Capital Loss?'] == "Yes" else 0
    input_data['Race_White'] = 1 if user_inputs['Race'] == "White" else 0
    input_data['Gender_Male'] = 1 if user_inputs['Gender'] == "Male" else 0
    input_data['Native Country_USA'] = 1 if user_inputs['Native Country'] == "United-States" else 0

    # 3. Handle One-Hot Encoding berdasarkan pilihan user
    # Untuk setiap kategori, cari kolom yang sesuai dan set nilainya menjadi 1
    input_data[f"Workclass_{user_inputs['Work Class']}"] = 1
    input_data[f"Marital Status_{user_inputs['Marital Status']}"] = 1
    input_data[f"Occupation_{user_inputs['Occupation']}"] = 1
    input_data[f"Relationship_{user_inputs['Relationship']}"] = 1

    # 4. Buat DataFrame dengan urutan kolom yang benar
    final_df = pd.DataFrame([input_data])
    final_df = final_df[MODEL_COLUMNS] # Memastikan urutan kolom 100% benar
    
    # 5. Lakukan prediksi
    prediction = XGBoost_Model.predict(final_df)
    probability = XGBoost_Model.predict_proba(final_df)
    
    return prediction[0], probability

# --- Main App ---

def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    
    stc.html(html_temp, height=150)
    
    if XGBoost_Model is None:
        return # Hentikan eksekusi jika model gagal dimuat

    # Kumpulkan input dalam sebuah dictionary untuk kemudahan
    user_inputs = {}

    col1, col2 = st.columns(2)
    with col1:
        user_inputs['Age'] = st.number_input("Age", min_value=17, max_value=90, value=38, help="Usia individu (17-90).")
        user_inputs['Education'] = st.selectbox("Education Level", EDUCATION_LIST, index=EDUCATION_LIST.index('Bachelors'))
        user_inputs['Work Class'] = st.selectbox("Work Class", WORKCLASS_LIST)
        user_inputs['Capital Gain?'] = st.radio("Capital Gain?", ("No", "Yes"), horizontal=True)
        user_inputs['Race'] = st.selectbox("Race", RACE_LIST)
        user_inputs['Native Country'] = st.selectbox("Native Country", NATIVE_COUNTRY_LIST)

    with col2:
        user_inputs['Final Weight'] = st.number_input("Final Weight (fnlwgt)", min_value=12285, max_value=1490400, value=189663, help="Bobot statistik dari sensus.")
        user_inputs['Hours per Week'] = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
        user_inputs['Occupation'] = st.selectbox("Occupation", OCCUPATION_LIST)
        user_inputs['Capital Loss?'] = st.radio("Capital Loss?", ("No", "Yes"), horizontal=True)
        user_inputs['Gender'] = st.selectbox("Gender", ("Male", "Female"))
        user_inputs['Marital Status'] = st.selectbox("Marital Status", MARITAL_STATUS_LIST)
    
    user_inputs['Relationship'] = st.selectbox("Relationship", RELATIONSHIP_LIST, help="Hubungan individu dalam keluarga.")
    
    st.write("---")
    
    if st.button("**Predict Income Category**", use_container_width=True):
        prediction, probability = predict(user_inputs)
        result_text = ">50K" if prediction == 1 else "<=50K"
        
        if prediction == 1:
            st.success(f"### Prediction: Income is likely **>50K**")
            st.info(f"Confidence: **{probability[0][1]*100:.2f}%**")
        else:
            st.warning(f"### Prediction: Income is likely **<=50K**")
            st.info(f"Confidence: **{probability[0][0]*100:.2f}%**")

if __name__ == "__main__":
    main()
