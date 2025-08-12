# app.py (REVISED)

import streamlit as st
import streamlit.components.v1 as stc
import xgboost
import pandas as pd
import numpy as np
import imblearn
import cloudpickle

# Muat model yang sudah di-train
with open("XGBoost_Model.pkl", "rb") as f:
    XGBoost_Model = cloudpickle.load(f)

# --- Tampilan UI (Tidak ada perubahan di sini) ---
html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Income Category Prediction App</h1>
                <h4 style="color:#fff;text-align:center">Final Project</h4>
                </div>"""

desc_temp = """ ### Income Category Prediction App
                This app predicts whether an individual's income is <=50K or >50K.

                #### Data Source
                UCI Machine Learning Repository: [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
                """

def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

# --- Definisi Variabel & Mapping (Tidak ada perubahan di sini) ---
education_map = {
    1: 'Preschool', 2: '1st-4th', 3: '5th-6th', 4: '7th-8th', 5: '9th',
    6: '10th', 7: '11th', 8: '12th', 9: 'HS-grad', 10: 'Some-college', 11: 'Assoc-acdm',
    12: 'Assoc-voc', 13: 'Bachelors', 14: 'Masters', 15: 'Prof-school', 16: 'Doctorate'
}
workclass_list = ['Federal-gov', 'Local-gov', 'Never-worked', 'Other', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
occupation_list = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
relationship_list = ['Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']

def run_ml_app():
    design = """<div style="padding:15px;">
                    <h1 style="color:#000">Income Category Prediction</h1>
                </div>"""
    st.markdown(design, unsafe_allow_html=True)
    
    # --- Input Widgets (Sedikit penyesuaian untuk konsistensi) ---
    left, right = st.columns((2,2))
    
    age = left.number_input("Age", min_value=17, max_value=100)
    education_str = right.selectbox("Education", list(education_map.values()))
    capital_gain_str = left.selectbox("Capital Gain", ("No", "Yes"))
    capital_loss_str = right.selectbox("Capital Loss", ("No", "Yes"))
    hours_per_week = left.number_input("Hours per Week", min_value=1, max_value=99)
    work_class_str = right.selectbox("Work Class", workclass_list)
    marital_status_str = left.selectbox("Marital Status", ("Married", "Never-married", "Divorced"))
    occupation_str = right.selectbox("Occupation", occupation_list)
    relationship_str = left.selectbox("Relationship", relationship_list)
    race_str = right.selectbox("Race", ("White", "Non-White"))
    gender_str = left.selectbox("Gender", ("Male", "Female"))
    native_country_str = right.selectbox("Native Country", ("USA", "Non-USA"))
    final_weight = left.number_input("Final Weight", min_value=12000, max_value=1500000)

    button = st.button("Predict")

    if button:
        # **PERBAIKAN KUNCI 1: Panggil predict dengan variabel mentah dari widget**
        result_prediction = predict(age, education_str, capital_gain_str, capital_loss_str, hours_per_week,
                                    work_class_str, marital_status_str, occupation_str, relationship_str,
                                    race_str, gender_str, native_country_str, final_weight)
        
        if result_prediction == 1:
            st.success("Prediction: Income is >50K")
        else:
            st.info("Prediction: Income is <=50K")


# **PERBAIKAN KUNCI 2: Ubah fungsi predict untuk melakukan semua preprocessing**
def predict(age, education, capital_gain, capital_loss, hours_per_week, workclass,
            marital_status, occupation, relationship, race, gender, native_country, final_weight):

    # --- Preprocessing Input Pengguna ---
    
    # Mapping Education ke EducationNum
    education_reverse_map = {v: k for k, v in education_map.items()}
    education_num = education_reverse_map[education]

    # Konversi input biner
    capital_gain_bin = 1 if capital_gain == "Yes" else 0
    capital_loss_bin = 1 if capital_loss == "Yes" else 0
    race_white = 1 if race == "White" else 0
    gender_male = 1 if gender == "Male" else 0
    native_country_usa = 1 if native_country == "USA" else 0

    # Dapatkan daftar kolom yang sama persis dengan saat training (dari notebook cell [86])
    # Urutan ini SANGAT PENTING
    training_columns = [
        'Capital Gain', 'capital loss', 'Race_White', 'Gender_Male', 'Native Country_USA',
        'Age', 'Final Weight', 'EducationNum', 'Hours per Week', 'Workclass_Federal-gov',
        'Workclass_Local-gov', 'Workclass_Never-worked', 'Workclass_Other', 'Workclass_Private',
        'Workclass_Self-emp-inc', 'Workclass_Self-emp-not-inc', 'Workclass_State-gov',
        'Workclass_Without-pay', 'Marital Status_Divorced', 'Marital Status_Married',
        'Marital Status_Never-married', 'Occupation_Adm-clerical', 'Occupation_Armed Forces',
        'Occupation_Craft-repair', 'Occupation_Exec-managerial', 'Occupation_Farming-fishing',
        'Occupation_Handlers-cleaners', 'Occupation_Machine-op-inspct', 'Occupation_Other-service',
        'Occupation_Priv-house-serv', 'Occupation_Prof-specialty', 'Occupation_Protective-serv',
        'Occupation_Sales', 'Occupation_Tech-support', 'Occupation_Transport-moving',
        'Relationship_Not-in-family', 'Relationship_Other-relative', 'Relationship_Own-child',
        'Relationship_Unmarried', 'Relationship_Wife'
    ]
    
    # Buat dictionary untuk input data
    input_data = {col: [0] for col in training_columns} # Inisialisasi semua dengan 0

    # Isi nilai-nilai sesuai input pengguna
    input_data['Age'] = [age]
    input_data['Final Weight'] = [final_weight]
    input_data['EducationNum'] = [education_num]
    input_data['Hours per Week'] = [hours_per_week]
    input_data['Capital Gain'] = [capital_gain_bin]
    input_data['capital loss'] = [capital_loss_bin]
    input_data['Race_White'] = [race_white]
    input_data['Gender_Male'] = [gender_male]
    input_data['Native Country_USA'] = [native_country_usa]

    # One-hot encode untuk fitur kategorikal
    if f'Workclass_{workclass}' in input_data:
        input_data[f'Workclass_{workclass}'][0] = 1
    if f'Marital Status_{marital_status}' in input_data:
        input_data[f'Marital Status_{marital_status}'][0] = 1
    if f'Occupation_{occupation}' in input_data:
        input_data[f'Occupation_{occupation}'][0] = 1
    if f'Relationship_{relationship}' in input_data:
        input_data[f'Relationship_{relationship}'][0] = 1

    # Buat DataFrame dari dictionary
    input_df = pd.DataFrame.from_dict(input_data)
    input_df = input_df[training_columns] # Pastikan urutan kolom sudah benar

    # --- Lakukan Prediksi ---
    prediction = XGBoost_Model.predict(input_df)
    
    return prediction[0]


if __name__ == "__main__":
    main()
