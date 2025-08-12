import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# --- 1. Ganti nama variabel pipeline agar lebih jelas ---
with open("income_prediction_pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

# --- FUNGSI UTAMA (TIDAK PERLU DIUBAH) ---
def main():
    st.set_page_config(page_title="Income Prediction", layout="wide")
    
    html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                   <h1 style="color:#fff;text-align:center">Income Category Prediction App</h1> 
                   </div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    run_ml_app()

# --- DAFTAR & MAPPING (TIDAK PERLU DIUBAH) ---
education_map = {
    1: 'Preschool', 2: '1st-4th', 3: '5th-6th', 4: '7th-8th', 5: '9th',
    6: '10th', 7: '11th', 8: '12th', 9: 'HS-grad', 10: 'Some-college', 11: 'Assoc-acdm',
    12: 'Assoc-voc', 13: 'Bachelors', 14: 'Masters', 15: 'Prof-school', 16: 'Doctorate'
}

workclass = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 
             'Self-emp-inc', 'Without-pay', 'Never-worked', 'Other']
             
marital_status = ['Married', 'Never-married', 'Divorced']

occupation = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 
              'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 
              'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 
              'Armed-Forces', 'Priv-house-serv']

relationship = ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']

race = ['White', 'Non-White']

gender = ['Male', 'Female']

native_country = ['USA', 'Non-USA']


# --- 2. REVISI TOTAL BAGIAN APLIKASI MACHINE LEARNING ---
def run_ml_app():
    st.subheader("Income Category Prediction")
    
    # --- Membuat kolom untuk layout ---
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=17, max_value=90, value=38)
        Work_Class = st.selectbox("Work Class", workclass)
        Final_Weight = st.number_input("Final Weight", min_value=12285, max_value=1490400, value=189663)
        Capital_Gain_Input = st.selectbox("Capital Gain", ("No", "Yes"))
        Hours_per_Week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
        Gender = st.selectbox("Gender", gender)
        
    with col2:
        # User memilih dari nama pendidikan, bukan angka
        Education = st.selectbox("Education", list(education_map.values()))
        Marital_Status = st.selectbox("Marital Status", marital_status)
        Occupation = st.selectbox("Occupation", occupation)
        Capital_Loss_Input = st.selectbox("Capital Loss", ("No", "Yes"))
        Relationship = st.selectbox("Relationship", relationship)
        Race = st.selectbox("Race", race)
        Native_Country = st.selectbox("Native Country", native_country)

    button = st.button("Predict")
    
    # Jika tombol ditekan
    if button:
        # --- 3. KUMPULKAN SEMUA INPUT MENJADI SATU DATAFRAME ---
        # Nama kolom HARUS SAMA PERSIS dengan saat training di notebook

        # Mapping input ke format yang benar (angka)
        education_reverse_map = {v: k for k, v in education_map.items()}
        education_num = education_reverse_map[Education]
        
        capital_gain = 1 if Capital_Gain_Input == "Yes" else 0
        capital_loss = 1 if Capital_Loss_Input == "Yes" else 0
        
        input_data = {
            'Age': [Age],
            'Workclass': [Work_Class],
            'Final Weight': [Final_Weight],
            'EducationNum': [education_num],
            'Marital Status': [Marital_Status],
            'Occupation': [Occupation],
            'Relationship': [Relationship],
            'Race': [Race],
            'Gender': [Gender],
            'Capital Gain': [capital_gain],
            'capital loss': [capital_loss],
            'Hours per Week': [Hours_per_Week],
            'Native Country': [Native_Country]
        }
        
        input_df = pd.DataFrame(input_data)
        
        st.write("### Input DataFrame:")
        st.dataframe(input_df)

        # --- 4. LAKUKAN PREDIKSI LANGSUNG DARI PIPELINE ---
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)
        
        # Tampilkan hasil
        if prediction[0] == 0:
            st.success("Prediksi: Penghasilan <=50K")
        else:
            st.warning("Prediksi: Penghasilan >50K")
            
        st.write("#### Probabilitas Prediksi:")
        st.write(f"Probabilitas <=50K: {prediction_proba[0][0]:.2%}")
        st.write(f"Probabilitas >50K: {prediction_proba[0][1]:.2%}")


if __name__ == "__main__":
    main()
