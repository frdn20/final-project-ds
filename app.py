import streamlit as st
import streamlit.components.v1 as stc
import xgboost
import pandas as pd
import numpy as np
import imblearn
import cloudpickle

# --- Model Loading ---
# Pastikan file XGBoost_Model.pkl ada di folder yang sama
with open("income_prediction_pipeline.pkl", "rb") as f:
    XGBoost_Model = cloudpickle.load(f)

# --- UI Templates ---
html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Income Category Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">Credit Team Decision Support</h4> 
                </div>"""

desc_temp = """### Income Category Prediction App 
                This app predicts whether an individual's income is <=50K or >50K.
                
                #### Data Source
                Kaggle: [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)
                """

# --- Feature Lists and Mappings (Sesuai dengan notebook) ---
education_map = {
    1: 'Preschool', 2: '1st-4th', 3: '5th-6th', 4: '7th-8th', 5: '9th',
    6: '10th', 7: '11th', 8: '12th', 9: 'HS-grad', 10: 'Some-college', 11: 'Assoc-acdm',
    12: 'Assoc-voc', 13: 'Bachelors', 14: 'Masters', 15: 'Prof-school', 16: 'Doctorate'
}

# Daftar kolom ini HARUS SAMA PERSIS dengan yang digunakan saat training
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

# --- Prediction Function (DIUBAH TOTAL) ---
def predict(age, final_weight, education, capital_gain_str, capital_loss_str, 
            hours_per_week, work_class, marital_status, occupation, 
            relationship, race, gender, native_country):
    
    # 1. Preprocessing Input dari User
    education_reverse_map = {v: k for k, v in education_map.items()}
    educationnum = education_reverse_map[education]

    # Ubah input string menjadi format numerik/biner
    input_data = {
        'Capital Gain': 1 if capital_gain_str == "Yes" else 0,
        'capital loss': 1 if capital_loss_str == "Yes" else 0,
        'Race_White': 1 if race == "White" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Native Country_USA': 1 if native_country == "United-States" else 0,
        'Age': age,
        'Final Weight': final_weight,
        'EducationNum': educationnum,
        'Hours per Week': hours_per_week
    }

    # 2. One-Hot Encoding untuk variabel kategorikal
    # Inisialisasi semua kolom yang mungkin dengan nilai 0
    for col in MODEL_COLUMNS:
        if col not in input_data:
            input_data[col] = 0

    # Set nilai 1 untuk kategori yang dipilih oleh user
    workclass_col = f"Workclass_{work_class}"
    if workclass_col in MODEL_COLUMNS:
        input_data[workclass_col] = 1

    marital_col = f"Marital Status_{marital_status}"
    if marital_col in MODEL_COLUMNS:
        input_data[marital_col] = 1

    occupation_col = f"Occupation_{occupation}"
    if occupation_col in MODEL_COLUMNS:
        input_data[occupation_col] = 1
        
    relationship_col = f"Relationship_{relationship}"
    if relationship_col in MODEL_COLUMNS:
        input_data[relationship_col] = 1

    # 3. Membuat DataFrame dengan urutan kolom yang benar
    # Ini adalah langkah paling krusial
    final_df = pd.DataFrame([input_data])
    final_df = final_df[MODEL_COLUMNS] # Memastikan urutan kolom 100% benar

    # 4. Membuat Prediksi
    prediction = XGBoost_Model.predict(final_df)
    probability = XGBoost_Model.predict_proba(final_df)

    result = ">50K" if prediction[0] == 1 else "<=50K"
    return result, probability

# --- Main App Logic (DIUBAH) ---
def run_ml_app():
    design = """<div style="padding:15px;">
                    <h1 style="color:#000">Income Category Prediction</h1>
                </div>"""
    st.markdown(design, unsafe_allow_html=True)

    # Mengambil daftar dari notebook agar konsisten
    Education_List = list(education_map.values())
    workclass_list = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked', 'Other']
    marital_status_list = ['Married', 'Never-married', 'Divorced'] # Sesuai penyederhanaan di notebook
    occupation_list = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
    relationship_list = ['Wife', 'Own-child', 'Not-in-family', 'Unmarried', 'Other-relative'] # Husband sudah di-drop
    race_list = ['White', 'Non-White']
    native_country_list = ['United-States', 'Non-United-States']

    left, right = st.columns((2, 2))
    
    # Kumpulkan semua input dari user
    Age = left.number_input("Age", min_value=17, max_value=90, value=38)
    Final_Weight = right.number_input("Final Weight", min_value=12285, max_value=1490400, value=189663)
    Education = left.selectbox("Education", Education_List, index=Education_List.index('Bachelors'))
    Hours_per_Week = right.number_input("Hours per Week", min_value=1, max_value=99, value=40)
    Capital_Gain_str = left.selectbox("Capital Gain?", ("No", "Yes"))
    Capital_Loss_str = right.selectbox("Capital Loss?", ("No", "Yes"))
    Work_Class = left.selectbox("Work Class", workclass_list)
    Marital_Status = right.selectbox("Marital Status", marital_status_list)
    Occupation = left.selectbox("Occupation", occupation_list)
    Relationship = right.selectbox("Relationship", relationship_list)
    Race = left.selectbox("Race", race_list)
    Gender = right.selectbox("Gender", ("Male", "Female"))
    Native_Country = left.selectbox("Native Country", native_country_list)

    button = st.button("Predict")

    if button:
        # Panggil fungsi predict dengan semua input mentah dari user
        result, probability = predict(
            age=Age, final_weight=Final_Weight, education=Education, capital_gain_str=Capital_Gain_str,
            capital_loss_str=Capital_Loss_str, hours_per_week=Hours_per_Week, work_class=Work_Class,
            marital_status=Marital_Status, occupation=Occupation, relationship=Relationship,
            race=Race, gender=Gender, native_country=Native_Country
        )
        
        st.success(f"**Prediction Result: The individual's income is likely {result}**")
        if result == ">50K":
            st.info(f"Probability of income >50K: **{probability[0][1]*100:.2f}%**")
        else:
            st.info(f"Probability of income <=50K: **{probability[0][0]*100:.2f}%**")


def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()


if __name__ == "__main__":
    main()
