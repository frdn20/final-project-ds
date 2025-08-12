import streamlit as st
import streamlit.components.v1 as stc
import xgboost
import pandas as pd
import numpy as np
import imblearn
import cloudpickle

with open("income_prediction_pipeline.pkl","rb") as f :
  XGBoost_Model = cloudpickle.load(f)

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Loan Eligibility Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">Made for: Credit Team</h4> 
                """

desc_temp = """ ### Income Category Prediction App 
                This app is used by 
                
                #### Data Source
                Kaggle: Link <Masukkan Link>
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

education_map = {
    1: 'Preschool', 2: '1st-4th grade',3: '5th-6th grade', 4: '7th-8th grade', 5: '9th grade',
    6: '10th grade',7: '11th grade',8: '12th grade',9: 'HS-grad',10: 'Some college',11: 'Assoc-acdm',
    12: 'Assoc-voc', 13: 'Bachelors degree', 14: 'Masters degree', 15: 'Prof-school',16: 'Doctorate degree'}
workclass = ['Federal-gov', 'Local-gov','Never-worked', 'Other', 'Private',
             'Self-emp-inc', 'Self-emp-not-inc','State-gov', 'Without-pay']
occupation = ['Adm-clerical','Armed-Forces', 'Craft-repair','Exec-managerial', 'Farming-fishing',
              'Handlers-cleaners', 'Machine-op-inspct','Other-service', 'Priv-house-serv',
              'Prof-specialty', 'Protective-serv','Sales', 'Tech-support','Transport-moving']
relationship = ['Relationship_Not-in-family',
       'Relationship_Other-relative', 'Relationship_Own-child',
       'Relationship_Unmarried', 'Relationship_Wife']

def run_ml_app():
    design = """<div style="padding:15px;">
                    <h1 style="color:#000">Income Category Prediction</h1>
                </div
             """
    st.markdown(design, unsafe_allow_html=True)
    Education_List = list(education_map.values())
    left, right = st.columns((2,2))
    Age = left.number_input("Age", min_value = 1, max_value = 100)
    Education = right.selectbox("Education", Education_List)
    Capital_Gain = left.selectbox("Capital Gain", ("Yes","No"))
    Capital_Loss = right.selectbox("Capital Loss", ("Yes","No"))
    Hours_per_Week = left.number_input("Hours per Week", min_value = 0, max_value = 100)
    Work_Class = right.selectbox("Work Class", workclass)
    Marital_Status = left.selectbox("Marital Status", ("Married","Never-married","Divorced"))
    Occupation = right.selectbox("Occupation", occupation)
    Relationship = left.selectbox("Relationship", [s.replace('Relationship_', '') for s in relationship])
    Race = right.selectbox("Race", ("White","Non White"))
    Gender = left.selectbox("Gender", ("Male","Female"))
    Native_Country = right.selectbox("Native Country", ("United State","Non United State"))
    Final_Weight = left.number_input("Final Weight", min_value = 12000, max_value = 1400000)
    button = st.button("Predict")
    
    #If button is clilcked
    if button:
        result = predict(capital_gain, capital_loss, race, gender, 
                         native_country, age, final_weight, educationnum,
                         hours_per_week, workclass_1, workclass_2, workclass_3, 
                         workclass_4, workclass_5, workclass_6, workclass_7,
                         workclass_8, workclass_9, marital_1, marital_2, 
                         marital_3, occupation_1, occupation_2, occupation_3,
                         occupation_4, occupation_5, occupation_6, occupation_7, 
                         occupation_8, occupation_9, occupation_10, occupation_11,
                         occupation_12, occupation_13, occupation_14, relationship_1, 
                         relationship_2, relationship_3, relationship_4, relationship_5)
        
def predict(capital_gain, capital_loss, race, gender, 
            native_country, age, final_weight, educationnum,
            hours_per_week, workclass_1, workclass_2, workclass_3, 
            workclass_4, workclass_5, workclass_6, workclass_7,
            workclass_8, workclass_9, marital_1, marital_2, 
            marital_3, occupation_1, occupation_2, occupation_3,
            occupation_4, occupation_5, occupation_6, occupation_7, 
            occupation_8, occupation_9, occupation_10, occupation_11,
            occupation_12, occupation_13, occupation_14, relationship_1, 
            relationship_2, relationship_3, relationship_4, relationship_5):

    #Preprocessing User Input
    capital_gain = 1 if Capital_Gain == "Yes" else 0
    capital_loss = 1 if Capital_Loss == "Yes" else 0
    race = 1 if Race == "White" else 0
    gender = 1 if Gender == "Male" else 0
    education_reverse_map = {v: k for k, v in education_map.items()}
    educationnum = education_reverse_map[Education]
    marital_1 = 1 if Marital_Status == "Divorced" else 0         
    marital_2 = 1 if Marital_Status == "Married" else 0
    marital_3 = 1 if Marital_Status == "Never-married" else 0
    for i, cat in enumerate(workclass, start=1):
        locals()[f"workclass_{i}"] = 1 if Work_Class == cat else 0
    for i, cat in enumerate(occupation, start=1):
        locals()[f"occupation_{i}"] = 1 if Occupation == cat else 0
    for i, cat in enumerate(relationship, start=1):
        locals()[f"relationship_{i}"] = 1 if Relationship == cat else 0
    native_country = 1 if Native_Country == "United State" else 0
    
    #Making prediction
    prediction = XGBoost_Model.predict([[capital_gain, capital_loss, race, gender, 
            native_country, age, final_weight, educationnum,
            hours_per_week, workclass_1, workclass_2, workclass_3, 
            workclass_4, workclass_5, workclass_6, workclass_7,
            workclass_8, workclass_9, marital_1, marital_2, 
            marital_3, occupation_1, occupation_2, occupation_3,
            occupation_4, occupation_5, occupation_6, occupation_7, 
            occupation_8, occupation_9, occupation_10, occupation_11,
            occupation_12, occupation_13, occupation_14, relationship_1, 
            relationship_2, relationship_3, relationship_4, relationship_5]])
    result = "<=50K" if prediction == 0 else ">50K"
    return result

if __name__ == "__main__":

    main()
