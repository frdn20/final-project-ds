import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import pickle
import xgboost as xgb

# Load model
with open('XGBoost_Model.pkl', 'rb') as file:
    XGBoost_Model = pickle.load(file)

# booster = best_model_pipeline_model_xgb.get_booster()
# booster.save_model("model.json")

# booster = xgb.Booster()
# booster.load_model("model.json")

# HTML header
html_temp = """
<style>
.header-container {
    background-color: #A3CEF1; /* warna soft pastel */
    padding: 20px;
    width: 100vw; /* lebar layar */
    height: 200px;
    margin-left: -3rem; 
    margin-right: -3rem;
    border-radius: 0px;
    box-sizing: border-box;
    text-align: center;
}
.header-container h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
}
.header-container h4 {
    color: white;
    margin: 0;
    font-weight: normal;
}
</style>

<div class="header-container">
    <h1>Census Income Prediction</h1>
    <h4>Predict whether income >50K or <=50K</h4>
</div>
"""

# """
# <div style="background-color:#A3CEF1; padding:15px; border-radius:0px; width:100vw; height:120px; margin-left:-3rem; margin-right:-3rem;">
#     <h1 style="color:#fff; text-align:center; margin:0;">Census Income Prediction</h1> 
#     <h4 style="color:#fff; text-align:center; margin:0;">Predict whether income >50K or <=50K</h4> 
# </div>
# """
desc_temp = """
### About  
This app predicts whether an individual's income is **>50K** or **<=50K** based on US Census data.  

#### Data Source
Kaggle: [Census Income Analysis and Modeling](https://www.kaggle.com/code/tawfikelmetwally/census-income-analysis-and-modeling/input)  
"""

# ML app page
def run_ml_app():
    st.markdown("### Fill in the details below to get prediction")

    col1, col2 = st.columns(2)

    # Numeric inputs
    age = col1.number_input("Age", min_value=17, max_value=90, value=30)
    final_weight = col2.number_input("Final Weight", min_value=10000, max_value=1500000, value=200000)
    education_num = col1.number_input("Education Number", min_value=1, max_value=16, value=10)
    hours_per_week = col2.number_input("Hours per Week", min_value=1, max_value=99, value=40)

    # Binary inputs
    capital_gain = col1.selectbox("Capital Gain", ["Yes", "No"])
    capital_loss = col2.selectbox("Capital Loss", ["Yes", "No"])

    # Categorical inputs
    race = col1.selectbox("Race", ["White", "Non White"])
    gender = col2.selectbox("Gender", ["Male", "Female"])
    native_country = col1.selectbox("Native Country", ["United-States", "Other"])

    workclass_options = ['Federal-gov', 'Local-gov', 'Never-worked', 'Other',
                         'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                         'State-gov', 'Without-pay']
    workclass = col2.selectbox("Workclass", workclass_options)

    marital_options = ['Divorced', 'Married', 'Never-married']
    marital_status = col1.selectbox("Marital Status", marital_options)

    occupation_options = ['Adm-clerical', 'Armed-Forces', 'Craft-repair',
                          'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners',
                          'Machine-op-inspct', 'Other-service', 'Priv-house-serv',
                          'Prof-specialty', 'Protective-serv', 'Sales',
                          'Tech-support', 'Transport-moving']
    occupation = col2.selectbox("Occupation", occupation_options)

    relationship_options = ['Not-in-family', 'Other-relative', 'Own-child',
                            'Unmarried', 'Wife']
    relationship = col1.selectbox("Relationship", relationship_options)

    # Predict button
    if st.button("Predict"):
        prediction = predict_income(capital_gain, capital_loss, race, gender, native_country,
                                    age, final_weight, education_num, hours_per_week,
                                    workclass, marital_status, occupation, relationship)

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("✅ Predicted Income: >50K")
        else:
            st.error("⚠️ Predicted Income: <=50K")

# Function untuk proses input and predict
def predict_income(capital_gain, capital_loss, race, gender, native_country,
                   age, final_weight, education_num, hours_per_week,
                   workclass, marital_status, occupation, relationship):
    # Binary features
    capital_gain_val = 1 if capital_gain == "Yes" else 0
    capital_loss_val = 1 if capital_loss == "Yes" else 0
    race_white = 1 if race == "White" else 0
    gender_male = 1 if gender == "Male" else 0
    native_country_usa = 1 if native_country == "United-States" else 0
    # Workclass
    workclass_options = ['Federal-gov', 'Local-gov', 'Never-worked', 'Other',
                         'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                         'State-gov', 'Without-pay']
    workclass_encoded = [1 if workclass == wc else 0 for wc in workclass_options]
    # Marital Status
    marital_options = ['Divorced', 'Married', 'Never-married']
    marital_encoded = [1 if marital_status == m else 0 for m in marital_options]
    # Occupation
    occupation_options = ['Adm-clerical', 'Armed-Forces', 'Craft-repair',
                          'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners',
                          'Machine-op-inspct', 'Other-service', 'Priv-house-serv',
                          'Prof-specialty', 'Protective-serv', 'Sales',
                          'Tech-support', 'Transport-moving']
    occupation_encoded = [1 if occupation == o else 0 for o in occupation_options]
    # Relationship
    relationship_options = ['Not-in-family', 'Other-relative', 'Own-child',
                            'Unmarried', 'Wife']
    relationship_encoded = [1 if relationship == r else 0 for r in relationship_options]
    
    # Susun sesuai feature_names_in_
    input_data = [[
        capital_gain_val, capital_loss_val, race_white, gender_male,
        native_country_usa, age, final_weight, education_num, hours_per_week,
        *workclass_encoded, *marital_encoded, *occupation_encoded, *relationship_encoded
    ]]

    input_df = pd.DataFrame(input_data, columns=best_model_pipeline_model_xgb.feature_names_in_)

    prediction = best_model_pipeline_model_xgb.predict(input_df)[0]
    return prediction

# Main app
def main():
    stc.html(html_temp)
    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Prediction":
        run_ml_app()

if __name__ == "__main__":
    main()

