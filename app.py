import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import cloudpickle
from pathlib import Path

# ---------------------
# Load trained pipeline
# ---------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str = "income_prediction_pipeline.pkl"):
    p = Path(path)
    if not p.exists():
        st.error(f"Model file not found: {p.resolve()}")
        st.stop()
    with open(p, "rb") as f:
        return cloudpickle.load(f)

model = load_model()

# ---------------------
# Constants (match training)
# ---------------------
education_map = {
    1: 'Preschool', 2: '1st-4th grade', 3: '5th-6th grade', 4: '7th-8th grade', 5: '9th grade',
    6: '10th grade', 7: '11th grade', 8: '12th grade', 9: 'HS-grad', 10: 'Some college', 11: 'Assoc-acdm',
    12: 'Assoc-voc', 13: 'Bachelors degree', 14: 'Masters degree', 15: 'Prof-school', 16: 'Doctorate degree'
}

workclass_cats = [
    'Federal-gov', 'Local-gov', 'Never-worked', 'Other', 'Private',
    'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'
]

occupation_cats = [
    'Adm-clerical','Armed-Forces','Craft-repair','Exec-managerial','Farming-fishing',
    'Handlers-cleaners','Machine-op-inspct','Other-service','Priv-house-serv',
    'Prof-specialty','Protective-serv','Sales','Tech-support','Transport-moving'
]

relationship_cats = [
    'Not-in-family','Other-relative','Own-child','Unmarried','Wife'
]

# EXACT columns used during training (order matters!)
columns_train = [
    'Capital Gain', 'capital loss', 'Race_White', 'Gender_Male', 'Native Country_USA',
    'Age', 'Final Weight', 'EducationNum', 'Hours per Week',
    'Workclass_Federal-gov','Workclass_Local-gov','Workclass_Never-worked','Workclass_Other',
    'Workclass_Private','Workclass_Self-emp-inc','Workclass_Self-emp-not-inc','Workclass_State-gov','Workclass_Without-pay',
    'Marital Status_Divorced','Marital Status_Married','Marital Status_Never-married',
    'Occupation_Adm-clerical','Occupation_Armed-Forces','Occupation_Craft-repair','Occupation_Exec-managerial',
    'Occupation_Farming-fishing','Occupation_Handlers-cleaners','Occupation_Machine-op-inspct','Occupation_Other-service',
    'Occupation_Priv-house-serv','Occupation_Prof-specialty','Occupation_Protective-serv','Occupation_Sales',
    'Occupation_Tech-support','Occupation_Transport-moving',
    'Relationship_Not-in-family','Relationship_Other-relative','Relationship_Own-child','Relationship_Unmarried','Relationship_Wife'
]

# ---------------------
# Feature builder
# ---------------------
def make_feature_row(
    *,
    Age: int,
    Final_Weight: int,
    Education: str,
    Hours_per_Week: int,
    Capital_Gain_flag: str,
    Capital_Loss_flag: str,
    Work_Class: str,
    Marital_Status: str,
    Occupation: str,
    Relationship: str,
    Race: str,
    Gender: str,
    Native_Country: str,
) -> pd.DataFrame:
    # Binary encodings from UI
    capital_gain = 1 if Capital_Gain_flag == 'Yes' else 0
    capital_loss = 1 if Capital_Loss_flag == 'Yes' else 0
    race_white   = 1 if Race == 'White' else 0
    gender_male  = 1 if Gender == 'Male' else 0
    native_usa   = 1 if Native_Country == 'United State' else 0  # UI label → training col

    # Map Education string → EducationNum as used in training
    reverse_edu = {v: k for k, v in education_map.items()}
    educationnum = reverse_edu[Education]

    # One-hot for multi-class features (match training names)
    workclass_flags = {f"Workclass_{wc}": int(Work_Class == wc) for wc in workclass_cats}
    occupation_flags = {f"Occupation_{oc}": int(Occupation == oc) for oc in occupation_cats}
    relationship_flags = {f"Relationship_{r}": int(Relationship == r) for r in relationship_cats}

    row = {
        'Capital Gain': capital_gain,
        'capital loss': capital_loss,
        'Race_White': race_white,
        'Gender_Male': gender_male,
        'Native Country_USA': native_usa,
        'Age': Age,
        'Final Weight': Final_Weight,
        'EducationNum': educationnum,
        'Hours per Week': Hours_per_Week,
        **workclass_flags,
        **occupation_flags,
        **relationship_flags,
    }

    # Ensure full schema & order
    X = pd.DataFrame([{c: row.get(c, 0) for c in columns_train}], columns=columns_train)
    return X

# ---------------------
# Predictor
# ---------------------

def predict_income(df_row: pd.DataFrame) -> str:
    pred = model.predict(df_row)[0]
    # Some frameworks return numpy scalar; ensure proper comparison
    pred_int = int(pred)
    return ">50K" if pred_int == 1 else "<=50K"

# ---------------------
# UI
# ---------------------
html_header = (
    "<div style=\"background-color:#111;padding:12px;border-radius:12px\">"
    "<h1 style=\"color:#fff;text-align:center;margin:0\">Income Category Prediction</h1>"
    "<p style=\"color:#aaa;text-align:center;margin:4px 0 0\">Adult Census Income — XGBoost pipeline</p>"
    "</div>"
)

def main():
    stc.html(html_header)
    menu = ["Predict"]
    _ = st.sidebar.selectbox("Menu", menu)

    # Form inputs
    col1, col2 = st.columns(2)
    Age = col1.number_input("Age", min_value=1, max_value=100, value=30)
    Education = col2.selectbox("Education", list(education_map.values()), index=12)  # default Bachelors
    Capital_Gain_flag = col1.selectbox("Capital Gain present?", ("No", "Yes"))
    Capital_Loss_flag = col2.selectbox("Capital Loss present?", ("No", "Yes"))
    Hours_per_Week = col1.number_input("Hours per Week", min_value=0, max_value=100, value=40)
    Final_Weight = col2.number_input("Final Weight", min_value=12000, max_value=1400000, value=200000)

    Work_Class = col1.selectbox("Workclass", workclass_cats)
    Marital_Status = col2.selectbox("Marital Status", ("Married", "Never-married", "Divorced"))

    Occupation = col1.selectbox("Occupation", occupation_cats)
    Relationship = col2.selectbox("Relationship", relationship_cats)

    Race = col1.selectbox("Race", ("White", "Non White"))
    Gender = col2.selectbox("Gender", ("Male", "Female"))
    Native_Country = col1.selectbox("Native Country", ("United State", "Non United State"))

    if st.button("Predict"):
        X = make_feature_row(
            Age=Age,
            Final_Weight=Final_Weight,
            Education=Education,
            Hours_per_Week=Hours_per_Week,
            Capital_Gain_flag=Capital_Gain_flag,
            Capital_Loss_flag=Capital_Loss_flag,
            Work_Class=Work_Class,
            Marital_Status=Marital_Status,
            Occupation=Occupation,
            Relationship=Relationship,
            Race=Race,
            Gender=Gender,
            Native_Country=Native_Country,
        )
        pred = predict_income(X)
        st.success(f"Prediksi Income: {pred}")
        with st.expander("Lihat fitur yang dikirim ke model"):
            st.dataframe(X)


if __name__ == "__main__":
    main()
