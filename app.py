import json
import numpy as np
import pandas as pd
import streamlit as st
import pickle

# ====== LOAD MODEL & FEATURE LIST ======
with open('income_prediction_pipeline.pkl', 'rb') as f:  # atau file model yang bener-bener lu simpan
    pipeline = pickle.load(f)

# Hardcode dulu kalau belum save json (lebih proper: load dari file)
COLUMNS_TRAIN = [
    'Capital Gain','capital loss','Race_White','Gender_Male','Native Country_USA',
    'Age','Final Weight','EducationNum','Hours per Week',
    'Workclass_Federal-gov','Workclass_Local-gov','Workclass_Never-worked','Workclass_Other',
    'Workclass_Private','Workclass_Self-emp-inc','Workclass_Self-emp-not-inc','Workclass_State-gov',
    'Workclass_Without-pay',
    'Marital Status_Divorced','Marital Status_Married','Marital Status_Never-married',
    'Occupation_Adm-clerical','Occupation_Armed-Forces','Occupation_Craft-repair',
    'Occupation_Exec-managerial','Occupation_Farming-fishing','Occupation_Handlers-cleaners',
    'Occupation_Machine-op-inspct','Occupation_Other-service','Occupation_Priv-house-serv',
    'Occupation_Prof-specialty','Occupation_Protective-serv','Occupation_Sales',
    'Occupation_Tech-support','Occupation_Transport-moving',
    'Relationship_Not-in-family','Relationship_Other-relative','Relationship_Own-child',
    'Relationship_Unmarried','Relationship_Wife'
]

def transform_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = {}

    # numeric (apa adanya)
    out['Age'] = df_raw['Age']
    out['Final Weight'] = df_raw['Final Weight']
    out['EducationNum'] = df_raw['EducationNum']
    out['Hours per Week'] = df_raw['Hours per Week']

    # binary engineered sama kayak training
    out['Capital Gain'] = (df_raw['Capital Gain'] > 0).astype(int)
    out['capital loss'] = (df_raw['capital loss'] > 0).astype(int)
    out['Race_White'] = (df_raw['Race'] == 'White').astype(int)
    out['Gender_Male'] = (df_raw['Gender'] == 'Male').astype(int)
    out['Native Country_USA'] = (df_raw['Native Country'] == 'USA').astype(int)

    # one-hot manual dgn set kategori fix
    def one_hot(col_name, value, categories, prefix):
        for c in categories:
            out[f'{prefix}_{c}'] = (value == c).astype(int)

    # Workclass
    workclass_cats = ['Federal-gov','Local-gov','Never-worked','Other','Private',
                      'Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay']
    one_hot('Workclass', df_raw['Workclass'], workclass_cats, 'Workclass')

    # Marital Status
    marital_cats = ['Divorced','Married','Never-married']
    one_hot('Marital Status', df_raw['Marital Status'], marital_cats, 'Marital Status')

    # Occupation (harus sama persis)
    occupation_cats = ['Adm-clerical','Armed-Forces','Craft-repair','Exec-managerial','Farming-fishing',
                       'Handlers-cleaners','Machine-op-inspct','Other-service','Priv-house-serv',
                       'Prof-specialty','Protective-serv','Sales','Tech-support','Transport-moving']
    one_hot('Occupation', df_raw['Occupation'], occupation_cats, 'Occupation')

    # Relationship (tanpa Husband karena di-drop saat training)
    relationship_cats = ['Not-in-family','Other-relative','Own-child','Unmarried','Wife']
    one_hot('Relationship', df_raw['Relationship'], relationship_cats, 'Relationship')

    X = pd.DataFrame(out)
    # pastikan urutan/kolom sama
    X = X.reindex(columns=COLUMNS_TRAIN, fill_value=0)
    return X

# ==== form streamlit lu (input_df) tetap sama ====

if st.button('Prediksi'):
    X = transform_input(input_df)  # <<— pakai transform
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    # … lanjut tampilan hasil
