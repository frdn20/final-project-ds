import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import cloudpickle  # Gunakan cloudpickle jika model Anda kompleks

# ====================================================================
# 1. MUAT PIPELINE, BUKAN HANYA MODEL
# Pastikan Anda sudah menyimpan pipeline yang lengkap (preprocessor + model)
# seperti yang kita diskusikan di jawaban sebelumnya.
# ====================================================================
try:
    with open("income_prediction_pipeline.pkl", "rb") as f:
        pipeline = cloudpickle.load(f)
except FileNotFoundError:
    st.error("Model pipeline 'income_prediction_pipeline_v2.pkl' tidak ditemukan. Pastikan file sudah ada di folder yang sama dengan app.py")
    st.stop()


# Peta untuk mengubah nama tampilan pendidikan menjadi angka
education_map = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
    'Assoc-acdm': 11, 'Assoc-voc': 12, 'Bachelors': 13, 'Masters': 14,
    'Prof-school': 15, 'Doctorate': 16
}

# Daftar kategori untuk dropdown (ini sudah bagus dari kode Anda)
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked', 'Other']
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
relationship_options = ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']


def run_ml_app():
    """Fungsi utama untuk menjalankan antarmuka aplikasi ML."""
    
    st.subheader("Prediksi Kategori Pendapatan 📊")

    # Layout kolom untuk input
    left, right = st.columns(2)

    # Mengumpulkan input dari pengguna
    age = left.number_input("Usia", min_value=17, max_value=90, value=37)
    workclass = right.selectbox("Kelas Pekerjaan", options=workclass_options)
    final_weight = left.number_input("Final Weight", min_value=12000, max_value=1500000, value=180000)
    
    # Ambil nilai EducationNum dari input selectbox
    education_label = right.selectbox("Pendidikan Terakhir", options=list(education_map.keys()))
    educationnum = education_map[education_label]
    
    marital_status = left.selectbox("Status Pernikahan", options=["Married", "Never-married", "Divorced"])
    occupation = right.selectbox("Pekerjaan", options=occupation_options)
    relationship = left.selectbox("Hubungan Keluarga", options=relationship_options)
    race = right.selectbox("Ras", options=["White", "Non-White"])
    gender = left.selectbox("Jenis Kelamin", options=["Male", "Female"])
    
    # Input Capital Gain/Loss dibuat biner agar sesuai dengan data training
    capital_gain = 1 if left.selectbox("Memiliki Capital Gain?", ("Ya", "Tidak")) == "Ya" else 0
    capital_loss = 1 if right.selectbox("Memiliki Capital Loss?", ("Ya", "Tidak")) == "Ya" else 0

    hours_per_week = left.number_input("Jam Kerja per Minggu", min_value=1, max_value=99, value=40)
    native_country = right.selectbox("Kewarganegaraan", options=["USA", "Non-USA"])
    
    # Tombol untuk memicu prediksi
    if st.button("🔮 Prediksi Sekarang"):
        # ====================================================================
        # 2. BUAT DATAFRAME DARI INPUT PENGGUNA
        # Strukturnya harus sama persis dengan data mentah X_train Anda.
        # ====================================================================
        input_data = {
            'Age': [age],
            'Workclass': [workclass],
            'Final Weight': [final_weight],
            'EducationNum': [educationnum],
            'Marital Status': [marital_status],
            'Occupation': [occupation],
            'Relationship': [relationship],
            'Race': [race],
            'Gender': [gender],
            'Capital Gain': [capital_gain],
            'capital loss': [capital_loss],
            'Hours per Week': [hours_per_week],
            'Native Country': [native_country]
        }
        input_df = pd.DataFrame(input_data)

        # ====================================================================
        # 3. LAKUKAN PREDIKSI DENGAN SATU BARIS KODE!
        # Pipeline akan secara otomatis melakukan scaling dan encoding.
        # ====================================================================
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)

        # Tampilkan hasilnya
        st.success("Hasil Prediksi:")
        result = ">50K" if prediction[0] == 1 else "<=50K"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediksi Pendapatan", result)
        with col2:
            st.metric("Tingkat Keyakinan", f"{prediction_proba[0][prediction[0]]:.2%}")
        
        with st.expander("Lihat Probabilitas Detail"):
            st.write({
                "Probabilitas <=50K": f"{prediction_proba[0][0]:.2%}",
                "Probabilitas >50K": f"{prediction_proba[0][1]:.2%}"
            })

def main():
    """Fungsi navigasi utama aplikasi."""
    stc.html("""
        <div style="background-color:#000;padding:10px;border-radius:10px">
            <h1 style="color:#fff;text-align:center">Aplikasi Prediksi Pendapatan</h1> 
            <h4 style="color:#fff;text-align:center">Dibangun untuk Proyek Akhir</h4> 
        </div>
    """)
    menu = ["Beranda", "Aplikasi Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Beranda":
        st.subheader("Selamat Datang!")
        st.markdown("""
            ### Aplikasi Prediksi Kategori Pendapatan
            Aplikasi ini menggunakan model Machine Learning untuk memprediksi apakah pendapatan tahunan seseorang berada di atas atau di bawah $50.000.
            
            #### Sumber Data
            Dataset yang digunakan adalah "Adult Census Income" dari UCI Machine Learning Repository.
            
            **Pilih "Aplikasi Machine Learning" dari menu sidebar untuk memulai.**
        """)
    elif choice == "Aplikasi Machine Learning":
        run_ml_app()

if __name__ == "__main__":
    main()
