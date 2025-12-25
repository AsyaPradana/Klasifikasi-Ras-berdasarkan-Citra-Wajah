import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Race Classification AI",
    page_icon="👤",
    layout="wide"
)

# --- STYLE CSS KUSTOM ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_single_model(model_name):
    # Menunjuk ke folder 'models/' sesuai struktur folder Anda
    model_file = f"models/{model_name}.h5" 
    
    if os.path.exists(model_file):
        return tf.keras.models.load_model(model_file)
    else:
        st.error(f"File {model_file} tidak ditemukan! Pastikan file ada di folder 'models'")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3829/3829733.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["Dashboard Prediksi", "Analisis Performa Model"])

    st.info("""
    **Identitas:**
    - Nama: Asya Cahya Pradana
    - NIM: 202210370311041
    """)

# --- HALAMAN 1: DASHBOARD PREDIKSI ---
if page == "Dashboard Prediksi":
    st.title("👤 Sistem Klasifikasi Ras Manusia")
    st.write("Unggah citra wajah untuk mendeteksi etnis berdasarkan model Deep Learning.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Input Citra")
        uploaded_file = st.file_uploader("Pilih gambar wajah...", type=["jpg", "jpeg", "png"])

        selected_model_name = st.selectbox(
            "Pilih Model Klasifikasi",
            ["CNN_Base", "MobileNetV2", "ResNet50"]
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption='Original Image', use_container_width=True)

        with col2:
            st.subheader("🎯 Hasil Prediksi")
            if st.button("Jalankan Prediksi"):
                with st.spinner(f'Memuat model {selected_model_name}...'):
                    # PERBAIKAN DI SINI: Memanggil fungsi yang benar
                    model = load_single_model(selected_model_name)

                if model is not None:
                    with st.spinner('Menganalisis fitur wajah...'):
                        # Preprocessing
                        img = image.convert('RGB') # Pastikan format RGB
                        img = img.resize((128, 128))
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = img_array / 255.0

                        # Prediksi
                        predictions = model.predict(img_array)
                        labels = ['Caucasoid', 'Negroid', 'Mongoloid']
                        result_idx = np.argmax(predictions)
                        confidence = np.max(predictions) * 100

                        # Tampilan Output
                        st.markdown(f"""
                        <div class="result-box">
                            <h2 style='text-align: center; color: #007bff;'>{labels[result_idx]}</h2>
                            <p style='text-align: center;'>Confidence Level: <b>{confidence:.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Bar Chart Probabilitas
                        fig = go.Figure([go.Bar(x=labels, y=predictions[0]*100, marker_color='#007bff')])
                        fig.update_layout(title="Probabilitas Per Kelas (%)", height=300)
                        st.plotly_chart(fig, use_container_width=True)
# --- HALAMAN 2: ANALISIS PERFORMA ---
else:
    st.title("📊 Analisis Performa Model")
    st.write("Perbandingan metrik evaluasi berdasarkan hasil pengujian (Accuracy, F1-Score).")

    # Ubah bagian data di Halaman 2 (Analisis Performa)
    data = {
    'Model': ['CNN_Base', 'MobileNetV2', 'ResNet50'],
    'Accuracy': [0.8633, 0.7708, 0.5117],
    'F1-Score': [0.8627, 0.7698, 0.5000]
    }
    df = pd.DataFrame(data)

    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    col_metrics1.metric("Best Accuracy", "86.75%", "CNN_Base")
    col_metrics2.metric("Avg MobileNet", "76.66%")
    col_metrics3.metric("Avg ResNet", "56.83%", "-19.83%")

    st.table(df)

    # Visualisasi Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Model'], y=df['Accuracy'], name='Accuracy', marker_color='#007bff'))
    fig.add_trace(go.Bar(x=df['Model'], y=df['F1-Score'], name='F1-Score', marker_color='#ff7f0e'))
    fig.update_layout(barmode='group', title="Visualisasi Akurasi vs F1-Score")
    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **Kesimpulan Analisis:** CNN Base (Non-Pretrained) mengungguli model raksasa karena dilatih khusus pada domain wajah manusia dengan parameter yang selaras dengan resolusi input.
    """)
