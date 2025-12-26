# 👤 Sistem Klasifikasi Ras Berdasarkan Citra Wajah

## 📌 Deskripsi Proyek
Proyek ini dikembangkan sebagai tugas Ujian Akhir Praktikum (UAP) Pembelajaran Mesin 2025. Fokus utama sistem ini adalah membangun model Deep Learning yang mampu mengklasifikasikan ras manusia berdasarkan citra wajah ke dalam tiga kategori etnis utama: **Caucasoid**, **Negroid**, dan **Mongoloid**. 

Sistem ini membandingkan performa antara model Convolutional Neural Network (CNN) Base yang dibangun dari awal (*from scratch*) dengan pendekatan Transfer Learning menggunakan arsitektur modern (**MobileNetV2** dan **ResNet50**). Model dengan performa terbaik diimplementasikan ke dalam aplikasi web interaktif menggunakan Streamlit.

---

# 📂 Dataset dan Preprocessing

## 1. Sumber Dataset
Dataset yang digunakan adalah **UTKFace Large Scale Dataset**, yang terdiri dari lebih dari 20.000 citra wajah dengan variasi usia, jenis kelamin, dan etnis yang luas.

***Link Dataset***: [UTKFace on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

## 2. Struktur Data & Labeling
Label ras diekstraksi dari nama file dengan format: `[age]_[gender]_[race]_[date].jpg`.
Contoh: `25_0_2_20170116174525125.jpg`
* **Race Index 2:**
  * 0: Caucasoid (White)
  * 1: Negroid (Black)
  * 2: Mongoloid (Asian)
  * 3: India
  * 4: Others

## 3. Tahap Preprocessing
Langkah-langkah berikut dilakukan untuk memastikan data siap latih dan seimbang:
* **Filtering Ras:** Hanya mengambil data label `0` (Caucasoid), `1` (Negroid), dan `2` (Mongoloid).
* **Data Balancing:** Membatasi jumlah sampel maksimal **2.000 citra per kategori** untuk mencegah model bias terhadap ras mayoritas.
* **Resizing:** Menyeragamkan ukuran citra menjadi 128 x 128 piksel.
* **Normalisasi:** Skalasi nilai piksel menjadi rentang [0, 1].
* **Augmentasi Data:** Menggunakan *horizontal flip* untuk meningkatkan generalisasi model.

---

# 🧠 Arsitektur Model
Proyek ini membandingkan tiga model:

1. **CNN Base (Non-Pretrained)**
   * Dibangun tanpa bobot terlatih sebelumnya.
   * Arsitektur menggunakan 4 blok Konvolusi (filter 4, 8, 16, 32).
   * Dilengkapi MaxPooling, Flatten, dan Dense Layer (512 neuron).

2. **MobileNetV2 (Transfer Learning)**
   * Menggunakan bobot *pretrained* ImageNet dengan *base layers* yang dibekukan (*frozen*).
   * Menambahkan GlobalAveragePooling dan Dense Layer kustom untuk klasifikasi 3 ras.

3. **ResNet50 (Transfer Learning)**
   * Menggunakan arsitektur *Residual Network* (50 layer) dengan bobot ImageNet.
   * Dirancang untuk menangkap fitur yang sangat kompleks melalui *skip connections*.

---

# 📊 Hasil Evaluasi dan Analisis

Model dievaluasi menggunakan beberapa metrik, termasuk classification report dan confusion matrix.

Berdasarkan eksperimen 10 epoch, berikut hasil perbandingannya:

| Metrik | CNN_Base | MobileNetV2 | ResNet50 |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.8675 | 0.7666 | 0.5683 |
| **Precision** | 0.8674 | 0.7689 | 0.5958 |
| **Recall** | 0.8675 | 0.7666 | 0.5683 |
| **F1-Score** | 0.8674 | 0.7672 | 0.5534 |
## Analisis Model: CNN_Base
<img width="1114" height="361" alt="Screenshot 2025-12-26 024222" src="https://github.com/user-attachments/assets/062b3d6f-1e36-4d38-9f7a-6c3c2e47a2aa" />
Model ini menunjukkan performa yang paling stabil dan paling baik di antara ketiganya.

***Learning Curve Akurasi***: Grafik menunjukkan kenaikan yang konsisten pada akurasi training hingga mencapai angka di atas 95%. Akurasi validation juga mengikuti tren naik secara stabil dan berakhir di kisaran 86-88%.

***Learning Curve Loss***: Grafik loss training menurun sangat mulus. Namun, pada grafik loss validation, terlihat sedikit kenaikan (uptick) di akhir epoch (epoch 8 ke 9).

***Analisis***: Model ini memiliki kemampuan generalisasi yang sangat baik. Meskipun ada indikasi awal overfitting ringan di akhir sesi, model ini tetap yang paling handal karena selisih antara akurasi training dan validation tidak terlalu ekstrem.

## Analisis Model: MobileNetV2
<img width="1112" height="363" alt="Screenshot 2025-12-26 024130" src="https://github.com/user-attachments/assets/de190dfa-17e0-4e89-b409-fa77d2588247" />

Model ini menunjukkan gejala Overfitting yang cukup jelas.

***Learning Curve Akurasi***: Terdapat celah (gap) yang lebar antara akurasi training (mencapai ~90%) dan akurasi validation yang tertahan di kisaran 75-77%.

***Learning Curve Loss***: Sementara loss training terus menurun, loss validation terlihat sangat fluktuatif dan cenderung tidak menunjukkan penurunan yang signifikan setelah epoch ke-4, bahkan berakhir lebih tinggi daripada saat memulai.

***Analisis***: Model "terlalu pintar" menghafal data training tetapi gagal mengenali pola pada data baru (validation). Hal ini sering terjadi pada model pretrained besar jika lapisan atasnya (top layers) terlalu kompleks untuk dataset yang relatif spesifik atau jika data tidak cukup bervariasi.

## Analisis Model: MobileNetV2
<img width="1110" height="361" alt="Screenshot 2025-12-26 024255" src="https://github.com/user-attachments/assets/33adc0ba-928e-480a-b76f-53e79f4ea7f6" />

Analisis Model: ResNet50
Model ini menunjukkan performa yang tidak stabil (Instability) dan akurasi yang rendah.

***Learning Curve Akurasi***: Grafik sangat fluktuatif (naik-turun tajam). Akurasi validation sempat melonjak tinggi di epoch 8 namun jatuh kembali di epoch 9. Akurasi rata-ratanya hanya berkisar di angka 50%.

***Learning Curve Loss***: Grafik loss validation terlihat sangat tidak menentu (zigzag), menunjukkan bahwa model kesulitan menemukan titik optimal (convergence) selama proses pelatihan.

***Analisis***: ResNet50 tampaknya terlalu kompleks untuk resolusi gambar atau jumlah data yang digunakan. Akurasi yang rendah menunjukkan bahwa fitur pretrained dari ImageNet tidak teradaptasi dengan baik pada tugas klasifikasi ras ini tanpa dilakukan fine-tuning yang lebih mendalam pada lapisan-lapisan bawahnya.

### Confusion Matrix
Di bawah ini adalah confusion matrix untuk ketiga model.

***MobileNetV2***

<img width="478" height="440" alt="Screenshot 2025-12-26 024157" src="https://github.com/user-attachments/assets/132934fe-3d34-4930-a6be-506e7ca5bb5f" />

***CNN Base***

<img width="480" height="438" alt="Screenshot 2025-12-26 024212" src="https://github.com/user-attachments/assets/d607b87c-09f4-48d5-ad43-81bab98b8719" />

***Resnet50***

<img width="477" height="437" alt="Screenshot 2025-12-26 024309" src="https://github.com/user-attachments/assets/d3ed98e7-c387-42d1-9cf2-47bb534d9957" />

### Tabel Analisis Perbandingan Model
| Nama Model | Akurasi | Hasil Analisis |
| :--- | :---: | :--- |
| **CNN Base** | **86.75%** | **Model Terbaik.** Paling efektif mempelajari fitur spesifik wajah manusia dari dataset ini. Terbukti arsitektur spesifik mampu mengungguli model umum pada data yang tersegmentasi baik. |
| **MobileNetV2** | **76.67%** | **Performa Menengah.** Stabil dan efisien, namun membutuhkan *fine-tuning* lebih dalam untuk mengenali detail rasial yang lebih halus dibanding objek umum ImageNet. |
| **ResNet50** | **56.83%** | **Performa Terendah.** Kompleksitas model tidak selaras dengan resolusi input $128 \times 128$, menyebabkan fitur ImageNet gagal beradaptasi optimal pada klasifikasi ras tanpa *unfreeze* layer bawah. |

---

# 💻 Panduan Instalasi & Menjalankan Website (Lokal)

1. **Persiapan Environment**
   Install library utama melalui terminal:
   ```bash
   pip install streamlit tensorflow pillow numpy pandas
   ```
2. ***Struktur File***

📁 project_uap/

├── 📄 app.py               # Source code Streamlit

├── 📁 models/              # Folder penyimpanan model .h5


│                 ├── CNN_Base.h5

│                 ├── MobileNetV2.h5

│                 └── ResNet50.h5

└── 📄 README.md

3. ***Menjalankan Aplikasi***

```bash streamlit run app.py```

---

# 📷 Sistem Sederhana Streamlit 📷
Aplikasi berbasis Streamlit ini bertujuan untuk memudahkan pengguna dalam melakukan prediksi ras secara live. Aplikasi ini dapat memprediksi ras (Kaukasoid, Mongoloid, Negroid) berdasarkan input yang diberikan.

***Halaman Utama***

<img width="1919" height="902" alt="Screenshot 2025-12-26 033448" src="https://github.com/user-attachments/assets/cce40f02-b276-4090-90ea-dd666bcd5ca8" />

Pada Halaman Utama, pengguna dapat menjalankan prediksi langsung dengan mengupload gambar dan memilih Model Klasifikasi.

***Menu Sidebar***

Menu Sidebar memiliki navigasi untuk menuju Halaman Utama dan Halaman Analisis Model, serta identitas pembuat Sistem.

***Halaman Analisis Model***

<img width="1919" height="906" alt="Screenshot 2025-12-26 033503" src="https://github.com/user-attachments/assets/8ae61efd-9afe-4ad7-a171-0ac00ced7d6b" />

Pengguna dapat melihat Analisis dari ketiga model yang dipakai

***Tampilan Hasil Prediksi***

<img width="1919" height="910" alt="Screenshot 2025-12-26 034107" src="https://github.com/user-attachments/assets/60192a8a-c4eb-4a83-ac43-f1bb07644c9a" />

Setelah memasukkan gambar, pengguna dapat melihat hasil prediksi dengan mengklik button Jalankan Prediksi. Tampilan diatas adalah contoh yang menampilkan hasil prediksi beserta probabilitas per kelas.

***Link Live App***

[Ras](https://klasifikasi-ras-berdasarkan-citra-wajah-8f797jdiuorojt9fa936lj.streamlit.app/)

# 🤝 Referensi

Dataset: [UTKFace on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

---
Dibuat oleh Asya Cahya Pradana (202210370311041) - UAP Pembelajaran Mesin 2025.
