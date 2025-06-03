# Cracking Neural Code

Selamat datang di repositori **Cracking Neural Code**!

Aplikasi ini adalah program untuk **klasifikasi digit EEG (electroencephalogram)** yang menggunakan machine learning untuk membedakan antara digit 6 dan 9 berdasarkan sinyal otak. Program ini menganalisis data sinyal EEG dan mengekstrak fitur spasial serta frekuensi untuk melatih model Support Vector Machine (SVM) yang dapat mengenali pola pikiran saat seseorang memikirkan digit tertentu.

## Fitur Utama

- 🧠 **Analisis Sinyal EEG**: Memproses data sinyal otak dari 14 channel EEG
- 🔍 **Ekstraksi Fitur**: Mengekstrak fitur hemisferik, regional, sinkronisasi, dan frekuensi
- 🤖 **Machine Learning**: Menggunakan SVM dengan kernel RBF untuk klasifikasi
- 📊 **Evaluasi Model**: Menampilkan akurasi, confusion matrix, dan classification report
- 📈 **Analisis Frekuensi**: Menganalisis power spektral dalam band alpha dan beta


## Persyaratan Sistem

Untuk menjalankan aplikasi ini, Anda memerlukan:

* **Python 3.11** (atau versi yang lebih baru direkomendasikan)
* **pip** (manajer paket Python, biasanya sudah termasuk dengan instalasi Python)


## Langkah-langkah Menjalankan Aplikasi

Ikuti langkah-langkah di bawah ini untuk mengatur dan menjalankan aplikasi di lingkungan lokal Anda.

### 1. Klon Repositori

Pertama, klon repositori ini ke mesin lokal Anda menggunakan Git:

```bash
git clone [https://github.com/braindecoding/cnc.git](https://github.com/braindecoding/cnc.git)
cd cnc
```

### 2. Instal Dependensi

Instal semua dependensi yang diperlukan menggunakan `pip`:

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Aplikasi ini membutuhkan dataset EEG dari **MindBigData - "MNIST" of Brain Digits**:

1. **Kunjungi halaman download**: https://www.mindbigdata.com/opendb/index.html
2. **Download file**: `MindBigData-EP-v1.0.zip` (408 MB)
3. **Extract file** dan cari file `EP1.01.txt`
4. **Buat folder** `Data/` di direktori proyek
5. **Letakkan** file `EP1.01.txt` di dalam folder `Data/`

### 4. Jalankan Aplikasi

Sekarang Anda siap untuk menjalankan aplikasi!

```bash
python main.py
```

## Hasil yang Diharapkan

Ketika aplikasi berhasil dijalankan dengan dataset, Anda akan melihat output seperti:

```
🚀 Best Model for EEG Digit Classification
==================================================
📂 Loading data for digits [6, 9]...
📖 Reading file: Data/EP1.01.txt
 Found: 100 digit-6, 100 digit-9
 Found: 200 digit-6, 200 digit-9
...
✅ Final count: 500 digit-6, 500 digit-9
📊 Data shape: (1000, 1792)

🧩 Extracting features...
✅ Features extracted: (1000, 8)

🤖 Training best model...
✅ Model accuracy: 0.7234

📊 Classification Report:
              precision    recall  f1-score   support
     Digit 6       0.71      0.75      0.73       150
     Digit 9       0.74      0.70      0.72       150

    accuracy                           0.72       300
   macro avg       0.72      0.72      0.72       300
weighted avg       0.72      0.72      0.72       300

 Confusion Matrix:
  112   38 | Digit 6
   45  105 | Digit 9
 6 9 <- Predicted
 Sensitivity (Digit 6): 0.7467
 Specificity (Digit 9): 0.7000

✅ Analysis completed!
🎯 Note: Accuracy above 0.5 indicates that spatial patterns
 can be detected in the EEG data to differentiate digits 6 and 9.
```

## Dataset Information

### 🧠 Tentang MindBigData

**MindBigData** adalah dataset EEG (electroencephalogram) yang berisi **1,207,293 sinyal otak** dari satu subjek yang dikumpulkan selama hampir 2 tahun (2014-2015). Dataset ini disebut sebagai **"MNIST" of Brain Digits** karena berisi sinyal otak saat subjek melihat dan memikirkan digit 0-9.

### 📊 Spesifikasi Dataset EP1.01.txt

- **Device**: Emotiv EPOC (14 channel EEG)
- **Channels**: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- **Sampling Rate**: ~128 Hz
- **Duration**: 2 detik per sinyal
- **Total Signals**: 910,476 sinyal EEG
- **Format**: Tab-separated values (TSV)

### 📋 Format Data

```
[id]    [event]    [device]    [channel]    [code]    [size]    [data]
```

- **id**: Nomor referensi
- **event**: ID event untuk membedakan sinyal dari lokasi otak berbeda
- **device**: "EP" untuk Emotiv EPOC
- **channel**: Lokasi otak sesuai sistem 10/20 (AF3, F7, F3, dll.)
- **code**: Digit yang dipikirkan (0-9, atau -1 untuk sinyal acak)
- **size**: Jumlah data points (~260 untuk 2 detik)
- **data**: Time-series data sinyal EEG (comma-separated)

### 📜 Lisensi Dataset

Dataset tersedia di bawah **Open Database License** dan dapat digunakan secara bebas untuk penelitian dengan memberikan atribusi kepada sumber.

**Sumber**: MindBigData by David Vivancos & Felix Cuesta
**Website**: https://www.mindbigdata.com/
**Paper**: [MindBigData the MNIST of Brain Digits v1.01](https://www.researchgate.net/publication/281817951_MindBigData_the_MNIST_of_Brain_Digits_v101)

## Struktur Proyek

Berikut adalah gambaran singkat tentang struktur direktori proyek ini:

```
cnc/
├── Data/
│   └── EP1.01.txt          # Dataset EEG (download terpisah)
├── main.py                 # Script utama aplikasi
├── requirements.txt        # Dependensi Python
├── .gitignore
└── README.md
```


## Dukungan

Jika Anda mengalami masalah atau memiliki pertanyaan, silakan buka *issue* di repositori GitHub ini.
