# Cracking Neural Code

Selamat datang di repositori **Cracking Neural Code**! Aplikasi ini dibangun dengan Python.

---

## Persyaratan Sistem

Untuk menjalankan aplikasi ini, Anda memerlukan:

* **Python 3.12** (atau versi yang lebih baru direkomendasikan)
* **pip** (manajer paket Python, biasanya sudah termasuk dengan instalasi Python)

---

## Langkah-langkah Menjalankan Aplikasi

Ikuti langkah-langkah di bawah ini untuk mengatur dan menjalankan aplikasi di lingkungan lokal Anda.

### 1. Klon Repositori

Pertama, klon repositori ini ke mesin lokal Anda menggunakan Git:

```bash
git clone [https://github.com/braindecoding/cnc.git](https://github.com/braindecoding/cnc.git)
cd cnc
```

### 2. Buat dan Aktifkan Virtual Environment

Sangat disarankan untuk menggunakan **virtual environment** untuk mengisolasi dependensi proyek Anda. Ini mencegah konflik dengan proyek Python lainnya.

#### Untuk macOS / Linux:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

#### Untuk Windows:

```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
```

### 3. Instal Dependensi

Setelah virtual environment aktif, instal semua dependensi yang diperlukan menggunakan `pip`:

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

Sekarang Anda siap untuk menjalankan aplikasi!

```bash
python main.py
```


## Struktur Proyek (Opsional)

Berikut adalah gambaran singkat tentang struktur direktori proyek ini:

```
cnc/
├── venv/                   # Virtual environment (dibuat setelah langkah 2)
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
└── [direktori atau file data]
```


## Dukungan

Jika Anda mengalami masalah atau memiliki pertanyaan, silakan buka *issue* di repositori GitHub ini.
