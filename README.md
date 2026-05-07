# Sistem Klasifikasi Sentimen Wisata Telaga Sarangan

Aplikasi web untuk menganalisis sentimen ulasan pengunjung wisata Telaga Sarangan menggunakan algoritma **Random Forest**. Dibangun dengan **Streamlit**.

> **Catatan:** File model (`model.pkl`) tidak disimpan di repository ini karena ukurannya besar. Model akan **diunduh otomatis dari Google Drive** saat aplikasi pertama kali dijalankan. Pastikan koneksi internet Anda aktif.

---

## Persyaratan Sistem

Sebelum memulai, pastikan komputer Anda sudah terinstal:

- **Python 3.9 atau lebih baru** — [Download Python](https://www.python.org/downloads/)
- **pip** (biasanya sudah terinstal bersama Python)

Untuk mengecek apakah Python sudah terinstal, buka **Terminal** (Mac/Linux) atau **Command Prompt** (Windows) lalu ketik:

```bash
python --version
```

---

## Cara Menjalankan Aplikasi (Langkah demi Langkah)

### Langkah 1 — Unduh atau Clone Repository Ini

Jika menggunakan Git:

```bash
git clone https://github.com/USERNAME/NAMA_REPO.git
```

Atau unduh sebagai file ZIP dari GitHub, lalu ekstrak ke folder pilihan Anda.

---

### Langkah 2 — Masuk ke Folder `web`

Buka Terminal / Command Prompt, lalu arahkan ke folder `web` di dalam project:

```bash
cd NAMA_REPO/web
```

> Ganti `NAMA_REPO` dengan nama folder hasil unduhan Anda.

---

### Langkah 3 — Buat Virtual Environment (Lingkungan Python Terisolasi)

Virtual environment berguna agar paket yang diinstal tidak mengganggu instalasi Python lain di komputer Anda.

```bash
python -m venv .venv
```

---

### Langkah 4 — Aktifkan Virtual Environment

**Mac / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

Setelah berhasil, nama terminal Anda akan berubah menjadi `(.venv)` di awal baris.

---

### Langkah 5 — Instal Semua Dependensi

```bash
pip install -r requirements.txt
```

Tunggu hingga proses instalasi selesai. Ini mungkin memakan waktu beberapa menit tergantung kecepatan internet Anda.

---

### Langkah 6 — Jalankan Aplikasi

```bash
streamlit run app.py
```

Setelah itu, terminal akan menampilkan pesan seperti ini:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Buka browser Anda (Chrome, Firefox, dll.) dan kunjungi:

```
http://localhost:8501
```

Aplikasi siap digunakan!

---

## Cara Menggunakan Aplikasi

1. Ketik ulasan pengunjung wisata Telaga Sarangan pada kolom teks yang tersedia.
2. Klik tombol **"Proses Analisis Sentimen"**.
3. Hasil klasifikasi akan muncul di bawah tombol:
   - **Sentimen Positif** — Ulasan bernada positif
   - **Sentimen Netral** — Ulasan bernada netral
   - **Sentimen Negatif** — Ulasan bernada negatif
4. Anda juga bisa menggunakan contoh ulasan yang tersedia di menu sebelah kiri.

---

## Cara Menghentikan Aplikasi

Kembali ke jendela Terminal, lalu tekan:

```
Ctrl + C
```

---

## Struktur Folder

```
web/
├── app.py            # Kode utama aplikasi Streamlit
├── requirements.txt  # Daftar paket Python yang dibutuhkan
└── README.md         # Panduan ini
```

> `model.pkl` tidak perlu diunggah ke GitHub. File ini akan diunduh otomatis dari Google Drive saat aplikasi pertama kali dibuka.

---

## Troubleshooting (Masalah Umum)

| Masalah | Solusi |
|---|---|
| `python: command not found` | Coba gunakan `python3` sebagai pengganti `python` |
| `pip: command not found` | Coba gunakan `pip3` atau `python -m pip` |
| `streamlit: command not found` | Pastikan virtual environment sudah aktif (lihat Langkah 4) |
| `model.pkl not found` | Pastikan file `model.pkl` ada di dalam folder `web/` |
| Port 8501 sudah digunakan | Jalankan dengan `streamlit run app.py --server.port 8502` |

---

## Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Antarmuka Web | Streamlit |
| Algoritma Klasifikasi | Random Forest (scikit-learn) |
| Pemrosesan Teks | PySastrawi (Stemmer & Stopword) |
| Bahasa Pemrograman | Python 3.9+ |

---

*Proyek Tugas Akhir — Klasifikasi Sentimen Wisata Telaga Sarangan*
