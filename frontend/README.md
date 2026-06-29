# 🚀 RevDadas Frontend (Next.js)

Aplikasi ini adalah hasil migrasi dashboard **RevDadas** (Revenue Daerah Cerdas) dari Streamlit ke Next.js. Pendekatan yang digunakan adalah **Jamstack (Static Generation)** agar aplikasi dapat berjalan sangat cepat dan bisa di-deploy ke layanan seperti Vercel tanpa butuh backend Python yang selalu menyala.

## 🏗️ Arsitektur Data (Static-ML)

Karena Next.js adalah framework Node.js dan model machine learning (Prophet & Isolation Forest) berjalan di Python, kita memisahkan proses analitik dari antarmuka pengguna:

1. **Pre-compute Script (`scripts/precompute.py`)**: Script Python ini memuat seluruh pipeline AI/ML dari Streamlit, menjalankannya untuk memproses data dari BPS, dan mengekspor hasilnya dalam bentuk JSON.
2. **Next.js Frontend**: Aplikasi Next.js akan membaca file JSON tersebut (`public/data/*.json`) saat memuat halaman, menghasilkan dashboard yang sangat cepat, responsif, dan interaktif.

## 💻 Cara Menjalankan Lokal

Karena aplikasi ini bergantung pada data JSON statis, kamu **wajib** menjalankan script pre-compute setidaknya satu kali sebelum menjalankan server Next.js.

### 1. Generate Data ML (Pre-compute)
Buka terminal dan arahkan ke direktori root project (satu tingkat di atas folder `frontend`), lalu jalankan:

```bash
# Aktifkan virtual environment (jika ada)
venv\Scripts\activate

# Jalankan script
python scripts/precompute.py
```
*Tunggu hingga script selesai memproses Prophet & Anomaly Detection dan menyimpan data di `frontend/public/data/`.*

### 2. Jalankan Dashboard Next.js
Buka terminal baru di folder `frontend` dan jalankan:

```bash
npm install
npm run dev
```

Buka [http://localhost:3000](http://localhost:3000) di browser untuk melihat hasilnya.

---

## 🚀 Cara Deploy ke Vercel

Aplikasi ini dirancang 100% *Vercel-ready* tanpa biaya server backend.

1. Pastikan folder `public/data` yang berisi file `.json` hasil *pre-compute* ikut di-commit ke Git.
2. Push repository (termasuk folder `frontend`) ke GitHub.
3. Di dashboard Vercel, pilih **Add New Project** dan arahkan ke repository tersebut.
4. Pada bagian **Root Directory**, setel ke `frontend`.
5. Klik **Deploy**.

> **Catatan Penting**: Setiap kali kamu memperbarui raw data (file CSV BPS), kamu harus **menjalankan ulang** `python scripts/precompute.py` secara lokal dan mem-push file JSON terbarunya ke GitHub. Vercel akan otomatis men-deploy ulang halaman dengan data terbaru.

## 🛠️ Teknologi yang Digunakan
- **Next.js** (App Router)
- **React** (Komponen & State Management)
- **Recharts** (Visualisasi Grafik)
- **React-Leaflet** (Peta Interaktif)
- **jsPDF** (Eksport Laporan PDF)
