# Rangkuman Hasil — Dataset APBD & Adu Mesin Peramalan

**Sumber data** https://djpk.kemenkeu.go.id/portal/data/apbd
**Cakupan** 38 provinsi · 2023–2026
**Commit** `2fefa27` (lokal, belum di-push)
**Tanggal** 8 September 2026
**Dokumen terkait** [`laporan-hasil-analisis-model-prophet.md`](laporan-hasil-analisis-model-prophet.md) · [`laporan model yang memungkinkan.md`](laporan%20model%20yang%20memungkinkan.md)

---

## Ringkasan Satu Halaman

| | |
|---|---|
| **Scraping** | 38 provinsi, 36.704 baris, 790 seri siap model |
| **Rentang** | Januari 2023 – September 2026 |
| **Mesin terpilih** | Profil Serapan Berjangkar — akurasi 70,1%, MASE 0,79 |
| **Pembanding** | Prophet — akurasi 54,3%, MASE 0,97, 45× lebih lambat |
| **Temuan kunci** | Prophet MASE 1,00 di rumpun inti = seri dengan tebakan sepele |

---

# Bagian 1 — Hasil Scraping

1.824 request ke endpoint `csv_apbd` dengan jeda sopan 0,25 detik, ditata jadi satu folder per provinsi.

## 1.1 Angka utama

| Aspek | Nilai |
|---|--:|
| Provinsi | **38** (seluruh portal DJPK) |
| Baris konsolidasi | **36.704** |
| Jenis akun | 24 |
| Seri siap model | **790** |
| Baris siap model | 31.934 |
| Rentang | 2023-01 → **2026-09** |

## 1.2 Sebaran dan kelengkapan

| Aspek | Nilai | Catatan |
|---|--:|---|
| Baris per tahun | 9.136 / 9.100 / 9.288 / 9.180 | 2023 / 2024 / 2025 / 2026 — merata |
| Akun per provinsi | 19 – 24 (median 21) | Papua terlengkap; Kalimantan Barat & Papua Pegunungan paling sedikit |
| Titik per seri | median 44, maks 45 | **672 dari 790** seri punya ≥36 titik |
| Nilai negatif / kosong | 0 / 0 | bersih setelah de-kumulasi |

## 1.3 Lima provinsi terbesar — Pendapatan Daerah 2025

| Provinsi | Realisasi | Pagu | Serapan |
|---|--:|--:|--:|
| DKI Jakarta | Rp 68,4 T | Rp 81,7 T | 84% |
| Jawa Barat | Rp 29,4 T | Rp 31,0 T | 95% |
| Jawa Timur | Rp 26,2 T | Rp 28,6 T | 91% |
| Jawa Tengah | Rp 23,8 T | Rp 24,7 T | 96% |
| Kalimantan Timur | Rp 17,7 T | Rp 20,1 T | 88% |

## 1.4 Keputusan data — 2022 tidak diambil

Diverifikasi langsung ke portal: SIKD hanya menyimpan angka **tahunan** untuk 2021–2022.

```
DKI Jakarta, realisasi kumulatif Pendapatan Daerah (Rp T)

2022:  67.3  67.3  67.3  67.3  67.3  67.3  67.3  67.3  67.3  67.3  67.3  67.3    ->  1 nilai unik
2023:   2.8   5.6  11.7  16.4  16.4  26.6  31.0  34.9  41.9  46.8  57.1  66.2    -> 11 nilai unik
```

Kalau 2022 dipaksa jadi deret bulanan, de-kumulasi menghasilkan **Januari = seluruh nilai setahun** dan **Februari–Desember = nol**. Itu merusak profil serapan yang jadi inti model.

Konsisten dengan keputusan yang sudah didokumentasikan di [`src/apbd_adapter.py`](../src/apbd_adapter.py).

## 1.5 Penandaan mutu

Dua masalah dideteksi otomatis dan ditandai per baris, lalu **dikeluarkan** dari berkas siap-model:

| Penanda | Baris | Arti |
|---|--:|---|
| `datar_tahunan` | 1.200 (3,3%) | akun yang 12 bulannya identik — seluruhnya pos bernilai nol |
| `belum_final` | 3.570 (9,7%) | bulan akhir tanpa kenaikan berarti |

Sebaran `belum_final` per tahun: 2023 = 163 · 2024 = 271 · 2025 = 299 · **2026 = 2.837**

**Perbaikan toleransi.** Deteksi awal memakai perbandingan persis, dan 12 baris Okt–Des 2026 lolos sebagai "data" padahal isinya sepersepuluh rupiah — derau pembulatan di sumber DJPK. Diganti ke toleransi relatif (`1e-6 × skala tahunan`); sisanya sekarang **nol**.

## 1.6 Struktur keluaran

```
C:\Project\revdadas\data\scraped\                     24 MB
├── _konsolidasi.csv          3,5 MB   skema revdadas, realisasi KUMULATIF
├── _statsforecast.csv        1,8 MB   unique_id, ds, y — siap Theta
├── _meta.json                5,5 KB   cakupan + catatan mutu
├── Aceh/  Bali/  …  Papua Barat Daya/            38 folder
│   ├── <Provinsi>_bulanan.csv         skema revdadas
│   ├── _meta.json                     catatan mutu provinsi
│   └── raw/                           48 XML mentah (cache, tidak dilacak git)
```

Berkas mentah dipertahankan di disk sebagai cache dan jejak audit — menjalankan ulang scraper tidak akan memukul server DJPK lagi. Tidak dilacak git agar 24 MB data yang bisa diregenerasi tidak membengkakkan riwayat repo.

**Menjalankan ulang:**

```bash
python scripts/scrape_djpk.py                  # semua provinsi 2023-2026
python scripts/scrape_djpk.py --tahun 2026     # satu tahun saja
python scripts/scrape_djpk.py --provinsi 09 10 # provinsi tertentu
```

---

# Bagian 2 — Adu Mesin Peramalan

Profil Serapan versus Prophet, diuji pada data yang sama dengan **protokol evaluasi yang disamakan**: seluruh parameter dipilih di validasi dalam lalu dikunci sebelum periode uji disentuh.

## 2.1 Mengapa protokol harus disamakan

Prophet versi asli memilih bobot ensemble dengan meminimalkan error **di periode uji**, lalu melaporkan error periode uji yang sama sebagai akurasi. Karena bobot `0.0` ikut jadi kandidat, angkanya secara matematis tidak mungkin kalah dari baseline — dan memang **0 dari 40 seri** kalah.

Tanpa penyamaan ini, Prophet akan terlihat unggul palsu.

## 2.2 Hasil keseluruhan

| Metrik | Profil Serapan | Prophet | Selisih |
|---|--:|--:|--:|
| **Akurasi headline** | **70,1%** | 54,3% | +15,8 poin |
| WAPE median | **29,9%** | 45,7% | −15,8 poin |
| MASE median | **0,79** | 0,97 | lebih baik |
| Seri layak (MASE<1) | **77 / 110** | 63 / 110 | +14 seri |
| **Waktu jalan** | **14 detik** | 10 mnt 35 dtk | **45× lebih cepat** |
| Dependensi | pandas, numpy | + prophet + Stan | — |

## 2.3 Per rumpun akun

| Rumpun | n | Serapan WAPE | Serapan MASE | Prophet WAPE | Prophet MASE |
|---|--:|--:|--:|--:|--:|
| Belanja | 16 | **15,1%** | **0,69** | 20,8% | 1,00 |
| Pendapatan inti | 24 | **16,8%** | **0,90** | 21,2% | 1,00 |
| Pendapatan rinci | 70 | **48,5%** | **0,75** | 83,2% | 0,96 |

**Angka paling penting di tabel itu adalah MASE 1,00 milik Prophet** di kedua rumpun inti. MASE 1,00 berarti Prophet praktis *seri* dengan tebakan sepele "nilai bulan sama tahun lalu" — seluruh mesin tren, changepoint, dan Fourier-nya tidak menghasilkan apa pun di atas itu.

## 2.4 Mengapa hasilnya begitu

Bukan karena Prophet model yang buruk, melainkan karena datanya tidak menyediakan apa yang Prophet cari:

```
autokorelasi sisa setelah pola bulan dibuang   +0,05 s/d +0,11
  -> praktis nol: tidak ada momentum untuk dipelajari
     komponen tren / changepoint / autoregresi hanya mencocokkan derau

ukuran model vs data
  Fourier tahunan order 10  = 20 parameter
  changepoint default       = 25
  data latih                = 30 titik      -> overfit struktural
```

Sebaliknya, Profil Serapan memakai satu informasi yang Prophet tidak bisa sentuh: **pagu Anggaran**. Pagu sudah diketahui sejak awal tahun, tersedia di 98,8% seri-tahun, dan rasio serapannya stabil (CV antar-tahun 6,8%).

Validasi memilih `omega = 0` untuk **seluruh** rumpun — artinya jangkar Anggaran murni, mengalahkan jangkar realisasi berjalan di setiap kombinasi.

## 2.5 Bentuk model

```
prediksi[bulan] = T × p[bulan]

  p[bulan] = normalisasi( γ · profil_sendiri + (1−γ) · 1/12 )
  T        = Anggaran × rasio_serapan_historis

  γ terpilih:  belanja 0,75   pendapatan inti 0,40   pendapatan rinci 0,15
```

Nilai γ yang berbeda tajam antar rumpun mencerminkan pengukuran: belanja punya musiman kuat (R² 91%) sehingga profilnya dipercaya banyak; pendapatan rinci hampir seluruhnya derau sehingga diratakan kuat-kuat.

**Dua mesin dapat dipilih:**

```bash
python scripts/precompute.py --engine serapan   # default
python scripts/precompute.py --engine prophet
```

---

# Bagian 3 — Perbaikan Lain yang Menyertai

| Perbaikan | Sebelum | Sesudah |
|---|--:|--:|
| Deteksi anomali | 5 / 3.960 | **14 / 3.960** |
| Seri yang diukur akurasinya | 40 | **110** |
| Sebaran anomali per tahun | 2024:4 · 2025:1 | 2023:3 · 2024:5 · 2025:6 |
| Gerbang tayang | tidak ada | **MASE < 1** |

Modul anomali sebelumnya membuang **193 dari 198** deteksi karena mencocokkan substring nama akun:

```python
is_lumpy = any(k in jenis.lower() for k in
    ['hibah','tidak terduga','modal','transfer','kekayaan daerah',
     'lain-lain','lainnya','darurat'])
```

Kata `'transfer'` menjaring TKDD dan Pendapatan Transfer Pemerintah Pusat — padahal keduanya justru arus **paling rutin** di seluruh dataset, dan TKDD adalah pos pendapatan terbesar:

| Akun | CV | %nol | %bulan aktif |
|---|--:|--:|--:|
| TKDD | 0,31 | 0,0 | 100 |
| Transfer Pemerintah Pusat | 0,36 | 3,8 | 97 |
| *(pembanding)* Retribusi Daerah | 0,76 | 3,8 | 96 |
| *(lumpy sejati)* Pendapatan Hibah | 2,36 | 18,4 | 60 |

Aturan itu juga **melewatkan** Retribusi Daerah yang di sebagian provinsi memang sporadis.

Diganti pengukuran per (Provinsi × Jenis_Pendapatan): lumpy bila `< 80% bulan aktif` **atau** `CV > 1,0`.

---

# Bagian 4 — Catatan untuk Eksperimen Theta

Berkas `_statsforecast.csv` sudah dalam konvensi statsforecast dan siap dipakai di lingkungan terpisah. **Mesin revdadas tidak diubah** — tidak ada `--engine theta` yang ditambahkan.

## 4.1 Apakah Theta bagus?

Theta metode yang sangat solid — pemenang M3, masih top-tier di M4, salah satu baseline terkuat yang ada.

Tetapi kekuatannya ada pada menangkap **tren dan momentum**, dan di data ini autokorelasi sisa praktis nol. Perkiraan: Theta akan mendarat di sekitar baseline musiman (**~20–22% WAPE**) — lebih baik dari Prophet, tetapi kalah dari Profil Serapan, karena **Theta tidak bisa memakai Anggaran**.

## 4.2 Saran: Theta pada rasio serapan

```
target = realisasi_kumulatif / Anggaran      # terbatas 0–1, stabil antar provinsi
ramalkan kurva itu dengan Theta
kalikan balik dengan Anggaran tahun target
```

Itu menggabungkan smoothing Theta yang teruji dengan jangkar pagu yang jadi keunggulan utama di data ini. Kolom `anggaran` tersedia di `_konsolidasi.csv`.

## 4.3 Ambang kelayakan

Apa pun hasilnya, hitung **MASE terhadap seasonal-naive**. Kalau tidak mencapai MASE < 1, metodenya belum layak pakai — ambang yang sama diterapkan ke model sendiri.

## 4.4 Kalau butuh lebih dari Theta

Kandidat terkuat bukan model tunggal, melainkan **model global**: satu LightGBM di seluruh 38 provinsi × akun sekaligus, dengan fitur dummy bulan, id provinsi, rasio serapan YTD, dan Anggaran. Dengan deret pendek 45 titik, meminjam pola antar provinsi lebih berharga daripada kecanggihan per-seri.

---

# Cakupan dan Status

Angka scraping dihitung dari `data/scraped/_konsolidasi.csv` dan `_statsforecast.csv`. Angka perbandingan mesin diperoleh dengan menjalankan `scripts/precompute.py` pada kedua mesin secara bergantian, holdout Jul–Des, seluruh parameter dipilih di validasi dalam.

**Status git:** lima commit berada di komputer lokal dan **belum di-push** ke GitHub mana pun.

```
2fefa27  Tambah scraper APBD DJPK: 38 provinsi, 2023-2026
ec1287d  Perbaiki deteksi anomali: sifat lumpy diukur dari data
a405f73  Gerbang MASE + naikkan akurasi 36,5% -> 29,9% WAPE
445ec24  Tambah laporan audit model dan skrip uji
3ce9f8b  Ganti mesin forecast: Prophet -> Profil Serapan
```

Push ke `M0zz07/testing-revdadas` terhalang izin — kredensial mesin ini milik akun `Wiessliem` yang belum jadi collaborator di sana.

**Keterbatasan:** Theta, `statsforecast`, dan LightGBM belum dijalankan di lingkungan ini — perkiraan pada Bagian 4 berasal dari pengukuran autokorelasi dan lantai oracle, bukan hasil eksperimen langsung.
