# RevDadas - Revenue Daerah Cerdas

**AI-Driven Predictive Analytics for Fraud Detection and Revenue Forecasting in Local Government Collections**

> Solusi prediktif untuk deteksi fraud dan prakiraan revenue pendapatan daerah berbasis AI

---

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Fitur Utama](#fitur-utama)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menggunakan](#cara-menggunakan)
- [Roadmap](#roadmap)

---

## 🎯 Tentang Proyek

RevDadas adalah sistem pemungutan pendapatan daerah yang cerdas, prediktif, dan berbasis AI untuk:

✅ **Mendeteksi fraud/under-reporting** secara otomatis menggunakan Isolation Forest  
✅ **Memprediksi revenue** 6-24 bulan ke depan per provinsi menggunakan Prophet  
✅ **Menghasilkan rekomendasi kebijakan** otomatis berbasis data  
✅ **Visualisasi interactive** untuk monitoring real-time

**Mendukung:**

- Pajak Bumi dan Bangunan (PBB)
- Retribusi Pasar
- Pajak Hotel & Restoran

---

## ⚡ Fitur Utama

| Fitur                        | Deskripsi                                               |
| ---------------------------- | ------------------------------------------------------- |
| **Forecasting**              | Prakiraan revenue 6-24 bulan menggunakan Facebook Prophet |
| **Anomaly Detection**        | Deteksi fraud & under-reporting dengan Isolation Forest |
| **Dashboard Interactive**    | Monitoring real-time dengan Streamlit                   |
| **Geospatial Visualization** | Peta heatmap Indonesia dengan Folium                    |
| **SHAP Explainability**      | Penjelasan model prediksi yang interpretable            |
| **Automated Insights**       | Rekomendasi kebijakan berbasis anomali                  |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.13+
- Git

### 1. Setup Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import pandas, prophet, sklearn; print('✅ All packages installed')"
```

---

## 📊 Dataset

### Data yang Digunakan (MVP - 3 Provinsi)

**Provinsi:**

- Jawa Barat
- Jawa Timur
- DKI Jakarta

**Jenis Pendapatan:**

- PBB (Pajak Bumi dan Bangunan)
- Retribusi Pasar
- Pajak Hotel & Restoran

### Download Data Publik

Data tersedia dari sumber publik berikut:

1. **BPS.go.id** - Realisasi Pendapatan Daerah
   - URL: https://www.bps.go.id/
   - Format: CSV/Excel
   - Periode: 2018-2025

2. **data.go.id** - Satu Data Indonesia
   - URL: https://data.go.id/
   - Format: CSV/JSON
   - Indikator: PDB, Populasi, Inflasi

3. **BIG.go.id** - Peta Geospatial
   - URL: https://big.go.id/
   - Format: GeoJSON

### Struktur Data File

```csv
Tahun,Bulan,Provinsi,Jenis_Pendapatan,Realisasi
2022,1,Jawa Barat,PBB,125600000000
2022,1,Jawa Barat,Retribusi Pasar,45300000000
...
```

---

## 📁 Struktur Proyek

```
revdadas/
├── data/
│   ├── raw/                      # Data mentah dari BPS
│   │   └── revenue_data.csv
│   └── processed/                # Data bersih & terproses
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Load & download data
│   ├── preprocessing.py          # Cleaning & feature engineering
│   ├── forecasting.py            # Prophet forecasting model
│   ├── anomaly_detection.py      # Isolation Forest detector
│   └── utils.py                  # Helper functions
│
├── models/
│   ├── forecaster.pkl            # Saved Prophet models
│   └── anomaly_detector.pkl      # Saved detector
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard
│
├── requirements.txt
├── .gitignore
└── README.md                     # Dokumentasi ini
```

---

## 💻 Cara Menggunakan

### 1. Quick Start Guide

```bash
cd revdadas
venv\Scripts\activate
```

Jalankan Dashboard dengan perintah berikut

```bash
cd dashboard
streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`

### 2. Train Forecasting Model

```python
from src import forecasting

# Initialize forecaster
forecaster = forecasting.RevenueForecaster(periods=12)

# Train & forecast
forecaster.train_and_forecast_all(df)
forecast = forecaster.forecast(
    provinsi='Jawa Barat',
    jenis_pajak='PBB'
)

print(forecast.head())
```

### 3. Detect Anomalies

```python
from src import anomaly_detection

# Initialize detector
detector = anomaly_detection.AnomalyDetector()

# Train & detect
detector.train(df)
result = detector.detect(df)

# Get insights
insights = detector.get_anomaly_insights(result, threshold=0.7)
for insight in insights:
    print(insight['Alert'])
```

---

## 🎓 Tech Stack

| Komponen            | Library      | Versi  |
| ------------------- | ------------ | ------ |
| **Data Processing** | Pandas       | 2.1.4  |
| **Numerical**       | NumPy        | 1.24.3 |
| **Forecasting**     | Prophet      | 1.1.5  |
| **ML/Anomaly**      | Scikit-learn | 1.4.2  |
| **Visualization**   | Plotly       | 5.18.0 |
| **Dashboard**       | Streamlit    | 1.28.1 |
| **Geospatial**      | Folium       | 0.14.0 |
| **Explainability**  | SHAP         | 0.43.0 |

---

## 👥 Tim

- **Ketua:** Kwik Andreas Jonathan (322301113)
- **Anggota:**
  - Clay Micholaz Fu
  - Moses Chisthoper Adisam
  - Gwyneth Eunice Widjaja

---

**Last Updated:** March 15, 2026  
**Status:** 🔄 Development (MVP Phase)
