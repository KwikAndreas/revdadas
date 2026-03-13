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
✅ **Memprediksi revenue** 6-12 bulan ke depan per provinsi menggunakan Prophet  
✅ **Menghasilkan rekomendasi kebijakan** otomatis berbasis data  
✅ **Visualisasi iuteractive** untuk monitoring real-time

**Mendukung:**

- Pajak Bumi dan Bangunan (PBB)
- Retribusi Pasar
- Pajak Hotel & Restoran

---

## ⚡ Fitur Utama

| Fitur                        | Deskripsi                                               |
| ---------------------------- | ------------------------------------------------------- |
| **Forecasting**              | Prakiraan revenue 12 bulan menggunakan Facebook Prophet |
| **Anomaly Detection**        | Deteksi fraud & under-reporting dengan Isolation Forest |
| **Dashboard Interactive**    | Monitoring real-time dengan Streamlit                   |
| **Geospatial Visualization** | Peta heatmap Indonesia dengan Folium                    |
| **SHAP Explainability**      | Penjelasan model prediksi yang interpretable            |
| **Automated Insights**       | Rekomendasi kebijakan berbasis anomali                  |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9+
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

### 1. Quick Start dengan Sample Data

```bash
cd dashboard
streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`

### 2. Load Data Anda Sendiri

```python
from src import data_loader, preprocessing

# Load data
loader = data_loader.BPSDataLoader()
df = loader.load_revenue_data('your_data.csv')

# Preprocess
preprocessor = preprocessing.DataPreprocessor()
df = preprocessor.clean_revenue_data(df)
```

### 3. Train Forecasting Model

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

### 4. Detect Anomalies

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

## 📈 Pipeline Eksekusi

### Week 1: Data Foundation

- [x] Setup project structure
- [x] Create data loader
- [ ] Download BPS data (Anda)
- [ ] EDA & data cleaning

### Week 2: Modeling

- [ ] Train Prophet forecasting
- [ ] Train Isolation Forest anomaly detection
- [ ] Model evaluation & tuning

### Week 3: Integration

- [ ] Feature engineering
- [ ] SHAP explainability
- [ ] Model persistence

### Week 4: Dashboard

- [ ] Build Streamlit UI
- [ ] Interactive visualizations
- [ ] Deployment setup

### Week 5: Polish & Demo

- [ ] Bug fixes & optimization
- [ ] Demo video (90 detik)
- [ ] Final documentation

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

## 📞 Support & FAQ

**Q: Bagaimana cara menambah lebih banyak provinsi?**  
A: Update filter di `sidebar` dashboard atau edit `selected_provinsi` di `app.py`

**Q: Bagaimana accuracy model saya?**  
A: Lihat metrics di output training atau gunakan MAPE/MAE dari Prophet

**Q: Bisa deploy ke cloud?**  
A: Ya! Gunakan Streamlit Cloud (gratis) atau Railway

---

## 📄 Lisensi

Proposal Hackathon Digdaya x Bank Indonesia 2026

---

## 👥 Tim

- **Ketua:** Kwik Andreas Jonathan (322301113)
- **Anggota:**
  - Clay Micholaz Fu
  - Moses Chisthoper Adisam
  - Gwyneth Eunice Widjaja

---

**Last Updated:** March 14, 2026  
**Status:** 🔄 Development (MVP Phase)
