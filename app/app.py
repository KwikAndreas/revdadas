import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import logging
from pathlib import Path
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(
    page_title="RevDadas - Revenue Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src import data_loader, preprocessing, forecasting, anomaly_detection, utils, policy, report, business
except ImportError:
    st.warning("Module src belum ditemukan. Pastikan folder src tersedia untuk menjalankan model.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    /* Reset & General Layout */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 95%; }
    
    /* Header Styling */
    .app-header {
        display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 15px; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px; margin-top: 48px;
    }
    .header-title-box { display: flex; align-items: center; gap: 15px; }
    .logo-box {
        background-color: #b91c1c; color: white; width: 40px; height: 40px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 8px; font-weight: bold; font-size: 20px;
    }
    .app-title { font-weight: 700; color: #1e293b; margin: 0; font-size: 18px; }
    .app-subtitle { color: #64748b; font-size: 12px; margin: 0; letter-spacing: 1px; }
    .header-center { font-weight: 600; color: #475569; font-size: 18px; }
    
    /* KPI Cards Styling */
    .kpi-container {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;
        text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;
    }
    .kpi-title { font-size: 11px; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
    .kpi-value { font-size: 28px; font-weight: 700; margin-bottom: 5px; }
    .kpi-value.dark { color: #0f172a; }
    .kpi-value.red { color: #dc2626; }
    .kpi-value.orange { color: #f97316; }
    .kpi-sub { font-size: 12px; font-weight: 600; }
    .sub-green { color: #10b981; }
    .sub-blue { color: #3b82f6; }
    .sub-gray { color: #64748b; }
    .sub-red { color: #ef4444; }

    /* Impact Calculator Card */
    .impact-card {
        background: #1e3a5f; color: white; padding: 25px;
        border-radius: 12px; height: 100%; position: relative;
    }
    .impact-header { display: flex; align-items: center; gap: 10px; font-weight: 600; margin-bottom: 20px;}
    .impact-title { font-size: 11px; color: #cbd5e1; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;}
    .impact-value { font-size: 36px; font-weight: 800; color: #4ade80; margin: 5px 0 15px 0;}
    .impact-quote {
        background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;
        font-size: 12px; font-style: italic; line-height: 1.5; color: #e2e8f0; margin-bottom: 20px;
    }
    .btn-rekomendasi {
        background-color: #4ade80; color: #1e3a5f; width: 100%; padding: 10px;
        border-radius: 8px; border: none; font-weight: 700; cursor: pointer; text-align: center;
    }

    /* AI Insights Card */
    .insight-card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; margin-top: 20px;
    }
    .insight-header { display: flex; align-items: center; gap: 10px; font-weight: 700; margin-bottom: 15px;}
    .warning-box {
        display: flex; gap: 15px; background: #f8fafc; padding: 15px;
        border-left: 4px solid #e2e8f0; border-radius: 0 8px 8px 0;
    }
    
    /* Section Titles */
    .section-title { font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 15px;}
    
    /* Sidebar overrides */
    .css-1d391kg { padding-top: 2rem; }
    .sidebar-title { font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 10px; margin-top: 20px;}

    /* Tombol aksi header (Refresh / Export PDF) — ditarik ke atas agar sejajar
       dengan baris header. Warna ditentukan oleh type primary/secondary + tema. */
    .hdr-actions { margin-top: -84px; height: 0; }
    .stButton > button {
        border-radius: 6px !important; font-weight: 600 !important;
        padding: 8px 15px !important; box-shadow: none !important;
    }
    .stButton > button:hover { transform: translateY(-1px); }
</style>
<style>
    /* secondary (Refresh): putih + border */
    button[kind="secondary"] { background:#ffffff !important; color:#0f172a !important; border:1px solid #cbd5e1 !important; }
    /* primary (Export PDF): biru tua sesuai desain */
    button[kind="primary"] { background:#1e3a5f !important; color:#ffffff !important; border:1px solid #1e3a5f !important; }

    /* Tombol Rekomendasi: hijau & menempel ke bawah Impact Calculator.
       Marker .rec-btn-zone berada di container tepat sebelum container tombol. */
    .rec-btn-zone { margin-top: -10px; }
    .element-container:has(.rec-btn-zone) + .element-container button {
        background:#4ade80 !important; color:#1e3a5f !important;
        border:none !important; font-weight:700 !important;
        border-radius:8px !important;
    }
    .element-container:has(.rec-btn-zone) + .element-container button:hover {
        background:#3ccb6f !important;
    }
""", unsafe_allow_html=True)

def get_data_file_timestamp():
    try:
        from src import utils
        data_path = Path(utils.get_data_path("processed")) / "revenue_consolidated.csv"
        if data_path.exists():
            return os.path.getmtime(data_path)
    except:
        pass
    return 0

@st.cache_data(show_spinner=False)
def load_data(data_timestamp=None):
    try:
        loader = data_loader.BPSDataLoader()
        df = loader.load_revenue_data()
        
        if df is None:
            df = loader.create_sample_data()
        
        preprocessor = preprocessing.DataPreprocessor()
        df = preprocessor.clean_revenue_data(df)
        df = preprocessor.create_features(df)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        logger.error(f"Data loading error: {e}")
        return None

@st.cache_data(show_spinner=False)
def train_models(_df, forecast_months, cache_key):
    """
    Latih model forecasting + anomaly detection.

    `_df` diberi prefix underscore agar Streamlit tidak ikut meng-hash isi
    DataFrame (sempat membuat hasil lama dipakai ulang sehingga perubahan
    periode tampak tidak berpengaruh). `cache_key` = penanda ringan & eksplisit.
    """
    try:
        forecaster = forecasting.RevenueForecaster(periods=forecast_months)
        forecast_results = forecaster.train_and_forecast_all(_df, run_backtest=True, backtest_horizon=6)
        
        contamination = 0.03 + (forecast_months - 6) * (0.04 / 18)  # Range 0.03-0.07
        detector = anomaly_detection.AnomalyDetector(contamination=max(0.02, min(0.08, contamination)))
        detector.train(_df)
        anomaly_results = detector.detect(_df)
        
        return forecast_results, anomaly_results, forecaster, detector
    except Exception as e:
        st.error(f"❌ Error training models: {e}")
        logger.error(f"Model training error: {e}")
        return None, None, None, None

def format_currency(value):
    """Format value as Rupiah"""
    if value >= 1e12:
        return f"Rp {value/1e12:.1f} T"
    elif value >= 1e9:
        return f"Rp {value/1e9:.1f} M"
    return f"Rp {value:.0f}"

PROVINCE_COORDS = {
    "Aceh": [4.6951, 96.7494], "Sumatera Utara": [2.1154, 99.5451],
    "Sumatera Barat": [-0.7399, 100.8000], "Riau": [0.2933, 101.7068],
    "Kepulauan Riau": [3.9457, 108.1429], "Jambi": [-1.4852, 102.4381],
    "Sumatera Selatan": [-3.3194, 103.9140], "Bangka Belitung": [-2.7411, 106.4406],
    "Bengkulu": [-3.5778, 102.3464], "Lampung": [-4.5586, 105.4068],
    "DKI Jakarta": [-6.2000, 106.8166], "Jawa Barat": [-6.9204, 107.6046],
    "Banten": [-6.4058, 106.0640], "Jawa Tengah": [-7.1500, 110.1403],
    "DI Yogyakarta": [-7.8754, 110.4262], "Jawa Timur": [-7.5361, 112.2384],
    "Bali": [-8.4095, 115.1889], "Nusa Tenggara Barat": [-8.6529, 117.3616],
    "Nusa Tenggara Timur": [-8.6574, 121.0794], "Kalimantan Barat": [-0.2788, 111.4753],
    "Kalimantan Tengah": [-1.6815, 113.3824], "Kalimantan Selatan": [-3.0926, 115.2838],
    "Kalimantan Timur": [0.5387, 116.4194], "Kalimantan Utara": [3.0731, 116.0414],
    "Sulawesi Utara": [0.6247, 123.9750], "Gorontalo": [0.6999, 122.4467],
    "Sulawesi Tengah": [-1.4300, 121.4456], "Sulawesi Barat": [-2.8441, 119.2321],
    "Sulawesi Selatan": [-3.6688, 119.9741], "Sulawesi Tenggara": [-4.1449, 122.1746],
    "Maluku": [-3.2385, 130.1453], "Maluku Utara": [1.5710, 127.8088],
    "Papua": [-4.2699, 138.0804], "Papua Barat": [-1.3361, 133.1747],
}

# Batas geografis Indonesia (untuk mengunci peta agar tidak bisa digeser keluar)
INDONESIA_BOUNDS = [[-11.5, 94.5], [6.5, 141.5]]  # [SW, NE]

def get_coords(prov_name):
    return PROVINCE_COORDS.get(prov_name, [-2.5, 118.0])

# --- MAIN APP ---
def main():
    # Header UI (judul) — tombol aksi diletakkan tepat di bawah, rata kanan,
    # memakai st.button asli (berfungsi) yang disamarkan agar tampak seperti desain.
    st.markdown("""
    <div class="app-header">
        <div class="header-title-box">
            <div class="logo-box">א</div>
            <div>
                <p class="app-title">RevDadas</p>
                <p class="app-subtitle">REVENUE DAERAH CERDAS</p>
            </div>
        </div>
        <div class="header-center">AI-Driven Revenue Forecasting & Fraud Detection</div>
        <div style="width:230px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Tombol aksi nyata (berfungsi) — diposisikan rata kanan seperti di header.
    # Lihat CSS .hdr-actions di atas untuk penyamaran tampilan.
    st.markdown('<div class="hdr-actions"></div>', unsafe_allow_html=True)
    _sp, _cr, _ce = st.columns([7, 1.1, 1.3])
    with _cr:
        refresh_clicked = st.button("🔄 Refresh", key="btn_refresh_real",
                                    width="stretch", type="secondary")
    with _ce:
        export_clicked = st.button("📥 Export PDF", key="btn_export_real",
                                   width="stretch", type="primary")

    if refresh_clicked:
        st.session_state.pop("want_export", None)
        st.cache_data.clear()
        st.rerun()
    if export_clicked:
        # set flag persisten agar tidak hilang saat rerun berikutnya
        st.session_state["want_export"] = True
    
    # Load data Backend with cache invalidation based on file timestamp
    with st.spinner("📥 Loading data..."):
        data_timestamp = get_data_file_timestamp()
        df = load_data(data_timestamp=data_timestamp)
    
    if df is None or df.empty:
        st.error("❌ Failed to load data")
        return
    
    # --- SIDEBAR UI---
    with st.sidebar:
        st.markdown('<p class="sidebar-title">JENIS PENDAPATAN</p>', unsafe_allow_html=True)
        all_pajak = sorted(df['Jenis_Pendapatan'].unique().tolist())
        jenis_opts = ["Semua Pendapatan"] + all_pajak
        selected_jenis = st.selectbox("Jenis Pendapatan", jenis_opts, label_visibility="collapsed")
        
        if selected_jenis == "Semua Pendapatan":
            selected_pajak = all_pajak
        else:
            selected_pajak = [selected_jenis]
        
        st.markdown('<p class="sidebar-title">PROVINSI TARGET</p>', unsafe_allow_html=True)
        all_provinsi = sorted(df['Provinsi'].unique().tolist())
        selected_provinsi = []
        for prov in all_provinsi:
            # tercentang beberapa provinsi untuk mockup
            is_checked = prov in ["DKI Jakarta", "Jawa Barat", "Jawa Timur"]
            if st.checkbox(prov, value=is_checked, key=f"prov_{prov}"):
                selected_provinsi.append(prov)
        if not selected_provinsi:
            selected_provinsi = all_provinsi
        
        # Periode Prediksi (Dynamic Text)
        periode_placeholder = st.empty()
        forecast_months = st.slider("Periode Prediksi", 6, 24, 9, key="slider_periode", label_visibility="collapsed")
        periode_placeholder.markdown(f'''
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px; margin-bottom:10px;">
            <span class="sidebar-title" style="margin:0;">PERIODE PREDIKSI</span>
            <span style="color:#dc2626; font-weight:700; font-size:12px;">{forecast_months} BULAN</span>
        </div>
        ''', unsafe_allow_html=True)
        
        c_kiri, c_kanan = st.columns(2)
        with c_kiri: st.markdown('<span style="font-size:11px; color:#94a3b8;">6 Bln</span>', unsafe_allow_html=True)
        with c_kanan: st.markdown('<span style="font-size:11px; color:#94a3b8; float:right;">24 Bln</span>', unsafe_allow_html=True)
        
        # Sensitivitas / Pencegahan Fraud (Dynamic Text)
        st.markdown("<br>", unsafe_allow_html=True)
        fraud_placeholder = st.empty()
        # Slider UI 1-100%, Backend butuh 0.0 - 1.0
        fraud_val_ui = st.slider("Pencegahan", 1, 100, 5, key="slider_fraud", label_visibility="collapsed")
        fraud_threshold = fraud_val_ui / 100.0  # Konversi untuk backend
        
        fraud_placeholder.markdown(f'''
        <div style="background:#fef2f2; padding:15px; border-radius:8px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:700; color:#b91c1c; margin-bottom:5px;">
                <span>PENCEGAHAN FRAUD</span>
                <span>{fraud_val_ui}%</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('<p style="font-size:10px; color:#ef4444; font-style:italic; margin-top:2px;">*Estimasi efektivitas audit AI</p>', unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Cache clear button
        if st.button("Refresh Data Cache", key="refresh_cache_btn"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown('<p style="font-size:11px; color:#94a3b8;">Status Sistem: <span style="color:#10b981;">● Terkoneksi (Satu Data)</span><br>Last Sync: Mar 2026, 14:20</p>', unsafe_allow_html=True)
    
    # Filter Data Backend
    filtered_df = df[
        (df['Provinsi'].isin(selected_provinsi)) &
        (df['Jenis_Pendapatan'].isin(selected_pajak))
    ]
    
    if filtered_df.empty:
        st.warning("⚠️ No data for selected combination")
        return
    
    # Train Models
    with st.spinner("🔄 AI Model Analyzing Data..."):
        cache_key = (int(forecast_months), tuple(sorted(selected_provinsi)), tuple(sorted(selected_pajak)))
        forecast_results, anomaly_results, forecaster, detector = train_models(filtered_df, forecast_months, cache_key)

    # Akurasi model (dari backtest)
    acc = forecaster.overall_accuracy() if forecaster is not None else None
    acc_table = forecaster.accuracy_summary() if forecaster is not None else None
    akurasi_txt = f"Akurasi backtest {acc['akurasi']:.0f}%" if acc else "Predicted by AI Model"
    
    # Perhitungan KPI dari Model Anda
    total_revenue = filtered_df['Realisasi'].sum()
    forecast_total = forecast_results['Prediksi'].sum() if forecast_results is not None and len(forecast_results) > 0 else 0
    
    # Fix: Hitung jumlah ACTUAL anomalies, bukan jumlah rows
    # anomaly_count = 0
    # if anomaly_results is not None and 'Anomaly' in anomaly_results.columns:
    #     anomaly_count = anomaly_results['Anomaly'].sum()
    
    # anomaly_pct = (anomaly_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    # potential_loss = total_revenue * (anomaly_pct / 100)

    anomaly_count = 0
    potential_loss = 0.0
    anomaly_pct = 0.0

    if anomaly_results is not None and 'Anomaly' in anomaly_results.columns:
        # Filter hanya data yang terdeteksi anomali
        anomalies_only = anomaly_results[anomaly_results['Anomaly'] == True]
        anomaly_count = len(anomalies_only)
        
        # 1. Potential Loss = Total uang dari transaksi yang mencurigakan (anomali)
        potential_loss = anomalies_only['Realisasi'].sum()
        
        # 2. Persentase Risiko = Proporsi nominal uang anomali terhadap total uang
        if total_revenue > 0:
            anomaly_pct = (potential_loss / total_revenue) * 100
    
    # 1. KPI CARDS
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">TOTAL REVENUE (AKTUAL)</div>
            <div class="kpi-value dark">{format_currency(total_revenue)}</div>
            <div class="kpi-sub sub-green">Real Data</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">FORECAST {forecast_months} BULAN</div>
            <div class="kpi-value red">{format_currency(forecast_total)}</div>
            <div class="kpi-sub sub-blue">{akurasi_txt}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">RISIKO FRAUD/ANOMALI</div>
            <div class="kpi-value orange">{anomaly_pct:.1f}%</div>
            <div class="kpi-sub sub-gray">{anomaly_count} records deteksi</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">REVENUE LOSS DETEKSI</div>
            <div class="kpi-value red">{format_currency(potential_loss)}</div>
            <div class="kpi-sub sub-red">Tindakan Diperlukan</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. MIDDLE ROW (Map & Impact Calculator)
    col_map, col_impact = st.columns([6.5, 3.5])
    
    with col_map:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
            <div class="section-title" style="margin:0;">Heatmap Potensi Revenue & Risiko</div>
            <div style="display:flex; gap:10px; font-size:10px; font-weight:700;">
                <span style="background:#dcfce7; color:#166534; padding:3px 8px; border-radius:4px;">OPTIMAL</span>
                <span style="background:#fef08a; color:#854d0e; padding:3px 8px; border-radius:4px;">MODERAT</span>
                <span style="background:#fee2e2; color:#991b1b; padding:3px 8px; border-radius:4px;">KRITIS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Peta Folium
        m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles="CartoDB positron",
                       min_zoom=5, max_bounds=True)
        m.fit_bounds(INDONESIA_BOUNDS)
        m.options["maxBounds"] = INDONESIA_BOUNDS
        m.options["maxBoundsViscosity"] = 1.0
        
        # Ekstrak data per provinsi untuk Map Popup
        for prov in selected_provinsi:
            prov_actual_df = filtered_df[filtered_df['Provinsi'] == prov]
            prov_total_rev = prov_actual_df['Realisasi'].sum()
            
            prov_forecast_val = 0
            if forecast_results is not None:
                prov_forecast_val = forecast_results[forecast_results['Provinsi'] == prov]['Prediksi'].sum()
                
            # Fix: Count actual anomalies, not total rows
            prov_risk_pct = 0
            # if anomaly_results is not None and 'Anomaly' in anomaly_results.columns:
            #     prov_anomalies = anomaly_results[
            #         (anomaly_results['Provinsi'] == prov) & 
            #         (anomaly_results['Anomaly'] == True)
            #     ]
            #     prov_risk_pct = (len(prov_anomalies) / len(prov_actual_df) * 100) if len(prov_actual_df) > 0 else 0

            if anomaly_results is not None and 'Anomaly' in anomaly_results.columns:
                prov_anomalies = anomaly_results[
                    (anomaly_results['Provinsi'] == prov) & 
                    (anomaly_results['Anomaly'] == True)
                ]
                prov_anomali_value = prov_anomalies['Realisasi'].sum()
                if prov_total_rev > 0:
                    prov_risk_pct = (prov_anomali_value / prov_total_rev) * 100
            
            # Tentukan warna heatmap berdasarkan risk
            if prov_risk_pct > 5.0: color = "#ef4444" # Merah
            elif prov_risk_pct > 2.0: color = "#eab308" # Kuning
            else: color = "#22c55e" # Hijau

            coords = get_coords(prov)
            
            popup_html = f"""
            <div style="font-family: 'Segoe UI', sans-serif; width: 170px;">
                <div style="font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 12px;">{prov}</div>
                <div style="font-size: 12px; margin-bottom: 6px; color: #475569;">
                    Rev Aktual: <span style="font-weight: 700; color: #0f172a;">{format_currency(prov_total_rev)}</span>
                </div>
                <div style="font-size: 12px; margin-bottom: 6px; color: #475569;">
                    Forecast: <span style="font-weight: 700; color: #3b82f6;">{format_currency(prov_forecast_val)}</span>
                </div>
                <div style="font-size: 12px; margin-bottom: 15px; color: #475569;">
                    Risiko Fraud: <span style="font-weight: 700; color: #dc2626;">{prov_risk_pct:.1f}%</span>
                </div>
                <button style="width: 100%; padding: 6px 0; background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; color: #64748b; font-size: 11px; font-weight: 600;">
                    Detail Wilayah
                </button>
            </div>
            """
            
            iframe = folium.IFrame(html=popup_html, width=210, height=160)
            popup = folium.Popup(iframe, max_width=210)
            
            folium.CircleMarker(
                location=coords, radius=14, color=color, fill=True, fill_color=color, fill_opacity=0.9, popup=popup
            ).add_to(m)
            folium.CircleMarker(
                location=coords, radius=14, color="white", weight=2, fill=False
            ).add_to(m)

        st_folium(m, height=400, width="100%", returned_objects=[])

    with col_impact:
        # Menghitung potensi tambahan kas berdasarkan slider "Pencegahan Fraud"
        potensi_tambahan = potential_loss * (fraud_val_ui / 100.0)
        
        st.markdown(f"""
        <div class="impact-card">
            <div class="impact-header">
                <span>🧮</span> Impact Calculator
            </div>
            <div class="impact-title">POTENSI TAMBAHAN KAS</div>
            <div class="impact-value">{format_currency(potensi_tambahan)}</div>
            <div class="impact-quote">
                "Dengan menekan fraud sebesar <b>{fraud_val_ui}%</b>, wilayah ini dapat mendanai pembangunan infrastruktur dan fasilitas publik baru."
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tombol Rekomendasi yang BENAR-BENAR berfungsi, didesain menempel ke
        # bagian bawah Impact Calculator (lihat CSS .rec-btn-zone).
        st.markdown('<div class="rec-btn-zone"></div>', unsafe_allow_html=True)
        recs_clicked = st.button("✨ Rekomendasi Kebijakan", key="btn_recs_real",
                                 width="stretch")
        
        # AI Insight Otomatis berdasarkan data Anomaly
        insight_text = "Sistem berjalan optimal. Tidak ada anomali signifikan."
        if anomaly_count > 0 and anomaly_results is not None:
            # Filter hanya anomalies yang terdeteksi (Anomaly == True)
            anomalies_only = anomaly_results[anomaly_results['Anomaly'] == True]
            if len(anomalies_only) > 0:
                top_anomaly = anomalies_only.iloc[0]
                top_prov = top_anomaly.get('Provinsi', 'N/A')
                top_jenis = top_anomaly.get('Jenis_Pendapatan', 'N/A')
                insight_text = f"Terdeteksi diskrepansi data dan anomali pada pencatatan <b>{top_jenis}</b> di wilayah <b>{top_prov}</b>."

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-header">
                <span style="color:#eab308;">⚡</span> AI Insights
            </div>
            <div class="warning-box">
                <div style="color:#f97316; font-size:20px;">!</div>
                <div>
                    <div style="font-size:10px; font-weight:700; color:#94a3b8; margin-bottom:5px;">PERINGATAN ANOMALI</div>
                    <div style="font-size:13px; color:#334155; line-height:1.4;">
                        {insight_text}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. BOTTOM ROW (Charts)
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<div class="section-title">Historical Revenue vs Forecast (Ensemble AI)</div>', unsafe_allow_html=True)
        
        # Agregasi data aktual & forecast berdasarkan Tanggal untuk Grafik
        actual_agg = filtered_df.groupby('Tanggal')['Realisasi'].sum().reset_index()
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=actual_agg['Tanggal'], y=actual_agg['Realisasi']/1e9,
            mode='lines+markers', name='Historical Revenue (M)',
            line=dict(color='#1e3a5f', width=3), marker=dict(size=6)
        ))
        
        if forecast_results is not None and not forecast_results.empty:
            forecast_agg = forecast_results.groupby('Tanggal')['Prediksi'].sum().reset_index()
            fig1.add_trace(go.Scatter(
                x=forecast_agg['Tanggal'], y=forecast_agg['Prediksi']/1e9,
                mode='lines+markers', name='AI Forecast (M)',
                line=dict(color='#b91c1c', width=3, dash='dash'), marker=dict(size=6)
            ))
            
        fig1.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            plot_bgcolor='white', height=300
        )
        fig1.update_yaxes(gridcolor='#f1f5f9')
        st.plotly_chart(fig1, width="stretch")

    with col_chart2:
        st.markdown('<div class="section-title">Proporsi Sumber Pendapatan</div>', unsafe_allow_html=True)
        
        # Agregasi data aktual berdasarkan Jenis Pendapatan
        prop_df = filtered_df.groupby('Jenis_Pendapatan')['Realisasi'].sum().reset_index()
        prop_df = prop_df.sort_values(by='Realisasi', ascending=True) # Sortir untuk Bar chart
        
        colors = ['#64748b', '#facc15', '#4ade80', '#dc2626', '#1e3a5f'] * 2 # Warna berulang jika kategori banyak
         
        fig2 = go.Figure(go.Bar(
            x=prop_df['Realisasi']/1e9, 
            y=prop_df['Jenis_Pendapatan'], 
            orientation='h',
            marker_color=colors[:len(prop_df)]
        ))
        
        fig2.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='white', height=300,
            showlegend=False,
            xaxis_title="Realisasi (Miliar Rp)"
        )
        fig2.update_xaxes(gridcolor='#f1f5f9')
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")
    st.markdown('<div class="section-title">📋 Detailed Data Logs</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab5, tab6, tab4 = st.tabs(["📈 Forecast Data", "🚨 Anomalies", "🎯 Akurasi Model", "💼 Rekomendasi Bisnis", "🎚️ Simulasi What-If", "📚 Metodologi"])
    
    with tab1:
        if forecast_results is not None:
            display_cols = ['Tanggal', 'Provinsi', 'Jenis_Pendapatan', 'Prediksi', 'Batas_Bawah', 'Batas_Atas']
            if 'Metode' in forecast_results.columns:
                display_cols.append('Metode')
            st.dataframe(forecast_results[display_cols], width="stretch")
            st.download_button("⬇️ Unduh Proyeksi (CSV)",
                               data=forecast_results[display_cols].to_csv(index=False).encode("utf-8"),
                               file_name="revdadas_proyeksi.csv", mime="text/csv", key="dl_fc")
    
    with tab2:
        if anomaly_results is not None and 'Anomaly' in anomaly_results.columns:
            an = anomaly_results[anomaly_results['Anomaly'] == True].copy()
            if len(an) > 0:
                if 'Anomaly_Score' in an.columns:
                    an = an.sort_values('Anomaly_Score', ascending=False)
                cols = [c for c in ['Tanggal', 'Provinsi', 'Jenis_Pendapatan',
                                    'Realisasi', 'Severity', 'Alasan'] if c in an.columns]
                st.dataframe(an[cols], width="stretch")
            else:
                st.success("✅ No anomalies detected")
        else:
            st.success("✅ No anomalies detected")

    with tab3:
        if acc_table is not None and len(acc_table) > 0:
            if acc is not None:
                cA, cB, cC = st.columns(3)
                cA.metric("Akurasi Model (median)", f"{acc['akurasi']:.0f}%")
                cB.metric("Seri Andal (WAPE<50%)", f"{int(acc['n_reliable'])}/{acc['n_series']}")
                cC.metric("Median WAPE", f"{acc['median_wape']:.1f}%")
            at = acc_table.copy()
            at['Keandalan'] = at['WAPE'].apply(
                lambda w: '🟢 Andal' if (w is not None and w < 30)
                else ('🟡 Cukup' if (w is not None and w < 50) else '🔴 Lemah'))
            st.dataframe(at[['Provinsi', 'Jenis_Pendapatan', 'Akurasi', 'WAPE', 'sMAPE', 'Keandalan']],
                         width="stretch")
            st.caption("Akurasi & WAPE dihitung lewat backtest holdout 6 bulan terakhir. "
                       "WAPE = Weighted Absolute Percentage Error (semakin kecil semakin baik).")
        else:
            st.info("Akurasi belum tersedia (data uji kurang dari ambang minimum).")

    with tab5:
        st.markdown("""
        <div style="font-size:13px; color:#475569; margin-bottom:18px; line-height:1.6;">
        Skor kelayakan <b>sektor bisnis</b> per provinsi, diturunkan dari kondisi & arah
        kas daerah (porsi Pajak Daerah, kemandirian fiskal, skala ekonomi, tren proyeksi).
        Skor 0&ndash;100 &mdash; semakin tinggi semakin mendukung iklim usaha sektor tersebut.
        </div>
        """, unsafe_allow_html=True)

        biz_recs = business.top_recommendations(filtered_df, forecast_results, top_n=5)
        if not biz_recs:
            st.info("Pilih minimal satu provinsi untuk melihat rekomendasi bisnis.")
        else:
            # Palet warna halus, selaras dengan navy brand (#1e3a5f).
            # Satu warna per label dipakai konsisten untuk dot, bar, dan teks.
            label_color = {
                "Sangat Prospektif": "#0f766e",
                "Prospektif":        "#1e3a5f",
                "Cukup / Hati-hati": "#b45309",
                "Kurang Mendukung":  "#991b1b",
            }
            scored_all = business.score_sectors(filtered_df, forecast_results)
            # Catatan: HTML dibangun TANPA indentasi/leading whitespace.
            # Jika baris HTML dimulai dengan >=4 spasi, Streamlit (CommonMark) akan
            # memperlakukannya sebagai indented code block, sehingga seluruh HTML
            # tampil sebagai teks mentah — itulah bug versi sebelumnya.
            for r in biz_recs:
                sectors = scored_all[r["provinsi"]]
                rows_html = ""
                for s in sectors:
                    c = label_color.get(s["label"], "#475569")
                    rows_html += (
                        f'<div style="display:flex;align-items:center;gap:14px;padding:11px 0;border-top:1px solid #f1f5f9;">'
                        f'<div style="flex:0 0 7px;width:7px;height:7px;border-radius:50%;background:{c};"></div>'
                        f'<div style="flex:0 0 190px;font-size:13px;font-weight:500;color:#0f172a;">{s["sektor"]}</div>'
                        f'<div style="flex:1;min-width:80px;background:#f1f5f9;height:6px;border-radius:4px;overflow:hidden;">'
                        f'<div style="background:{c};width:{s["skor"]}%;height:100%;border-radius:4px;"></div>'
                        f'</div>'
                        f'<div style="flex:0 0 34px;text-align:right;font-size:14px;font-weight:700;color:#0f172a;font-variant-numeric:tabular-nums;">{s["skor"]}</div>'
                        f'<div style="flex:0 0 145px;font-size:11px;color:{c};font-weight:600;text-align:right;letter-spacing:0.2px;">{s["label"]}</div>'
                        f'</div>'
                    )
                card_html = (
                    f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
                    f'padding:18px 22px 14px 22px;margin-bottom:14px;'
                    f'box-shadow:0 1px 2px rgba(15,23,42,0.04);">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:14px;">'
                    f'<div>'
                    f'<div style="font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;">Provinsi</div>'
                    f'<div style="font-size:18px;font-weight:700;color:#0f172a;margin-top:2px;">{r["provinsi"]}</div>'
                    f'</div>'
                    f'<div style="text-align:right;">'
                    f'<div style="font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;">Sektor Unggulan</div>'
                    f'<div style="font-size:14px;font-weight:600;color:#1e3a5f;margin-top:2px;">{r["top_sektor"]}</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:12px;color:#475569;padding:10px 12px;background:#f8fafc;'
                    f'border-left:3px solid #1e3a5f;border-radius:0 6px 6px 0;margin-bottom:4px;">'
                    f'<span style="font-weight:600;color:#334155;">Pendorong:</span> {r["top_alasan"]}'
                    f'</div>'
                    f'{rows_html}'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

            biz_rows = []
            for prov, secs in scored_all.items():
                for s in secs:
                    biz_rows.append({"Provinsi": prov, "Sektor": s["sektor"],
                                     "Skor": s["skor"], "Penilaian": s["label"],
                                     "Pendorong": s["alasan"]})
            biz_df = pd.DataFrame(biz_rows)
            st.download_button("Unduh Rekomendasi Bisnis (CSV)",
                               data=biz_df.to_csv(index=False).encode("utf-8"),
                               file_name="revdadas_rekomendasi_bisnis.csv",
                               mime="text/csv", key="dl_biz")
            st.caption("Indikator makro tingkat provinsi dari data APBD — bukan studi kelayakan usaha. "
                       "Bersifat indikatif sebagai bahan pertimbangan, bukan keputusan final.")

    with tab6:
        st.markdown(
            '<div style="font-size:13px;color:#475569;margin-bottom:14px;line-height:1.6;">'
            'Geser slider untuk mensimulasikan dampak <b>penyesuaian kebijakan</b> pada proyeksi '
            'pendapatan. Skenario diterapkan ke periode forecast (data historis tidak diubah). '
            'Cocok untuk menjawab "bagaimana jika tarif Pajak Daerah naik 10%" atau '
            '"kepatuhan retribusi tumbuh 5%".'
            '</div>',
            unsafe_allow_html=True,
        )

        if forecast_results is None or forecast_results.empty:
            st.info("Proyeksi belum tersedia. Pilih provinsi & jenis pendapatan terlebih dulu di sidebar.")
        else:
            jenis_in_fc = set(forecast_results["Jenis_Pendapatan"].unique().tolist())
            ADJUSTABLE = [
                ("Pajak Daerah", "Pajak Daerah"),
                ("Retribusi Daerah", "Retribusi"),
                ("Lain-Lain PAD yang Sah", "Lain-Lain PAD"),
                ("Pendapatan Transfer Pemerintah Pusat", "Transfer Pusat"),
            ]
            available = [(k, lbl) for k, lbl in ADJUSTABLE if k in jenis_in_fc]

            if not available:
                st.info("Jenis pendapatan utama tidak tersedia pada proyeksi terpilih.")
            else:
                st.markdown(
                    '<div style="font-size:11px;font-weight:600;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:0.8px;margin:6px 0 6px;">'
                    'Penyesuaian Kebijakan (%)</div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(len(available))
                adjustments = {}
                for col, (key, label) in zip(cols, available):
                    with col:
                        adjustments[key] = st.slider(
                            label, min_value=-30, max_value=30, value=0, step=1,
                            format="%d%%", key=f"adj_{key}",
                            help=f"Penyesuaian persentase pada proyeksi {key}",
                        )

                scenario = forecast_results.copy()
                scenario["Prediksi_Skenario"] = scenario["Prediksi"].astype(float)
                for jenis, pct in adjustments.items():
                    if pct != 0:
                        mask = scenario["Jenis_Pendapatan"] == jenis
                        scenario.loc[mask, "Prediksi_Skenario"] = (
                            scenario.loc[mask, "Prediksi"].astype(float) * (1 + pct / 100.0)
                        )
                scenario["Prediksi_Skenario"] = scenario["Prediksi_Skenario"].clip(lower=0)

                base_total = float(forecast_results["Prediksi"].sum())
                scen_total = float(scenario["Prediksi_Skenario"].sum())
                delta = scen_total - base_total
                delta_pct = (delta / base_total * 100.0) if base_total > 0 else 0.0

                k1, k2, k3 = st.columns(3)
                k1.metric("Baseline (Total Proyeksi)", f"Rp {base_total/1e12:,.2f} T")
                k2.metric(
                    "Skenario (Total Proyeksi)",
                    f"Rp {scen_total/1e12:,.2f} T",
                    delta=f"{delta/1e9:+,.1f} M" if abs(delta) > 0 else "0",
                )
                k3.metric("Selisih Relatif", f"{delta_pct:+.2f}%")

                monthly = (
                    scenario.groupby("Tanggal")[["Prediksi", "Prediksi_Skenario"]]
                    .sum().reset_index()
                )
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(
                    x=monthly["Tanggal"], y=monthly["Prediksi"] / 1e9,
                    mode="lines+markers", name="Baseline",
                    line=dict(color="#94a3b8", width=2, dash="dot"),
                    marker=dict(size=5),
                    hovertemplate="%{x|%b %Y}<br>Baseline: Rp %{y:,.1f} M<extra></extra>",
                ))
                fig_sim.add_trace(go.Scatter(
                    x=monthly["Tanggal"], y=monthly["Prediksi_Skenario"] / 1e9,
                    mode="lines+markers", name="Skenario",
                    line=dict(color="#1e3a5f", width=2.5),
                    marker=dict(size=6),
                    hovertemplate="%{x|%b %Y}<br>Skenario: Rp %{y:,.1f} M<extra></extra>",
                ))
                fig_sim.update_layout(
                    height=320, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="white", plot_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                    xaxis_title="", yaxis_title="Total Proyeksi (Miliar Rp)",
                )
                fig_sim.update_xaxes(gridcolor="#f1f5f9")
                fig_sim.update_yaxes(gridcolor="#f1f5f9", tickformat=",.0f")
                st.plotly_chart(fig_sim, width="stretch")

                brk = scenario.groupby("Jenis_Pendapatan").agg(
                    Baseline=("Prediksi", "sum"),
                    Skenario=("Prediksi_Skenario", "sum"),
                ).reset_index()
                brk["Selisih"] = brk["Skenario"] - brk["Baseline"]
                brk["Selisih_pct"] = (brk["Selisih"] / brk["Baseline"].replace(0, np.nan) * 100).fillna(0)
                brk_disp = pd.DataFrame({
                    "Jenis Pendapatan": brk["Jenis_Pendapatan"],
                    "Baseline (Rp Miliar)": (brk["Baseline"] / 1e9).round(1),
                    "Skenario (Rp Miliar)": (brk["Skenario"] / 1e9).round(1),
                    "Selisih (Rp Miliar)": (brk["Selisih"] / 1e9).round(1),
                    "Selisih (%)": brk["Selisih_pct"].round(2),
                }).sort_values("Baseline (Rp Miliar)", ascending=False)
                st.markdown(
                    '<div style="font-size:11px;font-weight:600;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:0.8px;margin:18px 0 6px;">'
                    'Rincian per Jenis Pendapatan</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(brk_disp, width="stretch", hide_index=True)

                st.caption(
                    "Catatan: simulasi mengasumsikan penyesuaian berlaku linear & seragam sepanjang "
                    "periode proyeksi, dan tidak memodelkan efek sekunder (mis. elastisitas konsumsi "
                    "terhadap kenaikan tarif). Indikatif untuk eksplorasi awal, bukan estimasi "
                    "pengaruh kebijakan yang final."
                )

    with tab4:
        st.markdown("""
        **Sumber data.** DJPK Kementerian Keuangan — Portal SIKD
        (`djpk.kemenkeu.go.id/portal/data/apbd`), realisasi bulanan kumulatif 2023-2025.

        **Model.** *Ensemble* Prophet + Naive-Seasonal dengan bobot adaptif per seri
        (dipilih lewat validasi internal). Lebih akurat & stabil dibanding Prophet murni
        pada deret APBD bulanan yang pendek. Prediksi dijaga **non-negatif**.

        **Deteksi anomali.** Isolation Forest multivariat (nilai ternormalisasi, perubahan
        bulanan, rasio ke rata-rata bergerak, deviasi musiman) - dihitung per
        (Provinsi x Jenis Pendapatan) + tingkat keparahan & alasan.

        **Validasi.** Backtest holdout 6 bulan; akurasi memakai **WAPE & sMAPE**
        (robust terhadap nilai kecil).

        **Catatan kejujuran data.**
        - 2021-2022 dikecualikan (SIKD hanya menyimpan angka tahunan, bukan progres bulanan).
        - 2025 kemungkinan masih *preliminer* (sebagian bulan akhir belum final).
        - Pos *lumpy/one-off* (mis. sebagian Lain-Lain PAD, Hibah) sulit diramal -
          keandalannya ditampilkan apa adanya pada tab Akurasi Model, tidak disembunyikan.

        **Rekomendasi Bisnis (per sektor).** Skor sektor diturunkan dari sinyal fiskal
        ternormalisasi antar provinsi terpilih: porsi *Pajak Daerah* (proksi aktivitas
        konsumsi/usaha lokal — pajak hotel, restoran, hiburan ada di sini), *kemandirian
        fiskal* (1 − porsi Transfer Pusat), *skala ekonomi* (pendapatan absolut), dan
        *tren proyeksi*. Tiap sektor punya bobot berbeda atas keempat sinyal (terbuka di
        `src/business.py`). Ini indikator MAKRO tingkat provinsi sebagai bahan
        pertimbangan awal, **bukan** studi kelayakan usaha.
        """)

    # --- REKOMENDASI KEBIJAKAN (dipicu tombol di Impact Calculator) ---
    recommendations = policy.generate_recommendations(
        filtered_df, forecast_results, anomaly_results, fraud_prevention_pct=fraud_val_ui
    )
    if recs_clicked:
        st.session_state["show_recs"] = True
    if st.session_state.get("show_recs"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">✨ Rekomendasi Kebijakan</div>', unsafe_allow_html=True)
        badge_color = {"Tinggi": "#dc2626", "Sedang": "#d97706", "Rendah": "#16a34a"}
        badge_bg = {"Tinggi": "#fee2e2", "Sedang": "#fef3c7", "Rendah": "#dcfce7"}
        for r in recommendations:
            p = r["prioritas"]
            st.markdown(f"""
            <div class="insight-card" style="margin-top:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-weight:700; color:#0f172a; font-size:14px;">{r['judul']}</span>
                    <span style="background:{badge_bg[p]}; color:{badge_color[p]}; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700;">{p}</span>
                </div>
                <div style="font-size:13px; color:#334155; line-height:1.5;">{r['detail']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("Rekomendasi bersifat indikatif sebagai bahan diskusi kebijakan, bukan keputusan final.")

    # --- EXPORT PDF (dipicu tombol "Export PDF" di header) ---
    # Memakai flag persisten di session_state agar tombol unduh TIDAK hilang
    # saat Streamlit melakukan rerun (penyebab umum "diklik tapi tak terjadi apa-apa").
    if st.session_state.get("want_export"):
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            summary = {
                "provinsi": selected_provinsi, "jenis": selected_pajak,
                "periode_bulan": forecast_months, "total_revenue": total_revenue,
                "forecast_total": forecast_total, "anomaly_pct": anomaly_pct,
                "anomaly_count": anomaly_count, "potential_loss": potential_loss,
                "akurasi": acc["akurasi"] if acc else 0,
                "n_series": acc["n_series"] if acc else 0,
                "n_reliable": acc["n_reliable"] if acc else 0,
            }
            pdf_bytes = report.build_pdf(summary, recommendations)
            col_dl1, col_dl2 = st.columns([3, 1])
            with col_dl1:
                st.success("✅ Laporan PDF siap. Klik tombol di kanan untuk mengunduh.")
            with col_dl2:
                clicked_dl = st.download_button(
                    "⬇️ Unduh PDF", data=pdf_bytes,
                    file_name=f"RevDadas_Laporan_{datetime.now():%Y%m%d_%H%M}.pdf",
                    mime="application/pdf", key="dl_pdf", width="stretch", type="primary")
            if clicked_dl:
                st.session_state.pop("want_export", None)
        except Exception as e:
            st.error(f"Gagal membuat PDF: {e}")
            st.session_state.pop("want_export", None)

if __name__ == "__main__":
    main()