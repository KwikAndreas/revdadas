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
    from src import data_loader, preprocessing, forecasting, anomaly_detection, utils
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
</style>
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
def train_models(df, forecast_months):
    try:
        forecaster = forecasting.RevenueForecaster(periods=forecast_months)
        forecast_results = forecaster.train_and_forecast_all(df)
        
        contamination = 0.03 + (forecast_months - 6) * (0.04 / 18)  # Range 0.03-0.07
        detector = anomaly_detection.AnomalyDetector(contamination=contamination)
        detector.train(df)
        anomaly_results = detector.detect(df)
        
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
    "DKI Jakarta": [-6.2000, 106.8166],
    "Jawa Barat": [-6.9204, 107.6046],
    "Jawa Timur": [-7.5361, 112.2384],
}

def get_coords(prov_name):
    return PROVINCE_COORDS.get(prov_name, [-0.7893, 113.9213]) 

# --- MAIN APP ---
def main():
    # Header UI
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
        <div style="display:flex; gap:10px;">
            <button style="padding:8px 15px; border-radius:6px; border:1px solid #cbd5e1; background:white; cursor:pointer;">🔄 Refresh</button>
            <button style="padding:8px 15px; border-radius:6px; border:none; background:#1e3a5f; color:white; cursor:pointer;">📥 Export PDF</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
        selected_jenis = st.selectbox("", jenis_opts, label_visibility="collapsed")
        
        if selected_jenis == "Semua Pendapatan":
            selected_pajak = all_pajak
        else:
            selected_pajak = [selected_jenis]
        
        st.markdown('<p class="sidebar-title">PROVINSI TARGET</p>', unsafe_allow_html=True)
        all_provinsi = sorted(df['Provinsi'].unique().tolist())
        selected_provinsi = []
        for prov in all_provinsi:
            # tercentang beberapa provinsi untuk mockup
            is_checked = prov in ["DKI Jakarta", "Jawa Barat", "Jawa Tengah"]
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
        if st.button("🔄 Refresh Data Cache", key="refresh_cache_btn"):
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
        forecast_results, anomaly_results, forecaster, detector = train_models(filtered_df, forecast_months)
    
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
            <div class="kpi-sub sub-blue">Predicted by AI Model</div>
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
        m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles="CartoDB positron")
        
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
            <div class="btn-rekomendasi">✨ Rekomendasi Kebijakan</div>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown('<div class="section-title">Historical Revenue vs Forecast (AI-LSTM)</div>', unsafe_allow_html=True)
        
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
        st.plotly_chart(fig1, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📋 Detailed Data Logs</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📈 Forecast Data", "🚨 Anomalies"])
    
    with tab1:
        if forecast_results is not None:
            display_cols = ['Tanggal', 'Provinsi', 'Jenis_Pendapatan', 'Prediksi', 'Batas_Bawah', 'Batas_Atas']
            st.dataframe(forecast_results[display_cols], use_container_width=True)
    
    with tab2:
        if len(anomaly_results) > 0:
            anomaly_df = pd.DataFrame(anomaly_results)
            st.dataframe(anomaly_df, use_container_width=True)
        else:
            st.success("✅ No anomalies detected")

if __name__ == "__main__":
    main()