"""
RevDadas Streamlit Dashboard
AI-Driven Predictive Analytics for Fraud Detection and Revenue Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_loader, preprocessing, forecasting, anomaly_detection, utils

# Configure page
st.set_page_config(
    page_title="RevDadas - Revenue Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load or create sample data"""
    loader = data_loader.BPSDataLoader()
    df = loader.create_sample_data()
    
    # Preprocess
    preprocessor = preprocessing.DataPreprocessor()
    df = preprocessor.clean_revenue_data(df)
    df = preprocessor.create_features(df)
    
    return df


@st.cache_data
def train_models(df):
    """Train forecasting and anomaly detection models"""
    
    # Train forecaster
    forecaster = forecasting.RevenueForecaster(periods=12)
    forecast_results = forecaster.train_and_forecast_all(df)
    
    # Train anomaly detector
    detector = anomaly_detection.AnomalyDetector()
    detector.train(df)
    anomaly_results = detector.detect(df)
    
    return forecast_results, anomaly_results, forecaster, detector


def main():
    # Header
    st.title("📊 RevDadas")
    st.subtitle("Predictive Analytics untuk Deteksi Fraud & Prakiraan Revenue Daerah")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Data loading option
        data_source = st.radio(
            "Pilih sumber data:",
            ["📁 Sample Data", "📂 Upload CSV"]
        )
        
        if data_source == "📁 Sample Data":
            st.info("Menggunakan data contoh untuk demo")
            df = load_sample_data()
            st.success("✅ Data berhasil dimuat")
        
        # Provinces filter
        all_provinsi = df['Provinsi'].unique()
        selected_provinsi = st.multiselect(
            "Pilih Provinsi:",
            all_provinsi,
            default=all_provinsi[:1]
        )
        
        # Pajak types filter
        all_pajak = df['Jenis_Pendapatan'].unique()
        selected_pajak = st.multiselect(
            "Pilih Jenis Pendapatan:",
            all_pajak,
            default=all_pajak
        )
    
    # Filter data
    filtered_df = df[
        (df['Provinsi'].isin(selected_provinsi)) &
        (df['Jenis_Pendapatan'].isin(selected_pajak))
    ]
    
    # Train models
    with st.spinner("🔄 Melatih model prediksi..."):
        forecast_results, anomaly_results, forecaster, detector = train_models(filtered_df)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Prakiraan Revenue", "🚨 Deteksi Anomali", "🗺️ Peta Peta Interactive", "📊 Dashboard"]
    )
    
    # TAB 1: Forecasting
    with tab1:
        st.header("Prakiraan Revenue 12 Bulan")
        
        if forecast_results is not None:
            st.dataframe(forecast_results, use_container_width=True)
            
            # Visualization
            for provinsi in selected_provinsi:
                st.subheader(f"📍 {provinsi}")
                
                prov_forecast = forecast_results[forecast_results['Provinsi'] == provinsi]
                prov_actual = filtered_df[filtered_df['Provinsi'] == provinsi]
                
                # Create figure
                fig = go.Figure()
                
                # Add actual data
                fig.add_trace(go.Scatter(
                    x=prov_actual['Tanggal'],
                    y=prov_actual['Realisasi'],
                    mode='lines+markers',
                    name='Realisasi Aktual',
                    line=dict(color='blue')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=prov_forecast['Tanggal'],
                    y=prov_forecast['Prediksi'],
                    mode='lines+markers',
                    name='Prakiraan',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=prov_forecast['Tanggal'],
                    y=prov_forecast['Batas_Atas'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=prov_forecast['Tanggal'],
                    y=prov_forecast['Batas_Bawah'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='Interval Kepercayaan 95%'
                ))
                
                fig.update_layout(
                    title=f"Revenue Forecast - {provinsi}",
                    xaxis_title="Tanggal",
                    yaxis_title="Revenue (Rp)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Gagal melatih model forecast")
    
    # TAB 2: Anomaly Detection
    with tab2:
        st.header("🚨 Deteksi Anomali (Potential Fraud)")
        
        if anomaly_results is not None:
            # Get insights
            anomalies = detector.get_anomaly_insights(anomaly_results, threshold=0.6)
            
            st.metric(
                "Total Anomali Terdeteksi",
                len(anomalies),
                delta=f"{len(anomalies)/len(anomaly_results)*100:.1f}% dari total"
            )
            
            if anomalies:
                st.subheader("Detail Anomali")
                
                anomaly_df = pd.DataFrame(anomalies)
                st.dataframe(anomaly_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    anomaly_df,
                    x='Provinsi',
                    y='Anomaly_Score',
                    color='Alert',
                    title="Skor Anomali per Provinsi",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ Tidak ada anomali terdeteksi")
        else:
            st.error("Gagal melatih model deteksi anomali")
    
    # TAB 3: Interactive Map
    with tab3:
        st.header("🗺️ Peta Interaktif Indonesia")
        st.info("💡 Fitur peta akan ditampilkan dengan warna: Hijau (normal), Kuning (warning), Merah (fraud risk)")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_revenue = filtered_df['Realisasi'].mean()
            st.metric("Rata-rata Revenue", f"Rp {avg_revenue/1e9:.1f}M")
        
        with col2:
            provinces = filtered_df['Provinsi'].nunique()
            st.metric("Jumlah Provinsi", provinces)
        
        with col3:
            pajak_types = filtered_df['Jenis_Pendapatan'].nunique()
            st.metric("Jenis Pajak", pajak_types)
    
    # TAB 4: Dashboard Overview
    with tab4:
        st.header("📊 Dashboard Ringkasan")
        
        # Revenue trend by province
        fig_revenue = px.line(
            filtered_df,
            x='Tanggal',
            y='Realisasi',
            color='Provinsi',
            title="Tren Revenue per Provinsi",
            height=400
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Distribution by pajak type
        fig_pajak = px.box(
            filtered_df,
            x='Jenis_Pendapatan',
            y='Realisasi',
            color='Jenis_Pendapatan',
            title="Distribusi Revenue per Jenis Pajak",
            height=400
        )
        st.plotly_chart(fig_pajak, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **RevDadas v0.1.0** | AI-Driven Predictive Analytics for Local Government Revenue  
    Powered by Prophet, Scikit-learn & Streamlit
    """)


if __name__ == "__main__":
    main()
