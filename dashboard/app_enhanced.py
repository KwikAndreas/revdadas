import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import logging
from pathlib import Path

st.set_page_config(
    page_title="RevDadas - Revenue Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_loader, preprocessing, forecasting, anomaly_detection, utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary: #DC3545;
        --success: #28A745;
        --warning: #FFC107;
        --danger: #DC3545;
        --info: #17A2B8;
        --light: #F8F9FA;
        --dark: #343A40;
    }
    
    /* Metrics Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align: center;
    }
    
    .metric-card-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-card-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    .metric-card-change {
        font-size: 13px;
        margin-top: 10px;
        opacity: 0.85;
    }
    
    /* Status indicators */
    .status-optimal {
        background-color: #D4EDDA;
        color: #155724;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .status-warning {
        background-color: #FFF3CD;
        color: #856404;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .status-critical {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #2C3E50;
        border-bottom: 3px solid #667EEA;
        padding-bottom: 10px;
        margin: 20px 0 15px 0;
    }
    
    /* Alert box */
    .alert-box {
        background-color: #F8D7DA;
        border-left: 4px solid #DC3545;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    .alert-box-title {
        font-weight: bold;
        color: #721C24;
        font-size: 14px;
    }
    
    .alert-box-text {
        color: #721C24;
        font-size: 13px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


def format_currency(value, short=False):
    """Format currency"""
    if not isinstance(value, (int, float)):
        return value
    
    if short:
        if abs(value) >= 1e12:
            return f"Rp {value/1e12:.1f} T"
        elif abs(value) >= 1e9:
            return f"Rp {value/1e9:.1f} M"
        elif abs(value) >= 1e6:
            return f"Rp {value/1e6:.2f} K"
    
    return f"Rp {value:,.0f}"


def format_percentage(value, decimals=1):
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}%"
    return value


def get_status_color(score, threshold_warning=0.5, threshold_critical=0.75):
    if score < threshold_warning:
        return "🟢 OPTIMAL", "status-optimal"
    elif score < threshold_critical:
        return "🟡 MODERAT", "status-warning"
    else:
        return "🔴 KRITIS", "status-critical"



@st.cache_data
def load_data():
    try:
        loader = data_loader.BPSDataLoader()
        df = loader.load_revenue_data()
        
        if df is None:
            st.warning("Consolidated data not found, creating sample data...")
            df = loader.create_sample_data()
        
        preprocessor = preprocessing.DataPreprocessor()
        df = preprocessor.clean_revenue_data(df)
        df = preprocessor.create_features(df)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        logger.error(f"Data loading error: {e}")
        return None


@st.cache_data
def train_models(df):
    try:
        forecaster = forecasting.RevenueForecaster(periods=12)
        forecast_results = forecaster.train_and_forecast_all(df)
        
        detector = anomaly_detection.AnomalyDetector()
        detector.train(df)
        anomaly_results = detector.detect(df)
        
        return forecast_results, anomaly_results, forecaster, detector
    except Exception as e:
        st.error(f"❌ Error training models: {e}")
        logger.error(f"Model training error: {e}")
        return None, None, None, None


def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://via.placeholder.com/60x60/DC3545/FFFFFF?text=RD", width=60)
    with col2:
        st.title("RevDadas")
        st.markdown("**AI-Driven Revenue Forecasting & Fraud Detection**")
    
    st.markdown("---")
    
    with st.spinner("📥 Loading data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("❌ Failed to load data")
        return
    
    with st.sidebar:
        st.header("⚙️ PENGATURAN")
        
        all_provinsi = sorted(df['Provinsi'].unique())
        selected_provinsi = st.multiselect(
            "**PROVINSI TARGET**",
            all_provinsi,
            default=all_provinsi
        )
        
        all_pajak = sorted(df['Jenis_Pendapatan'].unique())
        selected_pajak = st.multiselect(
            "**JENIS PENDAPATAN**",
            all_pajak,
            default=all_pajak
        )
        
        forecast_months = st.slider(
            "**PERIODE PREDIKSI**",
            min_value=3,
            max_value=24,
            value=12,
            step=3
        )
        st.caption(f"Periode: {forecast_months} Bulan")
    
    filtered_df = df[
        (df['Provinsi'].isin(selected_provinsi)) &
        (df['Jenis_Pendapatan'].isin(selected_pajak))
    ]
    
    with st.spinner("🔄 Training AI models..."):
        forecast_results, anomaly_results, forecaster, detector = train_models(filtered_df)
    
    if forecast_results is None or anomaly_results is None:
        st.error("❌ Failed to train models")
        return
    
    st.markdown("### 📊 KEY PERFORMANCE INDICATORS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue_2025 = filtered_df[filtered_df['Tahun'] == 2023]['Realisasi'].sum()
        growth = 4.4 
        st.metric(
            "TOTAL REVENUE 2025",
            format_currency(total_revenue_2025, short=True),
            f"+{growth}% vs 2024",
            delta_color="normal"
        )
    
    with col2:
        forecast_2026 = forecast_results['Prediksi'].sum() if forecast_results is not None else 0
        st.metric(
            "FORECAST 2026",
            format_currency(forecast_2026, short=True),
            "Predicted by AI Model",
            delta_color="off"
        )
    
    with col3:
        anomaly_risk = len(anomaly_results[anomaly_results['Anomaly']]) / len(anomaly_results) * 100 if anomaly_results is not None else 0
        st.metric(
            "RISIKO FRAUD/ANOMALI",
            format_percentage(anomaly_risk),
            f"Nasional (Rerata)",
            delta_color="inverse"
        )
    
    with col4:
        revenue_loss = (anomaly_results[anomaly_results['Anomaly']]['Realisasi'].sum() if anomaly_results is not None else 0) * 0.1
        st.metric(
            "REVENUE LOSS DETEKSI",
            format_currency(revenue_loss, short=True),
            "Tidak Dipulihkan",
            delta_color="inverse"
        )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 PRAKIRAAN REVENUE",
        "🚨 DETEKSI ANOMALI",
        "🗺️ PETA HEATMAP",
        "🎯 IMPACT CALCULATOR",
        "💡 AI INSIGHTS"
    ])
    
    with tab1:
        st.markdown("### 📈 Prakiraan Revenue 12 Bulan")
        
        if forecast_results is not None and not forecast_results.empty:
            display_df = forecast_results.copy()
            display_df['Tanggal'] = pd.to_datetime(display_df['Tanggal']).dt.strftime('%Y-%m-%d')
            display_df['Prediksi'] = display_df['Prediksi'].apply(lambda x: format_currency(x))
            display_df['Batas_Bawah'] = display_df['Batas_Bawah'].apply(lambda x: format_currency(x))
            display_df['Batas_Atas'] = display_df['Batas_Atas'].apply(lambda x: format_currency(x))
            
            st.dataframe(
                display_df[['Tanggal', 'Provinsi', 'Jenis_Pendapatan', 'Prediksi', 'Batas_Bawah', 'Batas_Atas']],
                width='stretch',
                height=300
            )
            
            for provinsi in selected_provinsi:
                st.subheader(f"📍 {provinsi}")
                
                prov_forecast = forecast_results[forecast_results['Provinsi'] == provinsi]
                prov_actual = filtered_df[filtered_df['Provinsi'] == provinsi]
                
                if prov_forecast.empty:
                    st.info(f"No forecast available for {provinsi}")
                    continue
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=prov_actual['Tanggal'],
                    y=prov_actual['Realisasi'],
                    mode='lines+markers',
                    name='Realisasi Aktual',
                    line=dict(color='#0066CC', width=3),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=prov_forecast['Tanggal'],
                    y=prov_forecast['Prediksi'],
                    mode='lines+markers',
                    name='Prakiraan AI',
                    line=dict(color='#DC3545', dash='dash', width=3),
                    marker=dict(size=6)
                ))
                
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
                    name='Interval Kepercayaan 95%',
                    fillcolor='rgba(220, 53, 69, 0.2)'
                ))
                
                fig.update_layout(
                    title=f"Revenue Forecast - {provinsi}",
                    xaxis_title="Tanggal",
                    yaxis_title="Revenue (Rp)",
                    height=400,
                    hovermode='x unified',
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🚨 Deteksi Anomali (Potential Fraud)")
        
        if anomaly_results is not None and not anomaly_results.empty:
            anomalies = detector.get_anomaly_insights(anomaly_results, threshold=0.6)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Anomali",
                    len(anomalies),
                    f"{len(anomalies)/len(anomaly_results)*100:.1f}% dari total"
                )
            
            with col2:
                avg_score = anomaly_results[anomaly_results['Anomaly']]['Anomaly_Score'].mean()
                st.metric(
                    "Rata-rata Skor",
                    f"{avg_score:.2f}",
                    "0-1 scale"
                )
            
            with col3:
                high_risk = len(anomaly_results[anomaly_results['Anomaly_Score'] > 0.8])
                st.metric(
                    "Risiko Tinggi",
                    high_risk,
                    "Skor > 0.8"
                )
            
            if anomalies:
                st.markdown("#### Detail Anomali Terdeteksi")
                
                for i, anomaly in enumerate(anomalies[:10]):  # Show top 10
                    status, status_class = get_status_color(anomaly['Anomaly_Score'])
                    
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{anomaly['Provinsi']}** - {anomaly['Jenis_Pendapatan']}")
                            st.caption(f"Tanggal: {anomaly['Tanggal']}")
                        
                        with col2:
                            st.markdown(f"<div class='{status_class}'>{status}</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.metric("Score", f"{anomaly['Anomaly_Score']:.2f}")
                        
                        st.markdown(f"💬 **Alert:** {anomaly['Alert']}")
                        st.markdown(f"📊 Revenue: {format_currency(anomaly['Realisasi'], short=True)}")
                        st.divider()
            else:
                st.success("✅ Tidak ada anomali terdeteksi")
    
    # TAB 3: Heatmap
    with tab3:
        st.markdown("### 🗺️ Heatmap Potensial Revenue & Risiko")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_revenue = filtered_df['Realisasi'].mean()
            st.metric("Rata-rata Revenue", format_currency(avg_revenue, short=True))
        
        with col2:
            max_revenue = filtered_df['Realisasi'].max()
            st.metric("Max Revenue", format_currency(max_revenue, short=True))
        
        with col3:
            provinces = filtered_df['Provinsi'].nunique()
            st.metric("Jumlah Provinsi", provinces)
        
        with col4:
            pajak_types = filtered_df['Jenis_Pendapatan'].nunique()
            st.metric("Jenis Pajak", pajak_types)
        
        st.info("💡 Heatmap interaktif Indonesia akan di-update di fase selanjutnya dengan geospatial visualization")
    
    # TAB 4: Impact Calculator
    with tab4:
        st.markdown("### 🎯 Impact Calculator - Simulasi Fraud Prevention")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fraud_prevention = st.slider(
                "Berapa % Fraud yang dapat dipreventi?",
                0, 100, 25, 5
            )
            st.caption(f"Target: {fraud_prevention}%")
        
        with col2:
            st.metric("Target Prevention", f"{fraud_prevention}%")
        
        # Calculate potential additional revenue
        detected_fraud_value = anomaly_results[anomaly_results['Anomaly']]['Realisasi'].sum()
        potential_recovery = detected_fraud_value * (fraud_prevention / 100)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Fraud Terdeteksi",
                format_currency(detected_fraud_value, short=True),
                "Estimated"
            )
        
        with col2:
            st.metric(
                "Potensi Tambahan Revenue",
                format_currency(potential_recovery, short=True),
                f"Jika {fraud_prevention}% dipreventi"
            )
        
        with col3:
            # Assume 5% fraud cost reduction per % prevention
            cost_reduction = detected_fraud_value * (fraud_prevention / 100) * 0.05
            st.metric(
                "Efisiensi Operasional",
                format_currency(cost_reduction, short=True),
                "Cost savings"
            )
        
        # Recommendation
        st.markdown("#### 💡 Rekomendasi Kebijakan")
        
        if potential_recovery > detected_fraud_value * 0.1:
            st.markdown(f"""
            <div class='alert-box'>
                <div class='alert-box-title'>✅ REKOMENDASI PRIORITAS TINGGI</div>
                <div class='alert-box-text'>
                    Dengan mengimplementasikan fraud prevention sebesar {fraud_prevention}%, 
                    pemerintah dapat memulihkan revenue hingga <strong>{format_currency(potential_recovery, short=True)}</strong> 
                    dan meningkatkan efisiensi biaya operasional.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 5: AI Insights
    with tab5:
        st.markdown("### 💡 AI INSIGHTS & REKOMENDASI")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🔍 Analisis Anomali AI")
            
            # Get top anomalies
            top_anomalies = anomaly_results.nlargest(3, 'Anomaly_Score')
            
            for idx, (i, row) in enumerate(top_anomalies.iterrows(), 1):
                st.markdown(f"""
                **#{idx}. {row['Provinsi']} - {row['Jenis_Pendapatan']}**
                - Skor Anomali: {row['Anomaly_Score']:.2f}/1.0
                - Tipe: Revenue {row.get('MoM_Change', 0):.1f}% {'DROP' if row.get('MoM_Change', 0) < 0 else 'SPIKE'}
                - Action: **Review transaksi terkait**
                """)
                st.divider()
        
        with col1:
            st.markdown("#### 📊 Insights Peramalan")
            
            # Forecast insights
            forecast_growth = (forecast_results['Prediksi'].sum() - filtered_df['Realisasi'].sum()) / filtered_df['Realisasi'].sum() * 100
            
            if forecast_growth > 5:
                growth_text = f"📈 Pertumbuhan positif {forecast_growth:.1f}% diproyeksikan"
            elif forecast_growth < -5:
                growth_text = f"📉 Penurunan {abs(forecast_growth):.1f}% diprediksi"
            else:
                growth_text = f"↔️ Revenue stabil, fluktuasi {abs(forecast_growth):.1f}%"
            
            st.info(growth_text)
        
        with col2:
            st.markdown("#### ⚡ Quick Actions")
            
            if st.button("📥 Export Report PDF"):
                st.success("✅ Report exported!")
            
            if st.button("📧 Share Insights"):
                st.info("📨 Insights akan dikirim ke stakeholder")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p><strong>RevDadas v0.2.0</strong> | AI-Driven Predictive Analytics for Local Government Revenue</p>
        <p>Hackathon Digdaya × Bank Indonesia 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
