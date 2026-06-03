"""
Anomaly detection module for fraud detection in revenue data
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from . import utils

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detect anomalies (potential fraud) in revenue data using Isolation Forest
    
    Anomalies detected:
    1. Sudden drops in revenue (under-reporting)
    2. Unusual spikes (data manipulation)
    3. Statistical outliers (inconsistent patterns)
    """
    
    def __init__(self, contamination=0.05):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected fraction of outliers (default: 5%)
        """
        self.contamination = contamination
        self.detectors = {}  # Store detectors for each province
        self.scaler = StandardScaler()
        self.features_ = None
        
    def create_features(self, df):
        """
        Create features for anomaly detection
        
        Features:
        1. Revenue value (normalized)
        2. Month-over-month change (%)
        3. Revenue ratio to moving average
        4. Seasonality deviation
        """
        df = df.copy()

        # Sort by series + date agar perhitungan sekuensial benar
        sort_keys = ['Provinsi', 'Jenis_Pendapatan', 'Tanggal'] if 'Jenis_Pendapatan' in df.columns else ['Provinsi', 'Tanggal']
        df = df.sort_values(sort_keys).reset_index(drop=True)

        # Kelompokkan per SERI (provinsi + jenis pendapatan). Sebelumnya hanya per
        # provinsi sehingga MoM/MA mencampur jenis pendapatan berbeda dan memunculkan
        # "lonjakan" semu. Pengelompokan per seri membuat anomali jauh lebih bermakna.
        grp_keys = ['Provinsi', 'Jenis_Pendapatan'] if 'Jenis_Pendapatan' in df.columns else ['Provinsi']

        # Feature 1: Revenue ternormalisasi dalam tiap seri
        df['Revenue_Norm'] = df.groupby(grp_keys)['Realisasi'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # Feature 2: Perubahan bulan-ke-bulan (%) per seri
        df['MoM_Change'] = df.groupby(grp_keys)['Realisasi'].pct_change() * 100
        df['MoM_Change'] = df['MoM_Change'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Feature 3: Rasio terhadap rata-rata bergerak 3 bulan per seri
        ma = df.groupby(grp_keys)['Realisasi'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['Ratio_to_MA'] = (df['Realisasi'] / ma).replace([np.inf, -np.inf], 1).fillna(1)

        # Feature 4: Deviasi musiman dari rata-rata bulan yang sama per seri
        seas_keys = grp_keys + ['Bulan'] if 'Bulan' in df.columns else grp_keys
        month_avg = df.groupby(seas_keys)['Realisasi'].transform('mean')
        df['Seasonality_Deviation'] = ((df['Realisasi'] - month_avg) / month_avg.abs() * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.info("Created 4 features for anomaly detection (per seri)")
        return df
    
    def train(self, df):
        """
        Train Isolation Forest detector on entire dataset
        """
        # Create features
        df = self.create_features(df)
        
        # Select feature columns
        feature_cols = ['Revenue_Norm', 'MoM_Change', 'Ratio_to_MA', 'Seasonality_Deviation']
        X = df[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train detector
        try:
            detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            detector.fit(X_scaled)
            
            # Store detector
            self.detectors['global'] = detector
            self.features_ = feature_cols
            
            logger.info("Global anomaly detector trained")
            return True
        
        except Exception as e:
            logger.error(f"Error training detector: {e}")
            return False
    
    def detect(self, df):
        """
        Detect anomalies in dataframe
        
        Returns dataframe with anomaly flags and scores
        """
        if 'global' not in self.detectors:
            logger.warning("Detector not trained. Call train() first.")
            return None
        
        # Create features
        df = self.create_features(df)
        
        # Select feature columns
        X = df[self.features_].fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        detector = self.detectors['global']
        anomaly_labels = detector.predict(X_scaled)
        anomaly_scores = detector.score_samples(X_scaled)
        
        # Add to dataframe
        df['Anomaly'] = anomaly_labels == -1  # -1 = anomaly, 1 = normal
        df['Anomaly_Score'] = -anomaly_scores  # higher = more anomalous

        # Tingkat keparahan berdasarkan persentil skor pada baris anomali
        if df['Anomaly'].any():
            sc = df.loc[df['Anomaly'], 'Anomaly_Score']
            thr_hi, thr_md = sc.quantile(0.66), sc.quantile(0.33)
        else:
            thr_hi = thr_md = 0
        def _sev(r):
            if not r['Anomaly']:
                return '-'
            if r['Anomaly_Score'] >= thr_hi:
                return 'Tinggi'
            if r['Anomaly_Score'] >= thr_md:
                return 'Sedang'
            return 'Rendah'
        df['Severity'] = df.apply(_sev, axis=1)

        # Alasan ringkas yang dapat dibaca manusia
        def _reason(r):
            if not r['Anomaly']:
                return '-'
            mom = r.get('MoM_Change', 0)
            seas = r.get('Seasonality_Deviation', 0)
            # Bila persentase meledak (baseline mendekati nol), jangan tampilkan angka konyol
            if abs(mom) >= 30:
                arah = 'lonjakan' if mom > 0 else 'penurunan'
                if abs(mom) > 500:
                    return f"{arah} nilai sangat ekstrem dibanding bulan sebelumnya"
                return f"{arah} tajam {abs(mom):.0f}% dari bulan sebelumnya"
            if abs(seas) >= 40:
                if abs(seas) > 500:
                    return "menyimpang sangat jauh dari pola musiman biasanya"
                return f"menyimpang {abs(seas):.0f}% dari pola musiman biasanya"
            return "pola tidak biasa dibanding tren historis"
        df['Alasan'] = df.apply(_reason, axis=1)
        
        n_anomalies = df['Anomaly'].sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        return df
    
    def get_anomaly_insights(self, df, threshold=0.7):
        """
        Generate insights about detected anomalies
        
        Args:
            df: Dataframe with anomaly labels and scores
            threshold: Anomaly score threshold (0-1)
            
        Returns:
            List of anomaly reports
        """
        anomalies = df[df['Anomaly_Score'] > threshold].copy()
        
        reports = []
        for _, row in anomalies.iterrows():
            report = {
                'Provinsi': row.get('Provinsi', 'Unknown'),
                'Jenis_Pendapatan': row.get('Jenis_Pendapatan', 'Unknown'),
                'Tanggal': row.get('Tanggal', 'Unknown'),
                'Realisasi': row.get('Realisasi', 0),
                'Anomaly_Score': row['Anomaly_Score'],
                'MoM_Change': row.get('MoM_Change', 0),
                'Seasonality_Deviation': row.get('Seasonality_Deviation', 0),
            }
            
            # Generate alert message
            if abs(row.get('MoM_Change', 0)) > 20:
                report['Alert'] = f"Penurunan/Kenaikan tajam {abs(row['MoM_Change']):.1f}%"
            elif abs(row.get('Seasonality_Deviation', 0)) > 30:
                report['Alert'] = f"Deviasi pola musiman {abs(row['Seasonality_Deviation']):.1f}%"
            else:
                report['Alert'] = "Anomali terdeteksi"
            
            reports.append(report)
        
        return reports
    
    def save_detector(self, path=None):
        """Save trained detector"""
        import pickle
        
        path = path or utils.get_models_path()
        filepath = Path(path) / "anomaly_detector.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'detector': self.detectors,
                'scaler': self.scaler,
                'features': self.features_
            }, f)
        
        logger.info(f"Detector saved to {filepath}")
    
    def load_detector(self, path=None):
        """Load saved detector"""
        import pickle
        
        path = path or utils.get_models_path()
        filepath = Path(path) / "anomaly_detector.pkl"
        
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.detectors = data['detector']
                self.scaler = data['scaler']
                self.features_ = data['features']
            logger.info(f"Detector loaded from {filepath}")
            return True
        
        return False


# Convenience function
def detect_anomalies(df):
    """Quick anomaly detection function"""
    detector = AnomalyDetector()
    detector.train(df)
    return detector.detect(df)
