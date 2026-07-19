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
        5. Deviasi (selisih absolut vs moving average — untuk KPI)
        6. Jumlah bulan berturut-turut bernilai 0 sebelum baris ini
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

        # Feature 5: Deviasi absolut vs MA (untuk KPI "Nilai Transaksi untuk Ditinjau")
        df['Deviasi'] = (df['Realisasi'] - ma).fillna(0)

        # Feature 6: Jumlah bulan berturut-turut bernilai 0 (atau sangat kecil) sebelumnya.
        # Berguna untuk membedakan "pencairan pertama setelah idle" vs "lonjakan mencurigakan".
        def _consecutive_zeros(series):
            result = []
            count = 0
            for val in series:
                result.append(count)
                if val <= 0:
                    count += 1
                else:
                    count = 0
            return result

        df['Bulan_Nol_Sebelumnya'] = df.groupby(grp_keys)['Realisasi'].transform(
            lambda x: pd.Series(_consecutive_zeros(x.values), index=x.index)
        )

        logger.info("Created 6 features for anomaly detection (per seri)")
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
            # Kritis jika realisasi 0 pada akun utama (non-lumpy)
            if r.get('Realisasi', 1) == 0:
                jenis_lower = r.get('Jenis_Pendapatan', '').lower()
                is_lumpy = any(k in jenis_lower for k in [
                    'hibah', 'tidak terduga', 'modal', 'transfer',
                    'kekayaan daerah', 'lain-lain', 'lainnya', 'darurat'
                ])
                if not is_lumpy:
                    return 'Kritis'

            if r['Anomaly_Score'] >= thr_hi:
                return 'Tinggi'
            if r['Anomaly_Score'] >= thr_md:
                return 'Sedang'
            return 'Rendah'
        df['Severity'] = df.apply(_sev, axis=1)

        # ---- Materiality: share nominal anomali terhadap total provinsi ----
        prov_totals = df.groupby('Provinsi')['Realisasi'].transform('sum')
        df['_prov_share'] = (df['Realisasi'] / prov_totals).replace([np.inf, -np.inf], 0).fillna(0)

        # Alasan ringkas dan klasifikasi Jenis Fraud yang memperhitungkan
        # Target, DNA Wilayah, sifat akun (lumpy), dan materialitas.
        def _reason_and_fraud(r):
            if not r['Anomaly']:
                return pd.Series(['-', '-'])
            
            mom = r.get('MoM_Change', 0)
            seas = r.get('Seasonality_Deviation', 0)
            persen = r.get('Persentase', 0)
            prov = r.get('Provinsi', '')
            jenis = r.get('Jenis_Pendapatan', '')
            bulan_nol = r.get('Bulan_Nol_Sebelumnya', 0)
            prov_share = r.get('_prov_share', 0)
            
            jenis_fraud = "Anomali Statistik"
            alasan = "Pola tidak biasa dibanding tren historis."
            
            # Regional DNA check (contoh untuk bbrp provinsi)
            is_andalan = False
            if prov == 'Bali' and 'Hotel' in jenis: is_andalan = True
            elif prov == 'DKI Jakarta' and ('Kendaraan' in jenis or 'Iklan' in jenis): is_andalan = True
            elif prov == 'Kalimantan Timur' and 'Bahan Bakar' in jenis: is_andalan = True
            
            # Daftar akun yang secara alamiah bersifat lumpy (pencairan tunggal /
            # tidak merata). Mencakup: Hibah, Dana Darurat, Belanja Modal,
            # Transfer, Dividen BUMD (Kekayaan Daerah), Lain-Lain PAD, dsb.
            jenis_lower = jenis.lower()
            is_lumpy = any(k in jenis_lower for k in [
                'hibah', 'tidak terduga', 'modal', 'transfer',
                'kekayaan daerah',   # dividen BUMD
                'lain-lain',         # Lain-Lain PAD yang Sah
                'lainnya',           # Pendapatan Lainnya
                'darurat',           # Dana Darurat
            ])

            # Jika ada 2+ bulan berturut-turut bernilai 0 sebelumnya, dan sekarang
            # muncul nilai — ini pola pencairan terjadwal, bukan manipulasi.
            is_first_disbursement = bulan_nol >= 2 and r.get('Realisasi', 0) > 0

            mom_val = abs(mom)
            mom_str = f"{mom_val:.0f}%" if mom_val <= 999 else f"{mom_val/100:.0f}×"

            # --- Klasifikasi anomali ---

            # 0) Realisasi 0 pada akun utama (indikasi krisis/sistem mati)
            if r.get('Realisasi', 1) == 0 and not is_lumpy:
                jenis_fraud = "Indikasi Data Kosong / Sistem Error"
                alasan = "Realisasi tercatat Rp 0. Sangat tidak wajar untuk akun utama/rutin."
                return pd.Series([jenis_fraud, alasan])

            # 1) Akun lumpy ATAU pencairan pertama setelah idle → wajar
            if is_lumpy or is_first_disbursement:
                jenis_fraud = "Wajar (Transaksi Insidental)"
                if is_first_disbursement:
                    alasan = (f"Pencairan pertama setelah {bulan_nol} bulan idle. "
                              f"Wajar untuk akun bersifat insidental.")
                elif mom > 100:
                    alasan = (f"Pencairan tunggal (lumpy) {mom_str} MoM. "
                              f"Secara bisnis wajar untuk akun non-rutin.")
                elif mom < -30:
                    alasan = "Penurunan pasca-pencairan tunggal. Wajar untuk pola akun insidental."
                else:
                    alasan = "Fluktuasi wajar pada akun bersifat insidental."

            # 2) Penurunan tajam (MoM < -50%)
            elif mom < -50:
                if persen > 90:
                    jenis_fraud = "Wajar (Target Tercapai)"
                    alasan = f"Turun tajam, namun wajar karena pencapaian tahunan sudah {persen:.1f}%."
                elif is_andalan:
                    jenis_fraud = "Fluktuasi Sektor Andalan"
                    alasan = f"Turun {mom_str} MoM pada sektor andalan daerah, kemungkinan faktor musiman/low-season."
                else:
                    jenis_fraud = "Indikasi Under-reporting"
                    alasan = f"Penurunan drastis {mom_str} MoM, padahal pencapaian target baru {persen:.1f}%."

            # 3) Lonjakan besar (MoM > 100%) — threshold dinaikkan dari 50%
            elif mom > 100:
                if persen > 100:
                    jenis_fraud = "Over-realization (Spike)"
                    alasan = f"Lonjakan {mom_str} MoM, pencapaian menembus target ({persen:.1f}%)."
                elif is_andalan:
                    jenis_fraud = "Peak-season Sektor Andalan"
                    alasan = f"Lonjakan {mom_str} MoM wajar karena faktor peak-season sektor unggulan daerah."
                elif prov_share < 0.005:
                    # Anomali pada akun yang share-nya <0.5% dari total provinsi → minor
                    jenis_fraud = "Anomali Minor"
                    alasan = (f"Lonjakan {mom_str} MoM, namun nominal sangat kecil "
                              f"(<0.5% total pendapatan provinsi). Risiko rendah.")
                else:
                    jenis_fraud = "Indikasi Spike / Manipulasi"
                    alasan = (f"Lonjakan {mom_str} MoM pada akun rutin, "
                              f"padahal total pencapaian masih {persen:.1f}%. Perlu verifikasi.")

            # 4) Penyimpangan musiman
            elif abs(seas) >= 40:
                seas_val = abs(seas)
                seas_str = f"{seas_val:.0f}%" if seas_val <= 999 else f"{seas_val/100:.0f}×"
                jenis_fraud = "Penyimpangan Musiman"
                alasan = f"Menyimpang {seas_str} dari pola musiman biasanya."
                
            return pd.Series([jenis_fraud, alasan])
            
        df[['Jenis_Fraud', 'Alasan']] = df.apply(_reason_and_fraud, axis=1)

        # Materiality flag: anomali yang share-nya < 0.5% dari total provinsi
        df['Materiality'] = 'Material'
        df.loc[(df['Anomaly']) & (df['_prov_share'] < 0.005), 'Materiality'] = 'Minor'
        df.loc[~df['Anomaly'], 'Materiality'] = '-'
        
        # Downgrade "Wajar" cases (but NOT "Anomali Minor", leave minor anomalies so detection isn't 0)
        downgrade_mask = df['Jenis_Fraud'].str.startswith('Wajar', na=False)
        df.loc[downgrade_mask, 'Anomaly'] = False
        df.loc[downgrade_mask, 'Severity'] = '-'
        df.loc[downgrade_mask, 'Jenis_Fraud'] = '-'
        df.loc[downgrade_mask, 'Alasan'] = '-'
        df.loc[downgrade_mask, 'Materiality'] = '-'
        
        # Set "Anomali Minor" to severity Rendah manually, so they don't trigger major alarms
        minor_mask = df['Jenis_Fraud'] == 'Anomali Minor'
        df.loc[minor_mask, 'Severity'] = 'Rendah'

        # Bersihkan kolom helper
        df.drop(columns=['_prov_share'], inplace=True, errors='ignore')
        
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
