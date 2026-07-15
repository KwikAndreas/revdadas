"""
Data loader for BPS revenue data
Handles downloading and loading data from various sources
"""

import pandas as pd
import logging
from pathlib import Path
from . import utils

logger = logging.getLogger(__name__)


class BPSDataLoader:
    """
    Load data from BPS (Badan Pusat Statistik)
    
    For MVP, we'll work with manually downloaded CSV files from:
    https://www.bps.go.id/
    
    Key datasets:
    - Realisasi Pendapatan Daerah per Jenis Pendapatan
    - Indikator Makroekonomi Daerah
    """
    
    # Nama file mentah APBD DJPK (sumber data asli)
    APBD_MASTER_FILENAME = "apbd_master_2021_2025.csv"

    def __init__(self, data_path=None):
        self.data_path = data_path or utils.get_data_path("raw")
        self.processed_path = utils.get_data_path("processed")

    def build_consolidated_from_apbd(self):
        """
        Bangun revenue_consolidated.csv dari data mentah APBD 38 Provinsi
        (data/raw/apbd_master_2021_2025.csv) bila tersedia.

        Mengembalikan DataFrame, atau None jika file mentah tidak ada.
        """
        master_path = self.data_path / self.APBD_MASTER_FILENAME
        if not master_path.exists():
            logger.warning(f"Data mentah APBD tidak ditemukan: {master_path}")
            return None

        out_path = self.processed_path / "revenue_consolidated.csv"
        
        # Load the new CSV structure
        df = pd.read_csv(master_path)
        
        # Unpivot the aggregated columns into "Jenis_Pendapatan"
        df_melt = pd.melt(
            df, 
            id_vars=['tahun', 'bulan', 'provinsi'], 
            value_vars=['PAD', 'TKDD', 'pendapatan_daerah', 'belanja_daerah', 'belanja_modal'],
            var_name='Jenis_Pendapatan', 
            value_name='Realisasi'
        )
        
        # Rename columns to match existing pipeline
        df_melt.rename(columns={
            'tahun': 'Tahun', 
            'bulan': 'Bulan', 
            'provinsi': 'Provinsi'
        }, inplace=True)
        
        # Create 'Tanggal' column
        df_melt['Tanggal'] = pd.to_datetime(
            df_melt['Tahun'].astype(str) + '-' + df_melt['Bulan'].astype(str) + '-01'
        )
        
        # Friendly names mapping for UI
        mapping = {
            'PAD': 'Pendapatan Asli Daerah (PAD)',
            'TKDD': 'Transfer ke Daerah dan Dana Desa (TKDD)',
            'pendapatan_daerah': 'Total Pendapatan Daerah',
            'belanja_daerah': 'Total Belanja Daerah',
            'belanja_modal': 'Belanja Modal'
        }
        df_melt['Jenis_Pendapatan'] = df_melt['Jenis_Pendapatan'].map(mapping)
        
        # Filter invalid
        df_melt['Realisasi'] = pd.to_numeric(df_melt['Realisasi'], errors='coerce').fillna(0.0)
        
        # Sort values
        df_melt = df_melt.sort_values(['Provinsi', 'Jenis_Pendapatan', 'Tahun', 'Bulan']).reset_index(drop=True)
        
        # DECUMULATE: The DJPK data is Year-To-Date cumulative. 
        # We must decumulate it to discrete monthly values.
        def decumulate(group):
            return group.diff().fillna(group)
            
        df_melt['Realisasi'] = df_melt.groupby(['Provinsi', 'Jenis_Pendapatan', 'Tahun'])['Realisasi'].transform(decumulate)
        
        # Clip to 0 (sometimes data corrections by govt make the diff negative)
        df_melt['Realisasi'] = df_melt['Realisasi'].clip(lower=0.0)
        
        df_melt.to_csv(out_path, index=False)
        logger.info(f"revenue_consolidated.csv dibangun dari data APBD asli "
                    f"({len(df_melt)} baris).")
        return df_melt

    def load_revenue_data(self, filename=None):
        """
        Load revenue data from CSV
        Expected columns: Tahun, Bulan, Provinsi, Jenis_Pendapatan, Realisasi
        
        If filename is None, loads consolidated data from processed folder.
        Jika consolidated belum ada, otomatis dibangun dari data APBD asli
        (data/raw/apbd_djpk_master_2021-2025.csv).
        """
        if filename is None:
            # Load consolidated data
            filepath = self.processed_path / "revenue_consolidated.csv"
            # Auto-build dari data APBD asli bila consolidated belum ada
            if not filepath.exists():
                logger.info("revenue_consolidated.csv belum ada. "
                            "Mencoba membangun dari data APBD asli...")
                built = self.build_consolidated_from_apbd()
                if built is not None:
                    return built
        else:
            filepath = self.data_path / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            # Convert Tanggal to datetime
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def load_makro_indicators(self, filename):
        """
        Load macroeconomic indicators
        Expected columns: Tahun, Provinsi, PDB, Populasi, Inflasi, Pengangguran
        """
        filepath = self.data_path / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None
    
    def create_sample_data(self):
        """
        Create sample data for testing
        This is temporary - replace with real BPS data
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        # Sample data untuk 3 provinsi + 3 jenis pajak
        provinsi = ["Jawa Barat", "Jawa Timur", "DKI Jakarta"]
        jenis_pajak = ["PBB", "Retribusi Pasar", "Pajak Hotel"]
        
        data = []
        start_date = datetime(2022, 1, 1)
        
        for prov in provinsi:
            for pajak in jenis_pajak:
                # Generate 36 bulan data dengan trend dan seasonality
                base_value = np.random.uniform(50, 150) * 1e9  # Rp 50-150 Miliar
                
                for month in range(36):
                    date = start_date + timedelta(days=30*month)
                    # Trend + Seasonality
                    trend = base_value * (1 + 0.05 * (month / 36))
                    seasonality = base_value * 0.1 * np.sin(2 * np.pi * month / 12)
                    noise = np.random.normal(0, base_value * 0.05)
                    
                    value = trend + seasonality + noise
                    
                    data.append({
                        "Tahun": date.year,
                        "Bulan": date.month,
                        "Provinsi": prov,
                        "Jenis_Pendapatan": pajak,
                        "Realisasi": max(0, value)  # Ensure non-negative
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample data with {len(df)} rows")
        return df
    
    def save_processed_data(self, df, filename):
        """Save processed data to CSV"""
        filepath = self.processed_path / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")


# Convenience function
def load_data(filename, data_path=None):
    """Quick load function"""
    loader = BPSDataLoader(data_path)
    return loader.load_revenue_data(filename)
