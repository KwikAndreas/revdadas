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
    
    def __init__(self, data_path=None):
        self.data_path = data_path or utils.get_data_path("raw")
        self.processed_path = utils.get_data_path("processed")
        
    def load_revenue_data(self, filename=None):
        """
        Load revenue data from CSV
        Expected columns: Tahun, Bulan, Provinsi, Jenis_Pendapatan, Realisasi
        
        If filename is None, loads consolidated data from processed folder
        """
        if filename is None:
            # Load consolidated data
            filepath = self.processed_path / "revenue_consolidated.csv"
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
