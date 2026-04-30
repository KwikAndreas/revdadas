"""
Data preprocessing and cleaning for RevDadas
"""

import pandas as pd
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data cleaning and transformation"""
    
    def __init__(self):
        self.processed_data = None
        
    def clean_currency_string(self, value):
        """Clean currency strings to numeric format
        
        Handles:
        - Indonesian format: 1.234.567,89
        - English format: 1,234,567.89  
        - Rupiah symbol: Rp 1.234.567,89
        - Parentheses for negative: (1234567)
        """
        if pd.isna(value) or value == "-":
            return None
        
        value = str(value).strip()
        
        # Remove Rp symbol and currency text
        value = re.sub(r'[Rp\s]', '', value)
        
        if not value:
            return None
        
        # Handle negative in parentheses
        multiplier = 1
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1]
            multiplier = -1
        
        # Remove quotes and spaces
        value = value.replace('"', '').replace(" ", "")
        
        # Detect decimal separator
        dot_count = value.count(".")
        comma_count = value.count(",")
        
        if comma_count > 0 and dot_count > 0:
            # Both present: last position is usually decimal
            if value.rfind(",") > value.rfind("."):
                # Comma is last: Indonesian format (1.234.567,89)
                value = value.replace(".", "").replace(",", ".")
            else:
                # Dot is last: English format (1,234,567.89)
                value = value.replace(",", "")
        elif comma_count > 0:
            # Only commas: check position
            if value.count(",") > 1 or len(value.split(",")[-1]) == 2:
                # Multiple commas or 2 decimals = Indonesian
                value = value.replace(".", "").replace(",", ".")
            # else: single comma, single decimal = English format
        
        try:
            return float(value) * multiplier
        except ValueError:
            logger.warning(f"Could not convert currency '{value}' to numeric")
            return None
    
    def clean_revenue_data(self, df):
        """
        Clean and prepare revenue data
        
        Steps:
        1. Clean currency format in Realisasi column
        2. Handle missing values
        3. Detect and handle outliers
        4. Ensure data types
        5. Sort by date
        """
        df = df.copy()
        logger.info("Starting data cleaning...")
        
        # 1. Clean currency strings FIRST
        if 'Realisasi' in df.columns:
            logger.info("Cleaning currency format in Realisasi column...")
            df['Realisasi'] = df['Realisasi'].apply(self.clean_currency_string)
        
        # 2. Handle missing values
        logger.info(f"Missing values before: {df.isnull().sum().sum()}")
        df = df.ffill().df.bfill()
        logger.info(f"Missing values after: {df.isnull().sum().sum()}")
        
        # 3. Ensure correct data types
        if 'Tahun' in df.columns:
            df['Tahun'] = df['Tahun'].astype(int)
        if 'Bulan' in df.columns:
            df['Bulan'] = df['Bulan'].astype(int)
        if 'Realisasi' in df.columns:
            df['Realisasi'] = pd.to_numeric(df['Realisasi'], errors='coerce')
        
        # 4. Create date column
        df['Tanggal'] = pd.to_datetime(
            df['Tahun'].astype(str) + '-' + df['Bulan'].astype(str).str.zfill(2) + '-01'
        )
        
        # 5. Remove rows with invalid data
        df = df.dropna(subset=['Realisasi'])
        df = df[df['Realisasi'] > 0]
        
        # 6. Sort by date
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. Shape: {df.shape}")
        logger.info(f"Realisasi range: {df['Realisasi'].min():.0f} - {df['Realisasi'].max():.0f}")
        self.processed_data = df
        return df
    
    def detect_outliers(self, df, column='Realisasi', method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score method
        
        Returns dataframe with outlier flag
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df['is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            df['is_outlier'] = np.abs((df[column] - mean) / std) > threshold
        
        n_outliers = df['is_outlier'].sum()
        logger.info(f"Detected {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
        
        return df
    
    def aggregate_by_period(self, df, period='M'):
        """
        Aggregate data by time period
        period: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
        """
        if 'Tanggal' not in df.columns:
            logger.error("'Tanggal' column not found")
            return df
        
        df = df.copy()
        df.set_index('Tanggal', inplace=True)
        
        agg_df = df.groupby('Provinsi').resample(period)['Realisasi'].sum().reset_index()
        logger.info(f"Aggregated data by {period} period. New shape: {agg_df.shape}")
        
        return agg_df
    
    def create_features(self, df):
        """
        Create additional features for modeling
        
        Features:
        - Month of year (seasonality)
        - Quarter
        - Year-over-year growth
        - Moving average
        """
        df = df.copy()
        
        if 'Tanggal' in df.columns:
            df['Bulan'] = df['Tanggal'].dt.month
            df['Kuartal'] = df['Tanggal'].dt.quarter
            df['Tahun'] = df['Tanggal'].dt.year
            df['Hari_Ke_Dalam_Tahun'] = df['Tanggal'].dt.dayofyear
        
        # Calculate year-over-year growth (if multiple years)
        if 'Tahun' in df.columns and 'Tahun' in df.columns:
            df = df.sort_values(['Provinsi', 'Tahun', 'Bulan'])
            df['YoY_Growth'] = df.groupby(['Provinsi', 'Bulan'])['Realisasi'].pct_change(12) * 100
        
        # Moving average
        df['MA_3'] = df.groupby('Provinsi')['Realisasi'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        logger.info(f"Created features. New shape: {df.shape}")
        return df


# Convenience function
def preprocess(df):
    """Quick preprocess function"""
    preprocessor = DataPreprocessor()
    return preprocessor.clean_revenue_data(df)
