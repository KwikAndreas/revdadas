"""
Data preprocessing and cleaning for RevDadas
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data cleaning and transformation"""
    
    def __init__(self):
        self.processed_data = None
        
    def clean_revenue_data(self, df):
        """
        Clean and prepare revenue data
        
        Steps:
        1. Handle missing values
        2. Detect and handle outliers
        3. Ensure data types
        4. Sort by date
        """
        df = df.copy()
        logger.info("Starting data cleaning...")
        
        # 1. Handle missing values
        logger.info(f"Missing values before: {df.isnull().sum().sum()}")
        df = df.fillna(method='ffill').fillna(method='bfill')
        logger.info(f"Missing values after: {df.isnull().sum().sum()}")
        
        # 2. Ensure correct data types
        if 'Tahun' in df.columns:
            df['Tahun'] = df['Tahun'].astype(int)
        if 'Bulan' in df.columns:
            df['Bulan'] = df['Bulan'].astype(int)
        if 'Realisasi' in df.columns:
            df['Realisasi'] = pd.to_numeric(df['Realisasi'], errors='coerce')
        
        # 3. Create date column
        df['Tanggal'] = pd.to_datetime(
            df['Tahun'].astype(str) + '-' + df['Bulan'].astype(str).str.zfill(2) + '-01'
        )
        
        # 4. Remove rows with invalid data
        df = df.dropna(subset=['Realisasi'])
        df = df[df['Realisasi'] > 0]
        
        # 5. Sort by date
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. Shape: {df.shape}")
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
