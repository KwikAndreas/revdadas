"""
Forecasting module using Prophet for time-series prediction
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from prophet import Prophet
from . import utils

logger = logging.getLogger(__name__)


class RevenueForecaster:
    """
    Forecast revenue using Facebook Prophet
    
    Prophet is chosen for:
    - Robust handling of seasonality
    - Built-in holiday effects
    - Easy to tune with interpretable parameters
    - Fast training
    """
    
    def __init__(self, periods=12, interval_width=0.95):
        """
        Initialize forecaster
        
        Args:
            periods: Number of periods to forecast (default: 12 months)
            interval_width: Confidence interval width (default: 0.95 = 95%)
        """
        self.periods = periods
        self.interval_width = interval_width
        self.models = {}  # Store models for each province-pajak combination
        self.forecasts = {}
        
    def prepare_data(self, df, provinsi, jenis_pajak):
        """
        Prepare data in Prophet format
        
        Expected format:
        - ds: date column (Tanggal)
        - y: value column (Realisasi)
        """
        # Filter data
        mask = (df['Provinsi'] == provinsi) & (df['Jenis_Pendapatan'] == jenis_pajak)
        data = df[mask][['Tanggal', 'Realisasi']].copy()
        
        # Rename for Prophet
        data.columns = ['ds', 'y']
        data = data.sort_values('ds').reset_index(drop=True)
        
        if len(data) < 12:
            logger.warning(f"Only {len(data)} data points for {provinsi}-{jenis_pajak}. Min 12 required.")
            return None
        
        logger.info(f"Prepared {len(data)} data points for {provinsi}-{jenis_pajak}")
        return data
    
    def train(self, df, provinsi, jenis_pajak):
        """
        Train Prophet model for specific province-pajak combination
        """
        data = self.prepare_data(df, provinsi, jenis_pajak)
        
        if data is None:
            return None
        
        try:
            # Initialize Prophet with optimized parameters
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=self.interval_width,
                changepoint_prior_scale=0.05,
            )
            
            # Suppress Prophet's verbose logging
            with utils.logging.getLogger("prophet").propagate(False):
                model.fit(data)
            
            key = f"{provinsi}_{jenis_pajak}"
            self.models[key] = model
            logger.info(f"Model trained for {key}")
            
            return model
        
        except Exception as e:
            logger.error(f"Error training model for {provinsi}-{jenis_pajak}: {e}")
            return None
    
    def forecast(self, provinsi, jenis_pajak):
        """
        Generate forecast for specific province-pajak combination
        """
        key = f"{provinsi}_{jenis_pajak}"
        
        if key not in self.models:
            logger.warning(f"Model not found for {key}. Train first.")
            return None
        
        model = self.models[key]
        
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=self.periods, freq='MS')
            forecast = model.predict(future)
            
            # Keep only future periods
            forecast = forecast[forecast['ds'] > forecast['ds'].iloc[-self.periods-1]]
            
            # Select relevant columns
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            result.columns = ['Tanggal', 'Prediksi', 'Batas_Bawah', 'Batas_Atas']
            result['Provinsi'] = provinsi
            result['Jenis_Pendapatan'] = jenis_pajak
            
            self.forecasts[key] = result
            logger.info(f"Forecast generated for {key}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating forecast for {key}: {e}")
            return None
    
    def train_and_forecast_all(self, df):
        """
        Train models and generate forecasts for all province-pajak combinations
        """
        provinsi_list = df['Provinsi'].unique()
        jenis_pajak_list = df['Jenis_Pendapatan'].unique()
        
        all_forecasts = []
        
        for provinsi in provinsi_list:
            for pajak in jenis_pajak_list:
                # Train
                self.train(df, provinsi, pajak)
                
                # Forecast
                forecast = self.forecast(provinsi, pajak)
                if forecast is not None:
                    all_forecasts.append(forecast)
        
        if all_forecasts:
            combined = pd.concat(all_forecasts, ignore_index=True)
            logger.info(f"Generated {len(combined)} forecast rows")
            return combined
        
        return None
    
    def save_models(self, path=None):
        """Save all trained models"""
        import pickle
        
        path = path or utils.get_models_path()
        
        for key, model in self.models.items():
            filepath = Path(path) / f"model_{key}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {filepath}")
    
    def load_models(self, path=None):
        """Load saved models"""
        import pickle
        
        path = path or utils.get_models_path()
        path = Path(path)
        
        for file in path.glob("model_*.pkl"):
            with open(file, 'rb') as f:
                model = pickle.load(f)
                key = file.stem.replace("model_", "")
                self.models[key] = model
            logger.info(f"Model loaded from {file}")


# Convenience function
def forecast_revenue(df, provinsi, jenis_pajak, periods=12):
    """Quick forecast function"""
    forecaster = RevenueForecaster(periods=periods)
    forecaster.train(df, provinsi, jenis_pajak)
    return forecaster.forecast(provinsi, jenis_pajak)
