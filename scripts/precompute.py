"""
Pre-compute script for RevDadas Next.js frontend.

Runs all ML models (Prophet, Isolation Forest, etc.) and exports results
as JSON files that the Next.js frontend can consume statically.
This eliminates the need for a live Python backend on Vercel.

Usage:
    cd experiment
    python scripts/precompute.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src import data_loader, preprocessing, forecasting, anomaly_detection, business, policy

OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Forecast periods to pre-compute
FORECAST_PERIODS = list(range(6, 25))


def json_serializer(obj):
    """Custom JSON serializer for types not handled by default."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data, filename):
    """Save data as JSON file."""
    path = OUTPUT_DIR / filename
    
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        elif isinstance(obj, float) and pd.isna(obj):
            return None
        return obj

    cleaned_data = clean_nan(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, default=json_serializer, ensure_ascii=False, indent=2)
    print(f"  [OK] Saved {path.name} ({path.stat().st_size / 1024:.1f} KB)")

def load_and_preprocess():
    """Load and preprocess data using existing modules."""
    print("[INFO] Loading data...")
    loader = data_loader.BPSDataLoader()
    df = loader.load_revenue_data()
    if df is None:
        df = loader.create_sample_data()

    preprocessor = preprocessing.DataPreprocessor()
    df = preprocessor.clean_revenue_data(df)
    df = preprocessor.create_features(df)

    print(f"  [OK] Loaded {len(df)} rows, {df['Provinsi'].nunique()} provinces, "
          f"{df['Jenis_Pendapatan'].nunique()} tax types")
    return df


def compute_forecasts(df):
    """Pre-compute forecasts efficiently by training once for 24 months and slicing."""
    print("\n[INFO] Computing forecasts (Training Prophet once for max 24 months)...")
    all_forecasts = {}
    
    max_period = max(FORECAST_PERIODS)
    master_forecaster = forecasting.RevenueForecaster(periods=max_period)
    
    # Train once
    results_24 = master_forecaster.train_and_forecast_all(df, run_backtest=True, backtest_horizon=6)
    
    if results_24 is not None and not results_24.empty:
        acc = master_forecaster.overall_accuracy()
        acc_table = master_forecaster.accuracy_summary()
        
        # Slice for each requested period
        for period in FORECAST_PERIODS:
            sliced = results_24.groupby(["Provinsi", "Jenis_Pendapatan"]).head(period)
            records = sliced.copy()
            records["Tanggal"] = records["Tanggal"].astype(str)
            all_forecasts[str(period)] = records.to_dict(orient="records")
    else:
        acc = None
        acc_table = None
        for period in FORECAST_PERIODS:
            all_forecasts[str(period)] = []

    accuracy_data = {
        "overall": acc if acc else None,
        "by_series": acc_table.to_dict(orient="records") if acc_table is not None else []
    }

    return all_forecasts, accuracy_data


def compute_anomalies(df):
    """Pre-compute anomaly detection (once, independent of forecast period)."""
    # Contamination fixed — anomaly detection menganalisis data historis,
    # tidak bergantung pada forecast period.
    contamination = 0.05
    
    detector = anomaly_detection.AnomalyDetector(contamination=contamination)
    detector.train(df)
    results = detector.detect(df)

    if results is not None:
        records = results.copy()
        records["Tanggal"] = records["Tanggal"].astype(str)
        # Convert boolean columns
        if "Anomaly" in records.columns:
            records["Anomaly"] = records["Anomaly"].astype(bool)
        data = records.to_dict(orient="records")
        return data
    return []


def compute_business_recs(df, forecast_results_df):
    """Pre-compute business sector recommendations."""
    scored = business.score_sectors(df, forecast_results_df)
    top_recs = business.top_recommendations(df, forecast_results_df, top_n=5)

    # Convert to JSON-serializable format
    scored_json = {}
    for prov, sectors in scored.items():
        scored_json[prov] = sectors  # Already list of dicts

    return {
        "scored": scored_json,
        "top_recommendations": top_recs,
    }


def compute_policy_recs(df, forecast_results_df, anomaly_results_df):
    """Pre-compute policy recommendations for various fraud prevention levels."""
    print("\n[INFO] Computing policy recommendations per province...")
    policy_data = {}

    provinces = df["Provinsi"].unique().tolist()
    
    for prov in provinces:
        policy_data[prov] = {}
        prov_df = df[df["Provinsi"] == prov].copy() if df is not None else None
        prov_fc = forecast_results_df[forecast_results_df["Provinsi"] == prov].copy() if forecast_results_df is not None else None
        prov_an = anomaly_results_df[anomaly_results_df["Provinsi"] == prov].copy() if anomaly_results_df is not None else None
        
        for pct in [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
            recs = policy.generate_recommendations(
                prov_df, prov_fc, prov_an,
                fraud_prevention_pct=pct
            )
            for r in recs:
                r["provinsi"] = prov
            policy_data[prov][str(pct)] = recs

    return policy_data


def main():
    print("=" * 60)
    print("RevDadas Pre-compute Script")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # 1. Load data
    df = load_and_preprocess()

    # 2. Export historical data
    print("\n[INFO] Exporting historical data...")
    hist = df.copy()
    hist["Tanggal"] = hist["Tanggal"].astype(str)
    save_json(hist.to_dict(orient="records"), "historical.json")

    # 3. Export metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "provinces": sorted(df["Provinsi"].unique().tolist()),
        "tax_types": sorted(df["Jenis_Pendapatan"].unique().tolist()),
        "forecast_periods": FORECAST_PERIODS,
        "date_range": {
            "min": str(df["Tanggal"].min()),
            "max": str(df["Tanggal"].max()),
        },
        "total_rows": len(df),
    }
    save_json(meta, "meta.json")

    # 4. Compute and export anomalies ONCE (on raw data)
    print("\n[INFO] Computing anomalies (single pass)...")
    anomaly_data = compute_anomalies(df)
    save_json(anomaly_data, "anomalies.json")

    # 5. EDA / Data Cleaning: Handle extreme values for stable forecasting
    print("\n[INFO] EDA: Handling extreme values for stable forecasting...")
    preprocessor = preprocessing.DataPreprocessor()
    df_clean = preprocessor.handle_outliers(df, method='iqr', threshold=3.0)

    # 6. Compute and export forecasts on CLEANED data
    all_forecasts, accuracy_data = compute_forecasts(df_clean)
    save_json(all_forecasts, "forecasts.json")
    save_json(accuracy_data, "accuracy.json")

    # 6. Compute business recommendations for each forecast period
    print("\n[INFO] Computing business recommendations for all periods...")
    all_biz = {}
    for period in FORECAST_PERIODS:
        fc_data = all_forecasts.get(str(period), [])
        if fc_data:
            fc_df = pd.DataFrame(fc_data)
            fc_df["Tanggal"] = pd.to_datetime(fc_df["Tanggal"])
        else:
            fc_df = None
        biz_data = compute_business_recs(df, fc_df)
        all_biz[str(period)] = biz_data
    save_json(all_biz, "business.json")

    # 7. Compute policy recommendations (using default 24-month base)
    fc_df_24 = pd.DataFrame(all_forecasts.get("24", []))
    if not fc_df_24.empty: 
        fc_df_24["Tanggal"] = pd.to_datetime(fc_df_24["Tanggal"])
    else:
        fc_df_24 = None

    # Use single anomaly result (no longer per-period)
    anom_df_24 = pd.DataFrame(anomaly_data) if anomaly_data else None
    if anom_df_24 is not None and not anom_df_24.empty:
        anom_df_24["Tanggal"] = pd.to_datetime(anom_df_24["Tanggal"])
    else:
        anom_df_24 = None

    policy_data = compute_policy_recs(df, fc_df_24, anom_df_24)
    save_json(policy_data, "policy.json")

    print("\n" + "=" * 60)
    print("[OK] Pre-compute complete!")
    print(f"   Output: {OUTPUT_DIR}")
    files = list(OUTPUT_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"   Files: {len(files)}, Total: {total_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
