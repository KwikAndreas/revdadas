"""
Pre-compute script for RevDadas Next.js frontend with Dynamic Dual-Model Architecture.

Mendukung model forecasting ganda yang terisolasi 100%:
- Theta Method (Primary Engine - Default): Cepat, parsimonious, anti-overfitting & anti-jomplang.
- Prophet (Secondary Optional Engine): Tersedia untuk perbandingan / A-B testing.

Kedua model disimpan terpisah di:
  frontend/public/data/models/theta/
  frontend/public/data/models/prophet/

File aktif yang dikonsumsi langsung oleh frontend (tanpa mengubah frontend):
  frontend/public/data/{forecasts.json, accuracy.json, business.json, policy.json}

Penggunaan:
    # 1. Precompute menggunakan Theta (Primary - default):
    python scripts/precompute.py --model theta

    # 2. Precompute menggunakan Prophet (Secondary):
    python scripts/precompute.py --model prophet

    # 3. Precompute kedua model sekaligus (Theta & Prophet):
    python scripts/precompute.py --model all

    # 4. Switch model aktif secara instan (< 0.1 detik dari cache tanpa re-train):
    python scripts/precompute.py --switch prophet
    python scripts/precompute.py --switch theta
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import anomaly_detection, business, data_loader, forecasting, policy, preprocessing

OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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


def save_json(data, target_path):
    """Save data as JSON file to a specific path."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        elif isinstance(obj, float) and pd.isna(obj):
            return None
        return obj

    cleaned_data = clean_nan(data)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, default=json_serializer, ensure_ascii=False, indent=2)
    print(f"  [OK] Saved {target_path.name} ({target_path.stat().st_size / 1024:.1f} KB)")


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


def compute_forecasts_for_model(df, model_type="theta"):
    """Pre-compute forecasts efficiently by training once for 24 months and slicing."""
    model_name = "Theta Method (Primary)" if model_type == "theta" else "Prophet (Secondary)"
    print(f"\n[INFO] Computing forecasts using {model_name}...")
    all_forecasts = {}

    max_period = max(FORECAST_PERIODS)
    master_forecaster = forecasting.RevenueForecaster(periods=max_period, model_type=model_type)

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
        "model": model_type,
        "model_title": model_name,
        "overall": acc if acc else None,
        "by_series": acc_table.to_dict(orient="records") if acc_table is not None else []
    }

    return all_forecasts, accuracy_data, results_24


def compute_anomalies(df):
    """Pre-compute anomaly detection (once, independent of forecast model)."""
    contamination = 0.05
    detector = anomaly_detection.AnomalyDetector(contamination=contamination)
    detector.train(df)
    results = detector.detect(df)

    if results is not None:
        records = results.copy()
        records["Tanggal"] = records["Tanggal"].astype(str)
        if "Anomaly" in records.columns:
            records["Anomaly"] = records["Anomaly"].astype(bool)
        return records.to_dict(orient="records")
    return []


def compute_business_recs(df, forecast_results_df):
    """Pre-compute business sector recommendations."""
    scored = business.score_sectors(df, forecast_results_df)
    top_recs = business.top_recommendations(df, forecast_results_df, top_n=5)

    scored_json = {}
    for prov, sectors in scored.items():
        scored_json[prov] = sectors

    return {
        "scored": scored_json,
        "top_recommendations": top_recs,
    }


def compute_policy_recs(df, forecast_results_df, anomaly_results_df):
    """Pre-compute policy recommendations for various fraud prevention levels."""
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


def precompute_model_package(df, df_clean, anomaly_data, model_type="theta"):
    """
    Eksekusi paket forecasting lengkap untuk satu model secara terisolasi.
    Output disimpan ke folder terisolasi: frontend/public/data/models/{model_type}/
    """
    model_dir = MODELS_DIR / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    model_title = "Theta Method (Primary)" if model_type == "theta" else "Prophet (Secondary)"

    print(f"\n>>> [START] Memproses Paket Model: {model_title}")

    # 1. Forecasts & Accuracy
    all_forecasts, accuracy_data, results_24 = compute_forecasts_for_model(df_clean, model_type=model_type)
    save_json(all_forecasts, model_dir / "forecasts.json")
    save_json(accuracy_data, model_dir / "accuracy.json")

    # 2. Business Recommendations (per period)
    print(f"[INFO] Computing business recommendations for {model_title}...")
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
    save_json(all_biz, model_dir / "business.json")

    # 3. Policy Recommendations (24-month base)
    print(f"[INFO] Computing policy recommendations for {model_title}...")
    fc_df_24 = pd.DataFrame(all_forecasts.get("24", []))
    if not fc_df_24.empty:
        fc_df_24["Tanggal"] = pd.to_datetime(fc_df_24["Tanggal"])
    else:
        fc_df_24 = None

    anom_df = pd.DataFrame(anomaly_data) if anomaly_data else None
    if anom_df is not None and not anom_df.empty:
        anom_df["Tanggal"] = pd.to_datetime(anom_df["Tanggal"])
    else:
        anom_df = None

    policy_data = compute_policy_recs(df, fc_df_24, anom_df)
    save_json(policy_data, model_dir / "policy.json")

    print(f">>> [SUCCESS] Paket Model {model_title} tersimpan aman di: {model_dir.name}/")


def activate_model(model_type):
    """
    Mengaktifkan model ke root frontend/public/data/ secara atomik.
    Frontend akan langsung membaca model ini tanpa modifikasi kode apapun.
    """
    model_dir = MODELS_DIR / model_type
    if not (model_dir / "forecasts.json").exists():
        print(f"[ERROR] Model '{model_type}' belum di-precompute! Jalankan: python scripts/precompute.py --model {model_type}")
        return False

    model_title = "Theta Method (Primary)" if model_type == "theta" else "Prophet (Secondary)"
    print(f"\n[INFO] Mengaktifkan model '{model_title}' ke antarmuka frontend...")

    # Salin 4 file utama ke root
    for fname in ["forecasts.json", "accuracy.json", "business.json", "policy.json"]:
        src = model_dir / fname
        dst = OUTPUT_DIR / fname
        shutil.copy2(src, dst)
        print(f"  [ACTIVE] {fname} -> {dst.name}")

    # Perbarui meta.json aktif
    meta_path = OUTPUT_DIR / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    meta["active_model"] = model_type
    meta["active_model_name"] = model_title
    meta["available_models"] = ["theta", "prophet"]
    meta["last_switched"] = datetime.now().isoformat()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, default=json_serializer, ensure_ascii=False, indent=2)

    print(f"[OK] Model '{model_title}' sekarang AKTIF pada Frontend!")
    return True


def main():
    parser = argparse.ArgumentParser(description="RevDadas Pre-compute Script (Dual-Model Architecture)")
    parser.add_argument(
        "--model", choices=["theta", "prophet", "all"], default="theta",
        help="Model forecasting to precompute: 'theta' (Primary default), 'prophet' (Secondary), or 'all'"
    )
    parser.add_argument(
        "--switch", choices=["theta", "prophet"], default=None,
        help="Instantly switch active model without re-training (loads from cached models/ directory)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("RevDadas Pre-compute & Model Management Script")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 65)

    # 1. Mode Instant Switch (jika parameter --switch digunakan)
    if args.switch:
        success = activate_model(args.switch)
        if success:
            print("\n[OK] Switch selesai tanpa re-training.")
            return
        else:
            print(f"[INFO] Mencoba komputasi otomatis untuk model '{args.switch}'...")
            args.model = args.switch

    # 2. Pipeline Umum (Load, Preprocess, Anomaly Detection, Base Meta)
    df = load_and_preprocess()

    print("\n[INFO] Exporting historical data...")
    hist = df.copy()
    hist["Tanggal"] = hist["Tanggal"].astype(str)
    save_json(hist.to_dict(orient="records"), OUTPUT_DIR / "historical.json")

    print("\n[INFO] Computing anomalies (single pass on raw data)...")
    anomaly_data = compute_anomalies(df)
    save_json(anomaly_data, OUTPUT_DIR / "anomalies.json")

    # Clean extreme values for stable forecasting
    print("\n[INFO] EDA: Handling extreme values for stable forecasting...")
    preprocessor = preprocessing.DataPreprocessor()
    df_clean = preprocessor.handle_outliers(df, method="iqr", threshold=3.0)

    # Base metadata
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
        "available_models": ["theta", "prophet"],
    }
    save_json(meta, OUTPUT_DIR / "meta.json")

    # 3. Eksekusi Model Sesuai Pilihan
    if args.model == "all":
        # Precompute Theta (Primary)
        precompute_model_package(df, df_clean, anomaly_data, model_type="theta")
        # Precompute Prophet (Secondary)
        precompute_model_package(df, df_clean, anomaly_data, model_type="prophet")
        # Aktifkan Theta secara default
        activate_model("theta")

    elif args.model == "theta":
        precompute_model_package(df, df_clean, anomaly_data, model_type="theta")
        activate_model("theta")

    elif args.model == "prophet":
        precompute_model_package(df, df_clean, anomaly_data, model_type="prophet")
        activate_model("prophet")

    print("\n" + "=" * 65)
    print("[COMPLETE] Semua data pre-compute berhasil diproses!")
    print(f"Lokasi Data: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
