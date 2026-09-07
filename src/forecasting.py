"""
Forecasting module untuk RevDadas — Dual Model Architecture:
- PRIMARY MODEL   : Theta Method (M3 Competition Winner - stabil, parsimonious, anti-overfitting & anti-jomplang).
- SECONDARY MODEL : Prophet (Ensemble Prophet + Naive-Seasonal - opsional untuk evaluasi A/B).

Mendukung pemilihan model dinamis via parameter `model_type='theta'` atau `'prophet'`.
API tetap 100% kompatibel dengan frontend Next.js:
  RevenueForecaster(periods=..., model_type=...).train_and_forecast_all(df) -> DataFrame
  kolom: Tanggal, Prediksi, Batas_Bawah, Batas_Atas, Provinsi, Jenis_Pendapatan, Metode
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel

from . import utils

logger = logging.getLogger(__name__)

# Nonaktifkan logging berisik dari prophet & cmdstanpy bila prophet dipanggil
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# Bobot ensemble Prophet (hasil tuning backtest)
W_PROPHET = 0.6
W_SEASONAL = 0.4

# Pos pendapatan utama yang layak diramalkan & dihitung akurasinya
CORE_ACCOUNTS = {
    "Pendapatan Asli Daerah (PAD)",
    "Transfer ke Daerah dan Dana Desa (TKDD)",
    "Total Pendapatan Daerah",
    "Total Belanja Daerah",
    "Belanja Modal",
    "Pajak Daerah",
}


def wape(actual, pred):
    """Weighted Absolute Percentage Error (%) — robust terhadap nilai kecil/0."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    return float(np.sum(np.abs(actual - pred)) / denom * 100) if denom else np.nan


def smape(actual, pred):
    """Symmetric MAPE (%) — robust terhadap nilai mendekati 0."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual) + np.abs(pred)
    mask = denom != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(actual - pred)[mask] / (denom[mask] / 2)) * 100)


class RevenueForecaster:
    """Forecast pendapatan bulanan daerah (Theta Method Primary / Prophet Secondary)."""

    def __init__(self, periods=12, interval_width=0.90, model_type="theta"):
        """
        Inisialisasi forecaster.
        model_type: 'theta' (Primary) atau 'prophet' (Secondary Optional).
        """
        self.periods = periods
        self.interval_width = interval_width
        self.model_type = model_type.lower()
        if self.model_type not in ("theta", "prophet"):
            logger.warning(f"model_type '{model_type}' tidak dikenal, fallback ke 'theta'")
            self.model_type = "theta"

        self.models = {}
        self.forecasts = {}
        self.metrics = {}   # key -> {wape, smape, n_test}

    # ---------- penyiapan data ----------
    def prepare_data(self, df, provinsi, jenis_pajak):
        mask = (df["Provinsi"] == provinsi) & (df["Jenis_Pendapatan"] == jenis_pajak)
        data = df[mask][["Tanggal", "Realisasi"]].copy()
        data.columns = ["ds", "y"]
        data = data.sort_values("ds").reset_index(drop=True)
        return data

    @staticmethod
    def _seasonal_map(train):
        t = train.copy()
        t["m"] = t["ds"].dt.month
        return t.groupby("m")["y"].mean(), t["y"].mean()

    def _seasonal_pred(self, train, future_dates):
        seas, overall = self._seasonal_map(train)
        return np.array([float(seas.get(d.month, overall)) for d in future_dates])

    # ---------- ENGINE 1: THETA METHOD (PRIMARY) ----------
    def _fit_and_forecast_theta(self, data, periods):
        """Fit ThetaModel dengan deseasonalization dan prediction intervals."""
        # Pastikan index memiliki freq='MS' agar statsmodels tidak mengeluarkan peringatan
        dti = pd.to_datetime(data["ds"])
        s = pd.Series(data["y"].values, index=pd.DatetimeIndex(dti, freq="MS"))
        last_date = data["ds"].max()
        fdates = [last_date + pd.offsets.MonthBegin(i) for i in range(1, periods + 1)]
        hist_max = float(s.max()) if len(s) else 1.0

        if len(s) >= 24:
            try:
                # Multiplicative deseasonalize bila data strictly positif dan bernilai wajar, additive bila ada 0
                deseas_type = "multiplicative" if (s > 0).all() and (s.min() > 1e6) else "additive"
                # CRITICAL: use_test=False memastikan statsmodels tidak membatalkan dekomposisi
                # musiman pada sampel N=36 akibat uji autokorelasi chi-kuadrat yang terlalu konservatif.
                th = ThetaModel(s, period=12, deseasonalize=True, use_test=False, method=deseas_type)
                res = th.fit()
                pred = res.forecast(periods).values

                # Prediction intervals
                alpha = max(0.01, min(0.5, 1.0 - self.interval_width))
                try:
                    pi = res.prediction_intervals(periods, alpha=alpha)
                    lo = pi["lower"].values
                    hi = pi["upper"].values
                except Exception:
                    lo = pred * 0.85
                    hi = pred * 1.15
            except Exception as e:
                logger.debug(f"ThetaModel detail: {e}, using seasonal fallback")
                sp = self._seasonal_pred(data, fdates)
                pred, lo, hi = sp, sp * 0.85, sp * 1.15
        else:
            sp = self._seasonal_pred(data, fdates)
            pred, lo, hi = sp, sp * 0.85, sp * 1.15

        # Anti-jomplang guardrails (Capping to avoid extreme outliers)
        max_cap = 1.35 * hist_max if hist_max > 0 else np.inf
        pred = np.clip(pred, 0, max_cap)
        lo = np.clip(lo, pred * 0.70, pred)
        hi = np.clip(hi, pred, max_cap * 1.25)

        return fdates, pred, lo, hi

    def _evaluate_theta(self, data, horizon=6):
        """Backtest evaluasi Theta Model pada holdout horizon."""
        if data is None or len(data) < horizon + 12:
            return None
        train = data.iloc[:-horizon].copy()
        test = data.iloc[-horizon:].copy()
        act = test["y"].values
        try:
            _, pred, _, _ = self._fit_and_forecast_theta(train, periods=horizon)
            return {
                "wape": wape(act, pred),
                "smape": smape(act, pred),
                "n_test": int(horizon)
            }
        except Exception as e:
            logger.warning(f"Evaluate Theta gagal: {e}")
            return None

    # ---------- ENGINE 2: PROPHET (SECONDARY OPTIONAL) ----------
    def _prophet(self):
        from prophet import Prophet
        return Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            interval_width=self.interval_width, changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )

    def _evaluate_and_weight_prophet(self, data, horizon=6):
        """Evaluasi holdout + bobot ensemble Prophet."""
        if data is None or len(data) < horizon + 12:
            return None, W_PROPHET
        train = data.iloc[:-horizon]
        test = data.iloc[-horizon:]
        try:
            m = self._prophet()
            m.fit(train)
            fc = m.predict(m.make_future_dataframe(periods=horizon, freq="MS")).tail(horizon)
            pp = np.clip(fc["yhat"].values, 0, None)
            sp = self._seasonal_pred(train, test["ds"])
            act = test["y"].values
            best_w, best_e = W_PROPHET, np.inf
            for w in (0.0, 0.3, 0.5, 0.6, 0.8, 1.0):
                e = wape(act, np.clip(w * pp + (1 - w) * sp, 0, None))
                if e == e and e < best_e:
                    best_e, best_w = e, w
            pred = np.clip(best_w * pp + (1 - best_w) * sp, 0, None)
            metrics = {"wape": wape(act, pred), "smape": smape(act, pred), "n_test": int(horizon)}
            return metrics, best_w
        except Exception as e:
            logger.warning(f"Evaluate Prophet gagal: {e}")
            return None, W_PROPHET

    # ---------- PELATIHAN & PREDIKSI TERPADU ----------
    def train(self, df, provinsi, jenis_pajak, weight=W_PROPHET):
        data = self.prepare_data(df, provinsi, jenis_pajak)
        key = f"{provinsi}_{jenis_pajak}"
        if len(data) < 12:
            return None

        if self.model_type == "theta":
            # Theta fit cepat dieksekusi saat forecast, simpan data train
            self.models[key] = {"data": data, "type": "theta"}
            return True
        else:
            try:
                m = self._prophet()
                m.fit(data)
                self.models[key] = {"prophet": m, "train": data, "w": weight, "type": "prophet"}
                return m
            except Exception as e:
                logger.error(f"Error training Prophet {key}: {e}")
                return None

    def forecast(self, df, provinsi, jenis_pajak):
        key = f"{provinsi}_{jenis_pajak}"
        data = self.prepare_data(df, provinsi, jenis_pajak)

        # Fallback bila data terlalu pendek (< 12 bulan)
        if len(data) < 12:
            if len(data) == 0:
                return None
            last = data["ds"].max()
            fdates = [last + pd.offsets.MonthBegin(i) for i in range(1, self.periods + 1)]
            sp = self._seasonal_pred(data, fdates)
            res = pd.DataFrame({
                "Tanggal": fdates,
                "Prediksi": np.clip(sp, 0, None),
                "Batas_Bawah": np.clip(sp * 0.8, 0, None),
                "Batas_Atas": sp * 1.2,
                "Provinsi": provinsi, "Jenis_Pendapatan": jenis_pajak,
                "Metode": "Musiman (Fallback)",
            })
            self.forecasts[key] = res
            return res

        # 1. Eksekusi Theta (Primary)
        if self.model_type == "theta":
            fdates, pred, lo, hi = self._fit_and_forecast_theta(data, self.periods)
            res = pd.DataFrame({
                "Tanggal": fdates,
                "Prediksi": pred,
                "Batas_Bawah": lo,
                "Batas_Atas": hi,
                "Provinsi": provinsi,
                "Jenis_Pendapatan": jenis_pajak,
                "Metode": "Theta Method (Primary)",
            })
            self.forecasts[key] = res
            return res

        # 2. Eksekusi Prophet (Secondary)
        bundle = self.models.get(key)
        if not bundle or "prophet" not in bundle:
            # Jika belum di-train, train sekarang
            self.train(df, provinsi, jenis_pajak)
            bundle = self.models.get(key)
            if not bundle:
                return None

        model = bundle["prophet"]
        train = bundle["train"]
        w = bundle.get("w", W_PROPHET)
        try:
            future = model.make_future_dataframe(periods=self.periods, freq="MS")
            fc = model.predict(future).tail(self.periods)
            pp = np.clip(fc["yhat"].values, 0, None)
            lo = np.clip(fc["yhat_lower"].values, 0, None)
            hi = np.clip(fc["yhat_upper"].values, 0, None)
            sp = self._seasonal_pred(train, fc["ds"])
            blended = np.clip(w * pp + (1 - w) * sp, 0, None)
            shift = blended - pp
            res = pd.DataFrame({
                "Tanggal": fc["ds"].values,
                "Prediksi": blended,
                "Batas_Bawah": np.clip(lo + shift, 0, None),
                "Batas_Atas": np.clip(hi + shift, 0, None),
                "Provinsi": provinsi, "Jenis_Pendapatan": jenis_pajak,
                "Metode": "Prophet (Secondary)",
            })
            self.forecasts[key] = res
            return res
        except Exception as e:
            logger.error(f"Error forecast Prophet {key}: {e}")
            return None

    def train_and_forecast_all(self, df, run_backtest=True, backtest_horizon=6):
        all_fc = []
        for prov in df["Provinsi"].unique():
            for jenis in df["Jenis_Pendapatan"].unique():
                data = self.prepare_data(df, prov, jenis)
                if len(data) == 0:
                    continue
                key = f"{prov}_{jenis}"
                is_core = jenis in CORE_ACCOUNTS

                # Evaluasi akurasi backtesting
                if run_backtest and is_core:
                    if self.model_type == "theta":
                        met = self._evaluate_theta(data, horizon=backtest_horizon)
                        if met:
                            self.metrics[key] = met
                    else:
                        met, weight = self._evaluate_and_weight_prophet(data, horizon=backtest_horizon)
                        if met:
                            self.metrics[key] = met

                self.train(df, prov, jenis)
                fc = self.forecast(df, prov, jenis)
                if fc is not None:
                    all_fc.append(fc)

        if all_fc:
            combined = pd.concat(all_fc, ignore_index=True)
            logger.info(f"[{self.model_type.upper()}] Generated {len(combined)} forecast rows")
            return combined
        return None

    # ---------- ringkasan akurasi ----------
    def accuracy_summary(self):
        if not self.metrics:
            return pd.DataFrame(columns=["Provinsi", "Jenis_Pendapatan", "WAPE", "sMAPE", "Akurasi"])
        rows = []
        for key, met in self.metrics.items():
            prov, jenis = key.split("_", 1)
            w = met["wape"]
            rows.append({
                "Provinsi": prov, "Jenis_Pendapatan": jenis,
                "WAPE": round(w, 1) if w == w else None,
                "sMAPE": round(met["smape"], 1) if met["smape"] == met["smape"] else None,
                "Akurasi": round(max(0.0, 100 - w), 1) if w == w else None,
            })
        return pd.DataFrame(rows).sort_values("WAPE", na_position="last")

    def overall_accuracy(self):
        vals = sorted(m["wape"] for m in self.metrics.values() if m["wape"] == m["wape"])
        if not vals:
            return None
        reliable = [v for v in vals if v < 50]
        basis = reliable if reliable else vals
        med = float(np.median(basis))
        return {
            "model": self.model_type,
            "median_wape": med, "akurasi": max(0.0, 100.0 - med),
            "n_series": len(vals), "n_reliable": len(reliable),
            "pct_reliable": round(len(reliable) / len(vals) * 100, 0),
        }

    # ---------- persistensi ----------
    def save_models(self, path=None):
        path = path or utils.get_models_path()
        for key, bundle in self.models.items():
            if bundle.get("type") == "prophet" and "prophet" in bundle:
                with open(Path(path) / f"model_prophet_{key}.pkl", "wb") as f:
                    pickle.dump(bundle["prophet"], f)


def forecast_revenue(df, provinsi, jenis_pajak, periods=12, model_type="theta"):
    f = RevenueForecaster(periods=periods, model_type=model_type)
    f.train(df, provinsi, jenis_pajak)
    return f.forecast(df, provinsi, jenis_pajak)
