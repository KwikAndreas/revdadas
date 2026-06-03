"""
Rekomendasi kebijakan otomatis untuk RevDadas.

Menghasilkan rekomendasi berbasis-data dari hasil forecast (pertumbuhan
proyeksi pendapatan) dan deteksi anomali (potensi kebocoran/fraud).
Sifatnya indikatif sebagai bahan diskusi kebijakan, bukan keputusan final.
"""

import pandas as pd
import numpy as np


def _fmt(v):
    v = float(v)
    if abs(v) >= 1e12:
        return f"Rp {v/1e12:.1f} T"
    if abs(v) >= 1e9:
        return f"Rp {v/1e9:.1f} M"
    return f"Rp {v:,.0f}"


def generate_recommendations(filtered_df, forecast_results, anomaly_results,
                             fraud_prevention_pct=5):
    """
    Kembalikan list dict rekomendasi:
        {judul, prioritas, detail}
    prioritas: 'Tinggi' | 'Sedang' | 'Rendah'
    """
    recs = []

    total_real = float(filtered_df["Realisasi"].sum()) if filtered_df is not None and len(filtered_df) else 0.0

    # --- 1. Pertumbuhan proyeksi per provinsi (dari forecast vs aktual terakhir) ---
    if forecast_results is not None and len(forecast_results) and filtered_df is not None:
        for prov in sorted(filtered_df["Provinsi"].unique()):
            akt = filtered_df[filtered_df["Provinsi"] == prov]
            fc = forecast_results[forecast_results["Provinsi"] == prov]
            if not len(akt) or not len(fc):
                continue
            # rata-rata realisasi bulanan aktual 12 bln terakhir
            akt_bulanan = akt.groupby("Tanggal")["Realisasi"].sum().sort_index()
            base = akt_bulanan.tail(12).mean()
            proj = fc.groupby("Tanggal")["Prediksi"].sum().mean()
            if base and base > 0:
                growth = (proj - base) / base * 100
                if growth < -2:
                    recs.append({
                        "judul": f"Waspada penurunan pendapatan di {prov}",
                        "prioritas": "Tinggi",
                        "detail": (f"Proyeksi rata-rata bulanan ({_fmt(proj)}) lebih rendah "
                                   f"{abs(growth):.1f}% dibanding rata-rata 12 bulan terakhir "
                                   f"({_fmt(base)}). Pertimbangkan intensifikasi pajak daerah "
                                   f"dan evaluasi target penerimaan."),
                    })
                elif growth > 8:
                    recs.append({
                        "judul": f"Momentum pertumbuhan di {prov}",
                        "prioritas": "Rendah",
                        "detail": (f"Proyeksi tumbuh {growth:.1f}% dibanding rata-rata terakhir. "
                                   f"Manfaatkan untuk memperkuat dana cadangan dan belanja "
                                   f"modal produktif."),
                    })

    # --- 2. Anomali / potensi kebocoran ---
    if anomaly_results is not None and "Anomaly" in anomaly_results.columns:
        anomalies = anomaly_results[anomaly_results["Anomaly"] == True]
        n_anom = len(anomalies)
        if n_anom > 0:
            loss = float(anomalies["Realisasi"].sum())
            potensi_selamat = loss * (fraud_prevention_pct / 100.0)
            # provinsi & jenis dengan anomali terbanyak
            top_prov = anomalies["Provinsi"].value_counts().idxmax()
            top_jenis = anomalies["Jenis_Pendapatan"].value_counts().idxmax()
            recs.append({
                "judul": "Audit pos pendapatan beranomali",
                "prioritas": "Tinggi",
                "detail": (f"Terdeteksi {n_anom} catatan anomali (nominal {_fmt(loss)}). "
                           f"Konsentrasi terbesar di {top_prov}, terutama pos "
                           f"\"{top_jenis}\". Audit terfokus dapat menyelamatkan ~"
                           f"{_fmt(potensi_selamat)} pada tingkat pencegahan "
                           f"{fraud_prevention_pct}%."),
            })

    # --- 3. Diversifikasi sumber pendapatan ---
    if filtered_df is not None and len(filtered_df):
        prop = filtered_df.groupby("Jenis_Pendapatan")["Realisasi"].sum()
        if prop.sum() > 0:
            share = (prop / prop.sum() * 100).sort_values(ascending=False)
            top_name = share.index[0]
            top_share = share.iloc[0]
            if top_share > 50:
                recs.append({
                    "judul": "Kurangi ketergantungan pada satu sumber",
                    "prioritas": "Sedang",
                    "detail": (f"Pos \"{top_name}\" menyumbang {top_share:.0f}% dari total "
                               f"pendapatan terpilih. Ketergantungan tinggi menambah risiko "
                               f"fiskal; perkuat diversifikasi (mis. retribusi & lain-lain PAD)."),
                })

    # --- 4. Fallback ---
    if not recs:
        recs.append({
            "judul": "Kondisi fiskal relatif stabil",
            "prioritas": "Rendah",
            "detail": ("Tidak ada sinyal risiko signifikan dari proyeksi maupun anomali pada "
                       "data terpilih. Pertahankan disiplin anggaran dan pemantauan rutin."),
        })

    # urutkan: Tinggi -> Sedang -> Rendah
    order = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}
    recs.sort(key=lambda r: order.get(r["prioritas"], 9))
    return recs
