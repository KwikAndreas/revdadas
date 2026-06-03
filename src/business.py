"""
Rekomendasi Sektor Bisnis berbasis indikator fiskal daerah (RevDadas).

Gagasan: kondisi & arah kas daerah adalah PROKSI iklim usaha. Struktur
pendapatan (porsi Pajak Daerah vs ketergantungan Transfer Pusat), ukuran
ekonomi, dan tren proyeksi dipakai untuk menilai kelayakan beberapa kategori
bisnis di tiap provinsi.

PENTING (kejujuran metodologis):
- Ini indikator MAKRO tingkat provinsi, bukan studi kelayakan usaha.
- Data APBD hanya sampai level agregat (mis. "Pajak Daerah"), bukan rincian
  pajak hotel/restoran. Karena itu skor sektor diturunkan dari sinyal struktural
  yang relevan, dengan bobot yang dijelaskan terbuka — bukan mengarang korelasi.
- Hasil bersifat indikatif sebagai bahan diskusi, bukan keputusan final.
"""

import numpy as np
import pandas as pd

# Akun pendapatan yang dipakai sebagai sinyal
PAJAK = "Pajak Daerah"
RETRIBUSI = "Retribusi Daerah"
PAD_LAIN = "Lain-Lain PAD yang Sah"
TRANSFER = "Pendapatan Transfer Pemerintah Pusat"

# Definisi sektor + bobot sensitivitas terhadap tiap sinyal (0..1).
# pajak_share : porsi Pajak Daerah (proksi aktivitas konsumsi/usaha lokal:
#               pajak hotel, restoran, hiburan, reklame berada di sini)
# kemandirian : 1 - porsi Transfer Pusat (proksi kekuatan ekonomi mandiri)
# skala       : ukuran ekonomi daerah (pendapatan absolut)
# tren        : arah proyeksi (pertumbuhan ke depan)
SECTORS = {
    "Food & Beverage": {
        "icon": "🍽️",
        "w": {"pajak_share": 0.40, "kemandirian": 0.20, "skala": 0.20, "tren": 0.20},
        "narasi": "Sangat dipengaruhi konsumsi & pariwisata lokal (pajak restoran, hotel, hiburan).",
    },
    "Ritel & Konsumer": {
        "icon": "🛍️",
        "w": {"pajak_share": 0.30, "kemandirian": 0.25, "skala": 0.30, "tren": 0.15},
        "narasi": "Mengikuti daya beli & kepadatan ekonomi; butuh pasar besar.",
    },
    "Pariwisata & Hospitality": {
        "icon": "🏨",
        "w": {"pajak_share": 0.45, "kemandirian": 0.15, "skala": 0.15, "tren": 0.25},
        "narasi": "Bergantung pajak hotel/hiburan & arus pengunjung.",
    },
    "Properti & Konstruksi": {
        "icon": "🏗️",
        "w": {"pajak_share": 0.20, "kemandirian": 0.30, "skala": 0.35, "tren": 0.15},
        "narasi": "Padat modal; perlu skala ekonomi & fiskal yang stabil.",
    },
    "Jasa & Ekonomi Digital": {
        "icon": "💼",
        "w": {"pajak_share": 0.25, "kemandirian": 0.30, "skala": 0.25, "tren": 0.20},
        "narasi": "Tumbuh di daerah dengan ekonomi mandiri & berkembang.",
    },
}


def _safe_div(a, b):
    return a / b if b else 0.0


def compute_signals(filtered_df, forecast_results=None):
    """Hitung sinyal fiskal ternormalisasi (0..1) per provinsi."""
    rows = []
    provinces = sorted(filtered_df["Provinsi"].unique())
    for prov in provinces:
        p = filtered_df[filtered_df["Provinsi"] == prov]
        total = float(p["Realisasi"].sum())
        pajak = float(p[p["Jenis_Pendapatan"] == PAJAK]["Realisasi"].sum())
        transfer = float(p[p["Jenis_Pendapatan"] == TRANSFER]["Realisasi"].sum())

        pajak_share = _safe_div(pajak, total)            # 0..~0.6
        kemandirian = 1.0 - _safe_div(transfer, total)   # 0..1

        # Tren: dari proyeksi vs rata-rata realisasi bulanan terakhir
        tren = 0.0
        if forecast_results is not None and len(forecast_results):
            fc = forecast_results[forecast_results["Provinsi"] == prov]
            if len(fc):
                proj = fc.groupby("Tanggal")["Prediksi"].sum().mean()
                base = p.groupby("Tanggal")["Realisasi"].sum().tail(12).mean()
                if base > 0:
                    tren = (proj - base) / base   # bisa negatif
        rows.append({"Provinsi": prov, "total": total,
                     "pajak_share_raw": pajak_share, "kemandirian_raw": kemandirian,
                     "tren_raw": tren})

    s = pd.DataFrame(rows)
    if s.empty:
        return s

    # Normalisasi min-max ke 0..1 (skala relatif antar provinsi terpilih)
    def norm(col):
        v = s[col]
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else pd.Series([0.6] * len(v), index=v.index)

    s["pajak_share"] = norm("pajak_share_raw")
    s["kemandirian"] = norm("kemandirian_raw")
    s["skala"] = norm("total")
    # tren: petakan ke 0..1 dengan 0% -> 0.5 (clip -20%..+20%)
    s["tren"] = ((s["tren_raw"].clip(-0.2, 0.2) + 0.2) / 0.4)
    return s


def score_sectors(filtered_df, forecast_results=None):
    """Kembalikan dict: {provinsi: [ {sektor, skor, label, alasan}, ... ] }."""
    signals = compute_signals(filtered_df, forecast_results)
    out = {}
    if signals.empty:
        return out
    for _, row in signals.iterrows():
        prov = row["Provinsi"]
        sector_scores = []
        for name, cfg in SECTORS.items():
            w = cfg["w"]
            score = (w["pajak_share"] * row["pajak_share"] +
                     w["kemandirian"] * row["kemandirian"] +
                     w["skala"] * row["skala"] +
                     w["tren"] * row["tren"]) * 100
            score = float(np.clip(score, 0, 100))
            if score >= 65:
                label = "Sangat Prospektif"
            elif score >= 50:
                label = "Prospektif"
            elif score >= 35:
                label = "Cukup / Hati-hati"
            else:
                label = "Kurang Mendukung"
            # alasan ringkas berbasis sinyal dominan
            drivers = []
            if row["pajak_share"] >= 0.6:
                drivers.append("aktivitas ekonomi lokal kuat")
            if row["kemandirian"] >= 0.6:
                drivers.append("kemandirian fiskal tinggi")
            elif row["kemandirian"] <= 0.35:
                drivers.append("masih bergantung transfer pusat")
            if row["skala"] >= 0.6:
                drivers.append("skala ekonomi besar")
            if row["tren"] >= 0.6:
                drivers.append("tren pendapatan menanjak")
            elif row["tren"] <= 0.4:
                drivers.append("tren pendapatan melemah")
            alasan = "; ".join(drivers) if drivers else "sinyal fiskal moderat"
            sector_scores.append({
                "sektor": name, "icon": cfg["icon"], "skor": round(score),
                "label": label, "narasi": cfg["narasi"], "alasan": alasan,
            })
        sector_scores.sort(key=lambda x: x["skor"], reverse=True)
        out[prov] = sector_scores
    return out


def top_recommendations(filtered_df, forecast_results=None, top_n=3):
    """Ringkasan: untuk tiap provinsi, sektor terbaik + rekomendasi tindakan."""
    scored = score_sectors(filtered_df, forecast_results)
    recs = []
    for prov, sectors in scored.items():
        best = sectors[0]
        weakest = sectors[-1]
        recs.append({
            "provinsi": prov,
            "top_sektor": best["sektor"], "top_icon": best["icon"],
            "top_skor": best["skor"], "top_label": best["label"],
            "top_alasan": best["alasan"],
            "hindari_sektor": weakest["sektor"], "hindari_skor": weakest["skor"],
            "semua": sectors[:top_n],
        })
    return recs
