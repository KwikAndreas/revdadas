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
    Kembalikan list dict rekomendasi komprehensif (B2G/Bapenda standard):
        {judul, prioritas, detail, kebijakan_existing, kelebihan, kekurangan, indikator_dampak, kaitan_bisnis, perbandingan, justifikasi}
    """
    recs = []

    total_real = float(filtered_df["Realisasi"].sum()) if filtered_df is not None and len(filtered_df) else 0.0

    # --- 1. Pertumbuhan proyeksi per provinsi ---
    if forecast_results is not None and len(forecast_results) and filtered_df is not None:
        for prov in sorted(filtered_df["Provinsi"].unique()):
            akt = filtered_df[filtered_df["Provinsi"] == prov]
            fc = forecast_results[forecast_results["Provinsi"] == prov]
            if not len(akt) or not len(fc):
                continue
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
                        "kebijakan_existing": "Menunggu realisasi akhir tahun tanpa intervensi proaktif.",
                        "kelebihan": ["Mencegah *shortfall* PAD sebelum terjadi", "Optimalisasi piutang pajak"],
                        "kekurangan": ["Membutuhkan sumber daya pengawasan (SDM) ekstra", "Potensi resistensi wajib pajak"],
                        "indikator_dampak": "Tingkat pencapaian target PAD (Persentase)",
                        "kaitan_bisnis": "Mencegah defisit anggaran yang mengganggu belanja modal Pemda.",
                        "perbandingan": "Intervensi proaktif vs Reaktif pasca-audit",
                        "justifikasi": "Penurunan PAD langsung menggerus kapasitas fiskal. Tindakan preventif lebih murah dari pada menambal defisit via utang/SiLPA."
                    })
                elif growth > 8:
                    recs.append({
                        "judul": f"Momentum pertumbuhan di {prov}",
                        "prioritas": "Rendah",
                        "detail": (f"Proyeksi tumbuh {growth:.1f}% dibanding rata-rata terakhir. "
                                   f"Sesuai amanat UU HKPD No. 1/2022, optimalkan momentum ini untuk "
                                   f"memperkuat kapasitas fiskal daerah melalui belanja modal produktif."),
                        "kebijakan_existing": "Belanja rutin mendominasi penggunaan kelebihan kas.",
                        "kelebihan": ["Menciptakan ruang fiskal yang kuat", "Memperbesar porsi belanja publik"],
                        "kekurangan": ["Perlu perencanaan matang agar serapan belanja tidak menumpuk di Q4"],
                        "indikator_dampak": "Rasio Belanja Modal terhadap total APBD",
                        "kaitan_bisnis": "Peningkatan PAD digunakan untuk mendongkrak ekonomi makro daerah (Multiplier Effect).",
                        "perbandingan": "Investasi jangka panjang vs Konsumsi birokrasi jangka pendek",
                        "justifikasi": "UU HKPD memandatkan mandatory spending yang progresif. Momentum kenaikan PAD adalah waktu terbaik."
                    })

    # --- 2. Anomali / potensi kebocoran ---
    if anomaly_results is not None and "Anomaly" in anomaly_results.columns:
        anomalies = anomaly_results[anomaly_results["Anomaly"] == True]
        n_anom = len(anomalies)
        if n_anom > 0:
            loss = float(anomalies["Realisasi"].sum())
            potensi_selamat = loss * (fraud_prevention_pct / 100.0)
            top_prov = anomalies["Provinsi"].value_counts().idxmax()
            top_jenis = anomalies["Jenis_Pendapatan"].value_counts().idxmax()
            recs.append({
                "judul": "Audit pos pendapatan beranomali",
                "prioritas": "Tinggi",
                "detail": (f"Terdeteksi {n_anom} catatan anomali (nominal {_fmt(loss)}). "
                           f"Sesuai kewenangan UU HKPD No. 1/2022, perketat pengawasan pajak di {top_prov}, "
                           f"terutama pos \"{top_jenis}\". Audit terfokus dapat menyelamatkan "
                           f"~{_fmt(potensi_selamat)} (pencegahan {fraud_prevention_pct}%)."),
                "kebijakan_existing": "Pemeriksaan pajak dilakukan secara random atau periodik (tahunan).",
                "kelebihan": ["Menyelamatkan kas daerah secara instan", "Memberi efek jera (deterrent effect)"],
                "kekurangan": ["Biaya audit lapangan yang cukup tinggi", "Rawan friksi di lapangan"],
                "indikator_dampak": "Rasio *Recovery Rate* Kebocoran (Nilai rupiah terselamatkan)",
                "kaitan_bisnis": f"Meningkatkan penerimaan sebesar {_fmt(potensi_selamat)} dan menekan angka kebocoran.",
                "perbandingan": "Audit Tertarget AI vs Audit Konvensional / Acak",
                "justifikasi": "Audit berbasis risiko (Risk-Based Audit) menggunakan deteksi anomali AI terbukti 10x lebih efisien dari audit konvensional."
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
                    "detail": (f"Pos \"{top_name}\" menyumbang {top_share:.0f}% dari PAD terpilih. "
                               f"Ketergantungan tinggi rawan goncangan ekonomi. Rujuk ruang lingkup "
                               f"pajak/retribusi baru pada UU HKPD No. 1/2022 untuk perkuat diversifikasi PAD."),
                    "kebijakan_existing": "Fokus PAD hanya bertumpu pada 1 atau 2 jenis pajak primadona.",
                    "kelebihan": ["Resiliensi fiskal yang kuat saat krisis", "Membuka basis pajak (tax base) baru"],
                    "kekurangan": ["Proses penyusunan Perda pajak baru memakan waktu (DPRD)", "Resistensi tarif baru"],
                    "indikator_dampak": "Indeks Konsentrasi PAD (Herfindahl-Hirschman Index)",
                    "kaitan_bisnis": "Mencegah volatilitas PAD dari guncangan sektoral spesifik.",
                    "perbandingan": "Ekstensifikasi (perluasan basis) vs Intensifikasi (peras existing)",
                    "justifikasi": "Daerah dengan sumber PAD yang terdiversifikasi lebih tahan terhadap krisis ekonomi sektoral (seperti saat pandemi)."
                })

    # --- 4. Fallback ---
    if not recs:
        recs.append({
            "judul": "Kondisi fiskal relatif stabil",
            "prioritas": "Rendah",
            "detail": ("Tidak ada sinyal risiko signifikan dari proyeksi maupun anomali pada "
                       "data terpilih. Pertahankan disiplin anggaran sesuai pedoman pengelolaan "
                       "keuangan daerah (UU HKPD)."),
            "kebijakan_existing": "Menjalankan rutinitas standar APBD.",
            "kelebihan": ["Stabilitas jalannya roda pemerintahan"],
            "kekurangan": ["Kurang agresif dalam optimalisasi PAD"],
            "indikator_dampak": "Capaian realisasi vs target PAD bulanan",
            "kaitan_bisnis": "Memastikan operasional pelayanan publik berjalan lancar sesuai DPA.",
            "perbandingan": "Operasional Normal",
            "justifikasi": "Sistem tidak mendeteksi deviasi signifikan. Fokus ke efisiensi operasional harian."
        })

    # urutkan: Tinggi -> Sedang -> Rendah
    order = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}
    recs.sort(key=lambda r: order.get(r["prioritas"], 9))
    return recs
