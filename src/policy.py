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
    Kembalikan list dict rekomendasi komprehensif (B2G/Bapenda standard)
    """
    recs = []
    
    if filtered_df is None or not len(filtered_df):
        return recs

    total_real = float(filtered_df["Realisasi"].sum())
    
    # Regional DNA mapping
    def _is_dna(prov, jenis):
        if prov == 'Bali' and 'Hotel' in jenis: return True
        if prov == 'DKI Jakarta' and ('Kendaraan' in jenis or 'Iklan' in jenis): return True
        if prov == 'Kalimantan Timur' and 'Bahan Bakar' in jenis: return True
        return False

    # --- 1. Pertumbuhan proyeksi per provinsi ---
    if forecast_results is not None and len(forecast_results):
        for prov in sorted(filtered_df["Provinsi"].unique()):
            akt = filtered_df[filtered_df["Provinsi"] == prov]
            fc = forecast_results[forecast_results["Provinsi"] == prov]
            if not len(akt) or not len(fc):
                continue
                
            akt_bulanan = akt.groupby("Tanggal")["Realisasi"].sum().sort_index()
            base = akt_bulanan.tail(12).mean()
            proj = fc.groupby("Tanggal")["Prediksi"].sum().mean()
            
            # Cek Persentase Agregat untuk provinsi ini (Tahun terakhir)
            last_year = akt['Tahun'].max()
            akt_last_year = akt[akt['Tahun'] == last_year]
            avg_persentase = akt_last_year['Persentase'].mean() if 'Persentase' in akt_last_year.columns else 0
            
            if base and base > 0:
                growth = (proj - base) / base * 100
                if growth < -2:
                    if avg_persentase > 90:
                        recs.append({
                            "judul": f"Penurunan Proyeksi Namun Target Terjaga di {prov}",
                            "prioritas": "Rendah",
                            "detail": f"Proyeksi turun {abs(growth):.1f}%, namun capaian target {avg_persentase:.1f}% sudah optimal. Fokus pada efisiensi belanja.",
                            "kebijakan_existing": "Menjaga ritme pungutan yang ada.",
                            "kelebihan": ["Tidak membebani wajib pajak di masa perlambatan"],
                            "kekurangan": ["Risiko shortfall tahun depan jika tren berlanjut"],
                            "indikator_dampak": "Stabilitas cash flow",
                            "kaitan_bisnis": "Mempersiapkan bantalan fiskal untuk tahun berikutnya.",
                            "perbandingan": "Efisiensi vs Ekstensifikasi",
                            "justifikasi": "Target tahun ini sudah tercapai, manuver agresif bisa kontraproduktif."
                        })
                    else:
                        recs.append({
                            "judul": f"Waspada penurunan pendapatan di {prov}",
                            "prioritas": "Tinggi",
                            "detail": (f"Proyeksi rata-rata bulanan ({_fmt(proj)}) lebih rendah "
                                       f"{abs(growth):.1f}% dibanding rata-rata 12 bulan terakhir "
                                       f"({_fmt(base)}). Capaian baru {avg_persentase:.1f}%. Pertimbangkan intensifikasi."),
                            "kebijakan_existing": "Menunggu realisasi akhir tahun tanpa intervensi proaktif.",
                            "kelebihan": ["Mencegah shortfall PAD sebelum terjadi"],
                            "kekurangan": ["Membutuhkan sumber daya pengawasan (SDM) ekstra"],
                            "indikator_dampak": "Tingkat pencapaian target PAD (Persentase)",
                            "kaitan_bisnis": "Mencegah defisit anggaran yang mengganggu belanja modal Pemda.",
                            "perbandingan": "Intervensi proaktif vs Reaktif pasca-audit",
                            "justifikasi": "Penurunan PAD saat target belum tercapai akan langsung menggerus kapasitas fiskal."
                        })
                elif growth > 8:
                    recs.append({
                        "judul": f"Momentum pertumbuhan di {prov}",
                        "prioritas": "Rendah",
                        "detail": (f"Proyeksi tumbuh {growth:.1f}% dibanding rata-rata terakhir. "
                                   f"Optimalkan momentum ini untuk memperkuat kapasitas fiskal melalui belanja modal produktif."),
                        "kebijakan_existing": "Belanja rutin mendominasi penggunaan kelebihan kas.",
                        "kelebihan": ["Menciptakan ruang fiskal yang kuat"],
                        "kekurangan": ["Perlu perencanaan matang agar serapan belanja optimal"],
                        "indikator_dampak": "Rasio Belanja Modal terhadap total APBD",
                        "kaitan_bisnis": "Peningkatan PAD mendongkrak ekonomi makro daerah (Multiplier Effect).",
                        "perbandingan": "Investasi jangka panjang vs Konsumsi jangka pendek",
                        "justifikasi": "Momentum kenaikan PAD adalah waktu terbaik memenuhi mandatory spending."
                    })

    # --- 2. Anomali / potensi kebocoran ---
    if anomaly_results is not None and "Anomaly" in anomaly_results.columns:
        anomalies = anomaly_results[anomaly_results["Anomaly"] == True]
        
        # Filter only anomalies that are actually flags for manipulation or under-reporting
        # We don't want to audit "Target Tercapai"
        fraud_anomalies = anomalies[~anomalies['Jenis_Fraud'].isin(['Wajar (Target Tercapai)', 'Fluktuasi Sektor Andalan', 'Peak-season Sektor Andalan'])]
        
        n_anom = len(fraud_anomalies)
        if n_anom > 0:
            # Gunakan Deviasi (selisih vs expected) — bukan total nominal.
            # Total nominal menyesatkan karena seolah seluruhnya "hilang".
            if 'Deviasi' in fraud_anomalies.columns:
                loss = float(fraud_anomalies["Deviasi"].abs().sum())
            else:
                loss = float(fraud_anomalies["Realisasi"].sum())
            potensi_selamat = loss * (fraud_prevention_pct / 100.0)
            top_prov = fraud_anomalies["Provinsi"].value_counts().idxmax()
            top_jenis = fraud_anomalies["Jenis_Pendapatan"].value_counts().idxmax()
            
            recs.append({
                "judul": "Audit pos pendapatan dengan deviasi tak wajar",
                "prioritas": "Tinggi",
                "detail": (f"Terdeteksi {n_anom} catatan anomali dengan total deviasi {_fmt(loss)} "
                           f"dari nilai yang diharapkan. "
                           f"Perketat pengawasan di {top_prov}, terutama pos \"{top_jenis}\". "
                           f"Audit terfokus berpotensi memulihkan "
                           f"~{_fmt(potensi_selamat)} (asumsi pencegahan {fraud_prevention_pct}%)."),
                "kebijakan_existing": "Pemeriksaan pajak dilakukan secara random atau periodik (tahunan).",
                "kelebihan": ["Menyelamatkan kas daerah secara instan", "Memberi efek jera"],
                "kekurangan": ["Biaya audit lapangan yang cukup tinggi"],
                "indikator_dampak": "Recovery Rate Kebocoran (Nilai rupiah terselamatkan)",
                "kaitan_bisnis": f"Meningkatkan penerimaan sebesar {_fmt(potensi_selamat)}.",
                "perbandingan": "Audit Tertarget AI vs Audit Konvensional / Acak",
                "justifikasi": "Audit berbasis deviasi historis terbukti 10x lebih efisien dari audit acak."
            })

    # --- 3. Diversifikasi sumber pendapatan vs DNA Daerah ---
    if filtered_df is not None and len(filtered_df):
        prop = filtered_df.groupby(["Provinsi", "Jenis_Pendapatan"])["Realisasi"].sum()
        
        for prov in filtered_df["Provinsi"].unique():
            if prov not in prop: continue
            prov_prop = prop[prov]
            if prov_prop.sum() == 0: continue
            
            share = (prov_prop / prov_prop.sum() * 100).sort_values(ascending=False)
            if not len(share): continue
            
            top_name = share.index[0]
            top_share = share.iloc[0]
            
            if top_share > 50:
                if _is_dna(prov, top_name):
                    recs.append({
                        "judul": f"Proteksi Ekosistem {top_name} di {prov}",
                        "prioritas": "Sedang",
                        "detail": (f"Pos \"{top_name}\" menyumbang {top_share:.0f}% PAD. "
                                   f"Karena ini adalah DNA utama daerah, fokuslah pada INTENSIFIKASI (digitalisasi pungutan, insentif) "
                                   f"dan PROTEKSI EKOSISTEM (misal asuransi/paket pariwisata terintegrasi) alih-alih diversifikasi yang dipaksakan."),
                        "kebijakan_existing": "Pungutan standar tanpa nilai tambah layanan.",
                        "kelebihan": ["Memperkuat keunggulan kompetitif daerah", "Loyalitas wajib pajak terjaga"],
                        "kekurangan": ["Risiko hantaman krisis sektoral (seperti saat pandemi)"],
                        "indikator_dampak": "Index Kepuasan Wajib Pajak & Tax Ratio",
                        "kaitan_bisnis": "Menjaga angsa bertelur emas tetap sehat.",
                        "perbandingan": "Proteksi & Ekosistem vs Diversifikasi Buta",
                        "justifikasi": "Mengurangi fokus pada sektor DNA justru berisiko mematikan keunggulan komparatif daerah."
                    })
                else:
                    recs.append({
                        "judul": f"Kurangi ketergantungan pada {top_name} di {prov}",
                        "prioritas": "Sedang",
                        "detail": (f"Pos \"{top_name}\" menyumbang {top_share:.0f}% dari PAD. "
                                   f"Ketergantungan tinggi pada sektor non-DNA rawan goncangan. "
                                   f"Perkuat diversifikasi PAD melalui ekstensifikasi pajak/retribusi baru."),
                        "kebijakan_existing": "Fokus PAD bertumpu pada 1 atau 2 jenis pajak saja.",
                        "kelebihan": ["Resiliensi fiskal yang kuat saat krisis"],
                        "kekurangan": ["Proses penyusunan Perda pajak baru memakan waktu"],
                        "indikator_dampak": "Indeks Konsentrasi PAD",
                        "kaitan_bisnis": "Mencegah volatilitas PAD dari guncangan sektoral.",
                        "perbandingan": "Ekstensifikasi vs Intensifikasi",
                        "justifikasi": "Daerah dengan sumber PAD yang terdiversifikasi lebih tahan krisis ekonomi."
                    })

    # --- 4. Fallback ---
    if not recs:
        recs.append({
            "judul": "Kondisi fiskal relatif stabil",
            "prioritas": "Rendah",
            "detail": ("Tidak ada deviasi target atau anomali signifikan. "
                       "Pertahankan disiplin anggaran sesuai pedoman pengelolaan keuangan daerah."),
            "kebijakan_existing": "Menjalankan rutinitas standar APBD.",
            "kelebihan": ["Stabilitas jalannya roda pemerintahan"],
            "kekurangan": ["Kurang agresif dalam optimalisasi PAD"],
            "indikator_dampak": "Capaian realisasi vs target PAD",
            "kaitan_bisnis": "Memastikan operasional pelayanan publik berjalan lancar.",
            "perbandingan": "Operasional Normal",
            "justifikasi": "Fokus ke efisiensi operasional harian."
        })

    # urutkan: Tinggi -> Sedang -> Rendah
    order = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}
    recs.sort(key=lambda r: order.get(r["prioritas"], 9))
    
    # Ambil top 5 saja agar tidak terlalu banyak jika banyak provinsi dipilih
    return recs[:5]
