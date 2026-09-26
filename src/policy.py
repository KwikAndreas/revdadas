"""
Rekomendasi kebijakan otomatis untuk RevDadas.

Menghasilkan rekomendasi berbasis-data dari hasil forecast (pertumbuhan
proyeksi pendapatan) dan deteksi anomali (potensi kebocoran/fraud).
Sifatnya indikatif sebagai bahan diskusi kebijakan, bukan keputusan final.
"""

import pandas as pd
import numpy as np

from .apbd_adapter import REVENUE_LEAF_ACCOUNTS

TOTAL_REVENUE = "Total Pendapatan Daerah"

# Akun agregat pendapatan -> akun penyusunnya (hierarki REVENUE_LEAF_ACCOUNTS)
AGGREGATE_CHILDREN = {
    "Pendapatan Asli Daerah (PAD)": [
        "Pajak Daerah",
        "Retribusi Daerah",
        "Hasil Pengelolaan Kekayaan Daerah yang Dipisahkan",
        "Lain-Lain PAD yang Sah",
    ],
    "Transfer ke Daerah dan Dana Desa (TKDD)": ["Pendapatan Transfer Pemerintah Pusat"],
    "Pendapatan Lainnya": [
        "Pendapatan Hibah",
        "Dana Darurat",
        "Lain-lain Pendapatan Sesuai dengan Ketentuan Peraturan Perundang-Undangan",
    ],
}


def _revenue_accounts(akt, fc):
    """
    Akun pembanding realisasi vs proyeksi (apel-ke-apel): akun total pendapatan,
    atau akun leaf bila total tidak tersedia — tidak pernah agregat + komponen + belanja.
    """
    akt_j, fc_j = set(akt["Jenis_Pendapatan"]), set(fc["Jenis_Pendapatan"])
    if TOTAL_REVENUE in akt_j and TOTAL_REVENUE in fc_j:
        return [TOTAL_REVENUE]
    leaf = [j for j in REVENUE_LEAF_ACCOUNTS if j in akt_j and j in fc_j]
    return leaf or sorted(akt_j & fc_j)


def _monthly_sum(df, value_col, accounts):
    return df[df["Jenis_Pendapatan"].isin(accounts)].groupby("Tanggal")[value_col].sum().sort_index()


def _isolate_aggregates(anomalies):
    """Buang anomali akun agregat bila komponennya ikut ter-flag di provinsi & bulan yang sama."""
    if not len(anomalies):
        return anomalies
    flagged = set(zip(anomalies["Provinsi"], anomalies["Tanggal"], anomalies["Jenis_Pendapatan"]))
    keep = [
        not any((p, t, c) in flagged for c in AGGREGATE_CHILDREN.get(j, []))
        for p, t, j in zip(anomalies["Provinsi"], anomalies["Tanggal"], anomalies["Jenis_Pendapatan"])
    ]
    return anomalies[keep]


def _fmt(v):
    v = float(v)
    if abs(v) >= 1e12:
        return f"Rp {v/1e12:.1f} T"
    if abs(v) >= 1e9:
        return f"Rp {v/1e9:.1f} M"
    if abs(v) >= 1e6:
        return f"Rp {v/1e6:.1f} Jt"
    return f"Rp {v:,.0f}".replace(",", ".")


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
                
            # Hanya total pendapatan: menjumlahkan semua akun mencampur agregat,
            # komponennya, dan belanja sehingga nominal & tren menjadi menyesatkan
            accounts = _revenue_accounts(akt, fc)
            akt_bulanan = _monthly_sum(akt, "Realisasi", accounts)
            base = akt_bulanan.tail(12).mean()
            proj = _monthly_sum(fc, "Prediksi", accounts).mean()

            # Capaian target tahun terakhir = Persentase realisasi kumulatif (YTD)
            # total pendapatan pada bulan terakhir yang tersedia
            last_year = akt['Tahun'].max()
            akt_last_year = akt[akt['Tahun'] == last_year]
            akt_last_total = akt_last_year[akt_last_year['Jenis_Pendapatan'] == TOTAL_REVENUE]
            if 'Persentase' in akt_last_year.columns and len(akt_last_total):
                avg_persentase = float(akt_last_total.sort_values('Tanggal')['Persentase'].iloc[-1])
            elif 'Persentase' in akt_last_year.columns:
                avg_persentase = akt_last_year['Persentase'].mean()
            else:
                avg_persentase = 0

            if base and base > 0:
                growth = (proj - base) / base * 100
                if growth < -2:
                    if avg_persentase > 90:
                        recs.append({
                            "judul": f"Penurunan Proyeksi Namun Target Terjaga di {prov}",
                            "prioritas": "Rendah",
                            "detail": f"Model mendeteksi proyeksi turun {abs(growth):.1f}%, namun capaian target TA {last_year} sudah {avg_persentase:.1f}% (optimal). Fokus pada efisiensi belanja.",
                            "kebijakan_existing": f"Pendekatan eksisting di {prov} cenderung menggenjot pungutan saat tren turun, yang berisiko menekan ekonomi lokal.",
                            "kelebihan": [f"Memberi ruang napas bagi wajib pajak di {prov}", "Mencegah efek Laffer Curve (pajak naik, penerimaan turun)"],
                            "kekurangan": [f"Mengancam postur APBD {prov} jika tren penurunan berlarut ke tahun depan"],
                            "indikator_dampak": "Stabilitas cash flow Pemda",
                            "kaitan_bisnis": "Mempersiapkan bantalan fiskal untuk mengantisipasi siklus ekonomi turun.",
                            "perbandingan": "Efisiensi Belanja vs Ekstensifikasi Pajak Agresif",
                            "justifikasi": f"Karena capaian di {prov} sudah {avg_persentase:.1f}%, manuver ekstraktif agresif saat ekonomi mendingin justru kontraproduktif."
                        })
                    else:
                        recs.append({
                            "judul": f"Waspada Perlambatan Kapasitas Fiskal di {prov}",
                            "prioritas": "Tinggi",
                            "detail": (f"Proyeksi rata-rata bulanan ({_fmt(proj)}) lebih rendah "
                                       f"{abs(growth):.1f}% dibanding rata-rata 12 bulan terakhir "
                                       f"({_fmt(base)}). Mengingat capaian target TA {last_year} baru {avg_persentase:.1f}%, pertimbangkan intensifikasi segera."),
                            "kebijakan_existing": f"Sistem pengawasan {prov} berjalan secara pasif, menunggu realisasi akhir tahun tanpa intervensi proaktif.",
                            "kelebihan": [f"Mencegah shortfall PAD progresif di {prov} sebelum terjadi", "Mengamankan pendanaan proyek strategis daerah"],
                            "kekurangan": ["Membutuhkan mobilisasi SDM pengawasan yang masif dan biaya operasional ekstra"],
                            "indikator_dampak": "Tingkat pencapaian target PAD (Persentase)",
                            "kaitan_bisnis": f"Mencegah defisit anggaran sebesar estimasi perlambatan {abs(growth):.1f}% yang dapat mengganggu belanja modal.",
                            "perbandingan": "Intervensi Proaktif vs Reaktif Pasca-Audit",
                            "justifikasi": f"Penurunan tren bulanan saat target belum tercapai ({avg_persentase:.1f}%) akan secara matematis menggerus kapasitas fiskal {prov}."
                        })
                elif growth > 8:
                    recs.append({
                        "judul": f"Optimalisasi Momentum Pertumbuhan di {prov}",
                        "prioritas": "Rendah",
                        "detail": (f"Proyeksi model mengindikasikan lonjakan tumbuh {growth:.1f}% dibanding rata-rata terakhir ({_fmt(base)} ke {_fmt(proj)}). "
                                   f"Akselerasikan momentum ini untuk memperkuat mandatory spending di {prov}."),
                        "kebijakan_existing": f"Surplus PAD di {prov} seringkali terserap ke belanja operasional non-produktif akibat lambatnya realokasi anggaran.",
                        "kelebihan": [f"Menciptakan ruang fiskal yang kuat untuk {prov}", "Meningkatkan daya serap anggaran di sektor riil"],
                        "kekurangan": [f"Perlu perencanaan birokrasi ekstra agar serapan belanja {prov} optimal dan tepat sasaran"],
                        "indikator_dampak": "Rasio Belanja Modal Produktif terhadap total APBD",
                        "kaitan_bisnis": f"Surplus proyeksi (tumbuh {growth:.1f}%) mendongkrak ekonomi makro {prov} melalui multiplier effect.",
                        "perbandingan": "Investasi Infrastruktur vs Konsumsi Jangka Pendek",
                        "justifikasi": f"Momentum percepatan PAD {prov} adalah window of opportunity terbaik untuk memenuhi batas mandatory spending."
                    })

    # --- 2. Anomali / potensi kebocoran ---
    if anomaly_results is not None and "Anomaly" in anomaly_results.columns:
        anomalies = anomaly_results[anomaly_results["Anomaly"] == True]

        # Selaras dengan KPI dashboard: akun pendapatan saja (tanpa akun total &
        # belanja), tahun anggaran terakhir, dan tanpa agregat yang komponennya
        # juga ter-flag — agar deviasi yang sama tidak terhitung dua-tiga kali
        anomalies = anomalies[
            (anomalies["Jenis_Pendapatan"] != TOTAL_REVENUE)
            & ~anomalies["Jenis_Pendapatan"].str.contains("Belanja", na=False)
        ]
        anom_year = None
        if "Tahun" in anomaly_results.columns and anomaly_results["Tahun"].notna().any():
            anom_year = int(anomaly_results["Tahun"].max())
            anomalies = anomalies[anomalies["Tahun"] == anom_year]
        anomalies = _isolate_aggregates(anomalies)

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
                "judul": f"Fokus Audit Anomali Terarah di {top_prov}",
                "prioritas": "Tinggi",
                "detail": (f"Algoritma mendeteksi {n_anom} catatan anomali"
                           f"{f' pada TA {anom_year}' if anom_year else ''} dengan total deviasi {_fmt(loss)} "
                           f"dari baseline wajar. "
                           f"Arahkan satgas pengawasan secara spesifik ke {top_prov}, khususnya pada pos \"{top_jenis}\". "
                           f"Audit prediktif ini berpotensi memulihkan kas daerah hingga "
                           f"~{_fmt(potensi_selamat)} (asumsi success rate {fraud_prevention_pct}%)."),
                "kebijakan_existing": f"Pemeriksaan rutin di {top_prov} pada pos {top_jenis} masih bersifat acak (random sampling) tanpa prioritas berbasis skor anomali.",
                "kelebihan": [f"Memulihkan kebocoran {_fmt(potensi_selamat)} secara efisien", f"Memberikan deterrence effect (efek jera) yang kuat di ekosistem {top_jenis}"],
                "kekurangan": [f"Resistensi awal dari wajib pajak/retribusi {top_jenis} terhadap pengawasan ketat"],
                "indikator_dampak": "Recovery Rate Kebocoran (Nilai rupiah yang dikembalikan ke Kas Daerah)",
                "kaitan_bisnis": f"Meningkatkan penerimaan secara instan sebesar {_fmt(potensi_selamat)} tanpa harus membuat Perda pajak baru.",
                "perbandingan": "Audit Berbasis Prediksi AI vs Sampling Acak Konvensional",
                "justifikasi": f"Model RevDadas melokalisasi deviasi pada {top_jenis}. Mengerahkan auditor ke titik spesifik ini 10x lebih hemat biaya daripada audit menyeluruh."
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
                        "detail": (f"Pos \"{top_name}\" mendominasi {top_share:.0f}% PAD {prov}. "
                                   f"Karena ini adalah DNA utama daerah, fokuslah pada INTENSIFIKASI (digitalisasi pungutan, insentif pelaku usaha) "
                                   f"dan PROTEKSI EKOSISTEM alih-alih memaksakan diversifikasi pada sektor yang tidak relevan."),
                        "kebijakan_existing": f"Pungutan {top_name} di {prov} masih bersifat ekstraktif tanpa adanya timbal balik nilai tambah layanan untuk wajib pajak.",
                        "kelebihan": [f"Memperkuat {top_name} sebagai economic engine utama {prov}", f"Mengamankan mayoritas PAD ({top_share:.0f}%) dari fluktuasi jangka pendek"],
                        "kekurangan": [f"Sangat rentan terhadap krisis makro yang secara spesifik menghantam sektor {top_name}"],
                        "indikator_dampak": "Index Kepuasan Wajib Pajak & Tax Ratio Sektoral",
                        "kaitan_bisnis": f"Menjaga 'angsa bertelur emas' ({top_name}) agar ekosistem usahanya tidak mati akibat over-taxation.",
                        "perbandingan": "Proteksi & Reinvestasi Ekosistem vs Ekstensifikasi Pajak Baru",
                        "justifikasi": f"Mengalihkan fokus dari sektor DNA ({top_share:.0f}% PAD) justru berisiko mematikan keunggulan komparatif {prov}."
                    })
                else:
                    recs.append({
                        "judul": f"Mitigasi Ketergantungan {top_name} di {prov}",
                        "prioritas": "Sedang",
                        "detail": (f"Pos \"{top_name}\" menyumbang porsi dominan {top_share:.0f}% dari PAD {prov}. "
                                   f"Ketergantungan ekstrem pada sektor yang bukan DNA asli daerah ini sangat rawan goncangan. "
                                   f"Perkuat diversifikasi PAD melalui ekstensifikasi ke sektor potensial lainnya."),
                        "kebijakan_existing": f"Strategi fiskal {prov} terlalu bertumpu pada {top_name} sehingga melupakan potensi dari retribusi/pajak sektor lain.",
                        "kelebihan": [f"Menyebar risiko fiskal agar {prov} tidak lumpuh saat sektor {top_name} terkontraksi"],
                        "kekurangan": [f"Penurunan sementara penerimaan {top_name} selama masa transisi diversifikasi"],
                        "indikator_dampak": "Indeks Konsentrasi PAD (Herfindahl-Hirschman Index)",
                        "kaitan_bisnis": f"Mencegah volatilitas kas daerah {prov} jika terjadi perubahan regulasi pusat terkait {top_name}.",
                        "perbandingan": "Ekstensifikasi Basis Pajak vs Intensifikasi Sektor Tunggal",
                        "justifikasi": f"Daerah dengan rasio ketergantungan {top_share:.0f}% pada satu sumber pajak rentan mengalami gagal bayar proyek multi-years saat krisis."
                    })

    # --- 4. Fallback ---
    if not recs:
        recs.append({
            "judul": "Kondisi Fiskal Relatif Stabil",
            "prioritas": "Rendah",
            "detail": ("Model AI tidak mendeteksi deviasi target atau anomali historis yang signifikan. "
                       "Pertahankan disiplin anggaran dan lanjutkan rutinitas pengelolaan keuangan daerah secara konservatif."),
            "kebijakan_existing": "Menjalankan rutinitas standar operasional prosedur (SOP) pencairan APBD.",
            "kelebihan": ["Menjamin stabilitas jalannya roda pemerintahan dan pelayanan publik"],
            "kekurangan": ["Cenderung kurang proaktif dalam mencari celah optimalisasi PAD baru"],
            "indikator_dampak": "Persentase realisasi vs target PAD",
            "kaitan_bisnis": "Memastikan likuiditas harian kas daerah tetap sehat untuk operasional standar.",
            "perbandingan": "Stabilitas Konservatif vs Inovasi Agresif",
            "justifikasi": "Saat prediktif indikator menunjukkan stabilitas tinggi, intervensi radikal justru dapat menimbulkan guncangan birokrasi."
        })

    # urutkan: Tinggi -> Sedang -> Rendah
    order = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}
    recs.sort(key=lambda r: order.get(r["prioritas"], 9))
    
    # Ambil top 5 saja agar tidak terlalu banyak jika banyak provinsi dipilih
    return recs[:5]
