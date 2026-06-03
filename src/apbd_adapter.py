"""
Adapter data APBD DJPK Kemenkeu -> skema RevDadas.

Mengubah file mentah `apbd_djpk_master_2021-2025.csv` (postur APBD,
realisasi KUMULATIF per bulan) menjadi `revenue_consolidated.csv` dengan
skema yang dibutuhkan aplikasi:

    Tanggal, Tahun, Bulan, Provinsi, Jenis_Pendapatan, Realisasi

Realisasi di sini adalah nilai BULANAN (hasil de-kumulasi), bukan kumulatif.

CATATAN KUALITAS DATA:
- 2021 & 2022 di SIKD hanya tersimpan sebagai ANGKA TAHUNAN (nilai sama di
  semua bulan), sehingga TIDAK bisa dijadikan deret bulanan. Tahun-tahun ini
  DIKECUALIKAN dari data bulanan aplikasi.
- 2023-2025 memiliki progres realisasi bulanan kumulatif yang asli.
- 2025 kemungkinan masih preliminer.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Akun pendapatan "leaf" yang dijumlahkan = total Pendapatan Daerah.
# (subtotal seperti PAD/TKDD/Pendapatan Daerah sengaja TIDAK dipakai agar
#  tidak terjadi penghitungan ganda.)
REVENUE_LEAF_ACCOUNTS = [
    "Pajak Daerah",
    "Retribusi Daerah",
    "Hasil Pengelolaan Kekayaan Daerah yang Dipisahkan",
    "Lain-Lain PAD yang Sah",
    "Pendapatan Transfer Pemerintah Pusat",
    "Pendapatan Hibah",
    "Dana Darurat",
    "Lain-lain Pendapatan Sesuai dengan Ketentuan Peraturan Perundang-Undangan",
    "Pendapatan Transfer Antar Daerah",
]

# Tahun dengan progres bulanan asli
MONTHLY_YEARS = [2023, 2024, 2025]


def build_revenue_consolidated(master_csv_path, years=None, accounts=None):
    """Baca master APBD dan kembalikan DataFrame skema RevDadas (bulanan)."""
    years = years or MONTHLY_YEARS
    accounts = accounts or REVENUE_LEAF_ACCOUNTS

    df = pd.read_csv(master_csv_path)
    df["akun"] = df["akun"].astype(str).str.strip()

    df = df[df["tahun"].isin(years) & df["akun"].isin(accounts)].copy()

    rows = []
    for (prov, akun, tahun), grp in df.groupby(["provinsi", "akun", "tahun"]):
        grp = grp.sort_values("bulan")
        prev = 0.0
        for _, r in grp.iterrows():
            kum = r["realisasi"]
            if pd.isna(kum):
                continue
            inc = kum - prev           # de-kumulasi
            prev = kum
            inc = max(inc, 0.0)        # bersihkan artefak revisi (negatif)
            if inc <= 0:
                continue
            bulan = int(r["bulan"])
            rows.append({
                "Tanggal": f"{int(tahun)}-{bulan:02d}-01",
                "Tahun": int(tahun),
                "Bulan": bulan,
                "Provinsi": prov,
                "Jenis_Pendapatan": akun,
                "Realisasi": inc,
            })

    out = pd.DataFrame(rows)
    out["Tanggal"] = pd.to_datetime(out["Tanggal"])
    out = out.sort_values(["Provinsi", "Jenis_Pendapatan", "Tanggal"]).reset_index(drop=True)
    logger.info(f"Adapter APBD: {len(out)} baris, "
                f"{out['Jenis_Pendapatan'].nunique()} jenis pendapatan, "
                f"{out['Provinsi'].nunique()} provinsi")
    return out


def save_revenue_consolidated(master_csv_path, out_csv_path, **kwargs):
    out = build_revenue_consolidated(master_csv_path, **kwargs)
    out.to_csv(out_csv_path, index=False)
    logger.info(f"Disimpan: {out_csv_path}")
    return out
