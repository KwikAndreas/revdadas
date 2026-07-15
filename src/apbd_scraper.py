"""
install:
    pip install requests pandas lxml openpyxl
"""

import requests
import pandas as pd
import io
import os
import time

base_url = "https://djpk.kemenkeu.go.id/portal/csv_apbd"
raw_dir = "djpk_raw"
output_file = "apbd_master_2021_2025.csv"
delay = 0.2
retry = 3

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36"
}

provinsi = {
    "01": "Aceh",
    "02": "Sumatera Utara",
    "03": "Sumatera Barat",
    "04": "Riau",
    "05": "Jambi",
    "06": "Sumatera Selatan",
    "07": "Bengkulu",
    "08": "Lampung",
    "09": "DKI Jakarta",
    "10": "Jawa Barat",
    "11": "Jawa Tengah",
    "12": "DI Yogyakarta",
    "13": "Jawa Timur",
    "14": "Kalimantan Barat",
    "15": "Kalimantan Tengah",
    "16": "Kalimantan Selatan",
    "17": "Kalimantan Timur",
    "18": "Sulawesi Utara",
    "19": "Sulawesi Tengah",
    "20": "Sulawesi Selatan",
    "21": "Sulawesi Tenggara",
    "22": "Bali",
    "23": "NTB",
    "24": "NTT",
    "25": "Maluku",
    "26": "Papua",
    "27": "Maluku Utara",
    "28": "Banten",
    "29": "Bangka Belitung",
    "30": "Gorontalo",
    "31": "Kepulauan Riau",
    "32": "Papua Barat",
    "33": "Sulawesi Barat",
    "34": "Kalimantan Utara",
    "35": "Papua Selatan",
    "36": "Papua Tengah",
    "37": "Papua Pegunungan",
    "38": "Papua Barat Daya",
}

tahun_list = range(2021, 2026)
periode_list = range(1, 13)

akun_target = {
    "pendapatan_daerah": "Pendapatan Daerah",
    "PAD": "PAD",
    "TKDD": "TKDD",
    "belanja_daerah": "Belanja Daerah",
    "belanja_modal": "Belanja Modal",
}

kolom_akun_kandidat = [
    "Akun", "Uraian", "URAIAN", "Kode Akun", "kode_akun",
    "Description", "Nama Akun", "NAMA AKUN",
]
kolom_nilai_kandidat = [
    "Realisasi", "REALISASI", "Realisasi (Rp)", "Nilai Realisasi",
    "Actual", "Nilai", "NILAI",
]

os.makedirs(raw_dir, exist_ok=True)


def to_float(value):
    if value is None:
        return 0.0
    s = str(value).strip()
    if s in ("", "-", "nan", "None", "N/A"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass
    s2 = s.replace(".", "").replace(",", ".")
    try:
        return float(s2)
    except ValueError:
        return 0.0


def cari_kolom(df, kandidat):
    for k in kandidat:
        if k in df.columns:
            return k
    for col in df.columns:
        for k in kandidat:
            if k.lower() in col.lower():
                return col
    return None


def parse_spreadsheetml(content: bytes) -> pd.DataFrame | None:
    try:
        import lxml.etree as ET

        raw_text = content.decode("utf-8", errors="ignore")

        if "<Workbook" not in raw_text and "<ss:Workbook" not in raw_text:
            return None

        root = ET.fromstring(content)

        NS = "urn:schemas-microsoft-com:office:spreadsheet"
        ns = f"{{{NS}}}"

        table = root.find(f".//{ns}Table")
        if table is None:
            table = root.find(".//Table")
        if table is None:
            return None

        rows_data = []
        for child in table:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag != "Row":
                continue
            cells = []
            for cell_el in child:
                if cell_el.tag.split("}")[-1] != "Cell":
                    continue
                data_el = None
                for d in cell_el:
                    if d.tag.split("}")[-1] == "Data":
                        data_el = d
                        break
                text = data_el.text.strip() if data_el is not None and data_el.text else ""
                cells.append(text)
            if any(c for c in cells):
                rows_data.append(cells)

        if len(rows_data) < 2:
            return None

        max_cols = max(len(r) for r in rows_data)
        for r in rows_data:
            r.extend([""] * (max_cols - len(r)))

        df = pd.DataFrame(rows_data[1:], columns=rows_data[0])
        return df

    except Exception as e:
        print(f"    [SpreadsheetML] error: {e}")
        return None


def parse_table(content: bytes) -> pd.DataFrame | None:
    df = parse_spreadsheetml(content)
    if df is not None:
        return df

    for engine in ("xlrd", "openpyxl"):
        try:
            df = pd.read_excel(io.BytesIO(content), engine=engine)
            if df.shape[0] > 0:
                return df
        except Exception:
            pass

    try:
        tables = pd.read_html(io.BytesIO(content))
        for t in tables:
            if t.shape[0] > 3:
                return t
    except Exception:
        pass

    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, engine="python")
            if df.shape[0] > 0 and df.shape[1] > 2:
                return df
        except Exception:
            pass

    return None


def is_valid_response(content: bytes) -> bool:
    try:
        if not content or len(content) < 100:
            return False
        
        text = content.decode("utf-8", errors="ignore")
        
        if not text.startswith("<?xml"):
            return False
        
        if "<!DOCTYPE html>" in text or "<html lang=" in text:
            return False
        
        return True
    except:
        return False


def fetch_raw(tahun: int, bulan: int, kode: str) -> bytes | None:
    cache_path = os.path.join(raw_dir, f"{tahun}_{bulan:02d}_{kode}.dat")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            cached = f.read()
        if is_valid_response(cached):
            return cached
        else:
            print(f"    [Cache corrupt, re-downloading...]")
            os.remove(cache_path)

    params = {
        "type": "apbd",
        "periode": bulan,
        "tahun": tahun,
        "provinsi": kode,
        "pemda": "00",
    }

    for attempt in range(retry):
        try:
            r = requests.get(
                base_url, params=params,
                headers=headers, timeout=60
            )
            if r.status_code == 200 and r.content:
                if not is_valid_response(r.content):
                    print(f"    [Invalid response, attempt {attempt+1}/{retry}]")
                    time.sleep(2)
                    continue
                
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                return r.content
        except requests.RequestException as e:
            print(f"    [Fetch] attempt {attempt+1} gagal: {e}")
        time.sleep(2)

    return None


def ambil(df: pd.DataFrame, akun_label: str) -> float:
    col_akun = cari_kolom(df, kolom_akun_kandidat)
    col_nilai = cari_kolom(df, kolom_nilai_kandidat)

    if col_akun is None or col_nilai is None:
        return 0.0

    series = df[col_akun].astype(str).str.strip()

    mask = series.str.lower() == akun_label.lower()
    if mask.any():
        return to_float(df.loc[mask, col_nilai].iloc[0])

    mask = series.str.contains(akun_label, case=False, na=False)
    if mask.any():
        return to_float(df.loc[mask, col_nilai].iloc[0])

    return 0.0


def main():
    hasil = []
    existing_keys = set()
    
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            hasil = existing_df.to_dict('records')
            for row in hasil:
                key = (row['tahun'], row['bulan'], row['provinsi'])
                existing_keys.add(key)
            print(f"Loaded {len(hasil)} existing records")
        except Exception:
            pass
    
    total = len(tahun_list) * len(periode_list) * len(provinsi)
    count = 0
    skipped = 0

    for tahun in tahun_list:
        for bulan in periode_list:
            for kode, nama in provinsi.items():
                count += 1
                
                if (tahun, bulan, nama) in existing_keys:
                    skipped += 1
                    if skipped % 100 == 0:
                        print(f"[{count}/{total}] Skipped {skipped} existing records")
                    continue
                
                label = f"[{count}/{total}] {tahun}-{bulan:02d} {nama}"
                print(label, end=" ... ", flush=True)

                try:
                    content = fetch_raw(tahun, bulan, kode)
                    if content is None:
                        print("SKIP (download gagal)")
                        continue

                    df = parse_table(content)
                    if df is None:
                        print("SKIP (parse gagal)")
                        continue

                    row = {
                        "tahun": tahun,
                        "bulan": bulan,
                        "provinsi": nama,
                    }
                    for col_name, akun_label in akun_target.items():
                        row[col_name] = ambil(df, akun_label)

                    hasil.append(row)
                    print("OK")

                except Exception as e:
                    print(f"ERROR: {e}")

                time.sleep(delay)

            if hasil:
                pd.DataFrame(hasil).to_csv(output_file, index=False)

    pd.DataFrame(hasil).to_csv(output_file, index=False)
    print(f"\nSelesai - {len(hasil)} baris")
    if skipped > 0:
        print(f"Skipped {skipped} records")


if __name__ == "__main__":
    main()
