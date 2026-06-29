import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_RAW_PATH = Path("data/raw")
DATA_PROCESSED_PATH = Path("data/processed")

REVENUE_CATEGORIES = {
    "Pajak Daerah": {
        "keywords": ["Pajak Daerah", "Regional Tax"],
        "bps_code": "1.1"
    },
    "Retribusi Daerah": {
        "keywords": ["Retribusi Daerah", "Regional Retribution"],
        "bps_code": "1.2"
    },
    "Hasil BUMN": {
        "keywords": ["Hasil Perusahaan Milik Daerah", "Regional-Owned Company"],
        "bps_code": "1.3"
    }
}


def clean_numeric_value(value):
    if pd.isna(value) or value == "-" or str(value).strip() == "":
        return None
    
    value = str(value).strip()
    
    if not value:
        return None
    
    multiplier = 1
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
        multiplier = -1
    
    value = value.replace('"', '')
    
    dot_count = value.count(".")
    comma_count = value.count(",")
    space_count = value.count(" ")

    value = value.replace(" ", "")
    
    if comma_count > 0 and dot_count > 0:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif dot_count > 0 and comma_count == 0:
        last_dot_pos = value.rfind(".")
        digits_after = len(value) - last_dot_pos - 1
        if digits_after <= 2 and dot_count == 1:
            pass 
        else:
            value = value.replace(".", "")
    elif comma_count > 0 and dot_count == 0:
        last_comma_pos = value.rfind(",")
        digits_after = len(value) - last_comma_pos - 1
        if digits_after <= 2 and comma_count == 1:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
    
    try:
        numeric = float(value) * multiplier
        return numeric
    except ValueError:
        logger.warning(f"Could not convert value '{value}' to numeric")
        return None


def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    if match:
        return int(match.group(1))
    return None


def extract_provincia_from_filename(filename):
    if "DKI Jakarta" in filename:
        return "DKI Jakarta"
    elif "Jawa Barat" in filename:
        return "Jawa Barat"
    elif "Jawa Timur" in filename:
        return "Jawa Timur"
    return None

def detect_currency_unit(filename):
    filename_lower = filename.lower()
    if "ribu rupiah" in filename_lower or "thousand rupiah" in filename_lower:
        return 'ribu_rupiah'
    elif "rupiah" in filename_lower and "ribu" not in filename_lower:
        return 'rupiah'
    return 'ribu_rupiah'


def parse_bps_csv(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        
        filename = filepath.name
        provinsi = extract_provincia_from_filename(filename)
        tahun = extract_year_from_filename(filename)
        currency_unit = detect_currency_unit(filename)
        
        if not provinsi or not tahun:
            print(f"⚠️  Skipping {filename}: Cannot extract provincia/tahun")
            return None
        
        multiplier = 1000 if currency_unit == 'ribu_rupiah' else 1
        unit_label = "ribu rupiah" if currency_unit == 'ribu_rupiah' else "rupiah"
        
        data_dict = {
            'Provinsi': provinsi,
            'Tahun': tahun,
            'Unit': currency_unit,
            'Data': {}
        }
        
        for idx, row in df.iterrows():
            row_text = str(row[0]) if pd.notna(row[0]) else ""
            row_value = row[1] if len(row) > 1 else None
            
            for category_name, category_info in REVENUE_CATEGORIES.items():
                for keyword in category_info['keywords']:
                    if keyword in row_text:
                        numeric_value = clean_numeric_value(row_value)
                        if numeric_value:
                            normalized_value = numeric_value * multiplier
                            data_dict['Data'][category_name] = normalized_value
                            log_value = normalized_value / 1e12 if normalized_value >= 1e12 else normalized_value / 1e9
                            unit_label_display = "T" if normalized_value >= 1e12 else "M"
                            print(f"✅ Found {category_name}: Rp {log_value:.2f}{unit_label_display} [{unit_label}] ({filename})")
        
        return data_dict if data_dict['Data'] else None
    
    except Exception as e:
        print(f"❌ Error parsing {filepath}: {e}")
        return None


def consolidate_all_files():
    all_data = []
    
    csv_files = sorted(DATA_RAW_PATH.glob("*.csv"))
    print(f"\n📁 Found {len(csv_files)} CSV files\n")
    
    for filepath in csv_files:
        print(f"📄 Processing: {filepath.name}")
        parsed = parse_bps_csv(filepath)
        if parsed:
            all_data.append(parsed)
    
    return all_data


def generate_monthly_data(annual_data):
    monthly_records = []
    
    for record in annual_data:
        provinsi = record['Provinsi']
        tahun = record['Tahun']
        
        for category_name, annual_value in record['Data'].items():
            base_monthly = annual_value / 12
            
            for bulan in range(1, 13):
                if bulan in [1, 4, 7, 10]:
                    seasonal_factor = 1.05
                elif bulan in [12]:
                    seasonal_factor = 1.15
                else:
                    seasonal_factor = 0.95 + np.random.uniform(-0.03, 0.03)
                
                monthly_value = base_monthly * seasonal_factor
                
                monthly_records.append({
                    'Tahun': tahun,
                    'Bulan': bulan,
                    'Tanggal': f"{tahun}-{bulan:02d}-01",
                    'Provinsi': provinsi,
                    'Jenis_Pendapatan': category_name,
                    'Realisasi': monthly_value
                })
    
    return pd.DataFrame(monthly_records)


def main():
    print("\n" + "="*70)
    print("🔧 BPS DATA CONSOLIDATION PIPELINE")
    print("="*70)
    
    print("\n1️⃣  STEP 1: Parsing BPS CSV Files")
    print("-" * 70)
    annual_data = consolidate_all_files()
    print(f"\n✅ Successfully parsed {len(annual_data)} files")
    
    print("\n2️⃣  STEP 2: Generating Monthly Data from Annual Format")
    print("-" * 70)
    df_monthly = generate_monthly_data(annual_data)
    print(f"✅ Generated {len(df_monthly)} monthly records")
    print(f"   - Provinces: {df_monthly['Provinsi'].unique().tolist()}")
    print(f"   - Categories: {df_monthly['Jenis_Pendapatan'].unique().tolist()}")
    print(f"   - Years: {sorted(df_monthly['Tahun'].unique())}")
    
    print("\n3️⃣  STEP 3: Saving Processed Data")
    print("-" * 70)
    
    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    output_file = DATA_PROCESSED_PATH / "revenue_consolidated.csv"
    df_monthly.to_csv(output_file, index=False)
    print(f"✅ Saved to: {output_file}")
    
    print("\n📊 Sample Data (first 10 rows):")
    print("-" * 70)
    print(df_monthly.head(10).to_string())
    
    print("\n📈 Summary Statistics:")
    print("-" * 70)
    print(df_monthly.groupby(['Provinsi', 'Jenis_Pendapatan'])['Realisasi'].agg(['min', 'mean', 'max']))
    
    print("\n" + "="*70)
    print("✅ DATA CONSOLIDATION COMPLETED!")
    print("="*70 + "\n")
    
    return df_monthly

if __name__ == "__main__":
    df = main()