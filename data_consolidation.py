"""
Data Consolidation Script untuk BPS Revenue Data
Parse format BPS yang berantakan dan consolidate ke format standard
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
DATA_RAW_PATH = Path("data/raw")
DATA_PROCESSED_PATH = Path("data/processed")

# Mapping kategori revenue yang relevan dari struktur BPS
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
    """Convert BPS numeric string to float, handling various Rupiah formats
    
    Handles formats like:
    - "1.234.567,89" (Indonesian: dot for thousands, comma for decimal)
    - "1,234,567.89" (English: comma for thousands, dot for decimal)
    - "1234567.89" (No separators)
    - "1 234 567,89" (Space for thousands, comma for decimal)
    - "-1234567" (Negative)
    """
    if pd.isna(value) or value == "-" or str(value).strip() == "":
        return None
    
    value = str(value).strip()
    
    if not value:
        return None
    
    # Handle negative values in parentheses: (1234567)
    multiplier = 1
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
        multiplier = -1
    
    # Remove text, spaces (but track them), quotes
    value = value.replace('"', '')
    
    # Detect format by analyzing separator positions
    dot_count = value.count(".")
    comma_count = value.count(",")
    space_count = value.count(" ")
    
    # Clean spaces first (spaces are always thousands separators)
    value = value.replace(" ", "")
    
    # Determine decimal separator:
    # - If both dots and commas exist: last one is decimal separator
    # - If only one type exists: check position/frequency
    if comma_count > 0 and dot_count > 0:
        # Both exist - last one is decimal separator
        if value.rfind(",") > value.rfind("."):
            # Indonesian format: 1.234.567,89
            value = value.replace(".", "").replace(",", ".")
        else:
            # English format: 1,234,567.89
            value = value.replace(",", "")
    elif dot_count > 0 and comma_count == 0:
        # Only dots - could be thousands separator or decimal
        # If 3 or fewer digits after last dot, likely decimal
        last_dot_pos = value.rfind(".")
        digits_after = len(value) - last_dot_pos - 1
        if digits_after <= 2 and dot_count == 1:
            # Single dot with 1-2 digits after = decimal
            pass  # Keep as is
        else:
            # Multiple dots = thousands separators (Indonesian style)
            value = value.replace(".", "")
    elif comma_count > 0 and dot_count == 0:
        # Only commas - could be thousands or decimal
        # If 3 or fewer digits after last comma and only 1 comma, likely decimal
        last_comma_pos = value.rfind(",")
        digits_after = len(value) - last_comma_pos - 1
        if digits_after <= 2 and comma_count == 1:
            # Single comma with 1-2 digits after = decimal (English style)
            value = value.replace(",", ".")
        else:
            # Multiple commas = thousands separators (English style)
            value = value.replace(",", "")
    
    try:
        numeric = float(value) * multiplier
        return numeric
    except ValueError:
        logger.warning(f"Could not convert value '{value}' to numeric")
        return None


def extract_year_from_filename(filename):
    """Extract year from BPS filename"""
    # Pattern: ...2021.csv, ...2021 (1).csv, ...2021-2022.csv
    match = re.search(r'(\d{4})', filename)
    if match:
        return int(match.group(1))
    return None


def extract_provincia_from_filename(filename):
    """Extract province name from filename"""
    if "DKI Jakarta" in filename:
        return "DKI Jakarta"
    elif "Jawa Barat" in filename:
        return "Jawa Barat"
    elif "Jawa Timur" in filename:
        return "Jawa Timur"
    return None

def detect_currency_unit(filename):
    """Detect currency unit from filename
    
    Returns:
        'ribu_rupiah' if file mentions "ribu rupiah"
        'rupiah' if file mentions "Rupiah" or similar
        None if unable to detect
    """
    filename_lower = filename.lower()
    if "ribu rupiah" in filename_lower or "thousand rupiah" in filename_lower:
        return 'ribu_rupiah'
    elif "rupiah" in filename_lower and "ribu" not in filename_lower:
        return 'rupiah'
    return 'ribu_rupiah'  # Default assumption


def parse_bps_csv(filepath):
    """
    Parse BPS CSV file dan extract revenue categories
    Handles both 'ribu rupiah' (thousands) and 'rupiah' (full) formats
    
    Returns: dict with structure:
    {
        'Provinsi': str,
        'Tahun': int,
        'Unit': str ('ribu_rupiah' or 'rupiah'),
        'Data': {
            'Pajak Daerah': float (in rupiah),
            'Retribusi Daerah': float,
            ...
        }
    }
    """
    try:
        # Read without header dulu untuk explorasi struktur
        df = pd.read_csv(filepath, header=None)
        
        filename = filepath.name
        provinsi = extract_provincia_from_filename(filename)
        tahun = extract_year_from_filename(filename)
        currency_unit = detect_currency_unit(filename)
        
        if not provinsi or not tahun:
            print(f"⚠️  Skipping {filename}: Cannot extract provincia/tahun")
            return None
        
        # Determine multiplier based on unit
        multiplier = 1000 if currency_unit == 'ribu_rupiah' else 1
        unit_label = "ribu rupiah" if currency_unit == 'ribu_rupiah' else "rupiah"
        
        # Flatten data untuk parsing
        data_dict = {
            'Provinsi': provinsi,
            'Tahun': tahun,
            'Unit': currency_unit,
            'Data': {}
        }
        
        # Parse rows untuk mencari kategori revenue
        for idx, row in df.iterrows():
            row_text = str(row[0]) if pd.notna(row[0]) else ""
            row_value = row[1] if len(row) > 1 else None
            
            # Check match dengan kategori
            for category_name, category_info in REVENUE_CATEGORIES.items():
                for keyword in category_info['keywords']:
                    if keyword in row_text:
                        numeric_value = clean_numeric_value(row_value)
                        if numeric_value:
                            # Normalize ke rupiah penuh
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
    """Consolidate semua BPS files"""
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
    """
    Generate monthly data dari annual data
    Asumsi: revenue didistribusi equally per bulan, dengan seasonality 10-15%
    """
    monthly_records = []
    
    for record in annual_data:
        provinsi = record['Provinsi']
        tahun = record['Tahun']
        
        for category_name, annual_value in record['Data'].items():
            # Bagi rata per bulan
            base_monthly = annual_value / 12
            
            for bulan in range(1, 13):
                # Add seasonality: Q1 & Q4 sedikit lebih tinggi
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
    """Main consolidation pipeline"""
    print("\n" + "="*70)
    print("🔧 BPS DATA CONSOLIDATION PIPELINE")
    print("="*70)
    
    # Step 1: Parse semua files
    print("\n1️⃣  STEP 1: Parsing BPS CSV Files")
    print("-" * 70)
    annual_data = consolidate_all_files()
    print(f"\n✅ Successfully parsed {len(annual_data)} files")
    
    # Step 2: Generate monthly data
    print("\n2️⃣  STEP 2: Generating Monthly Data from Annual Format")
    print("-" * 70)
    df_monthly = generate_monthly_data(annual_data)
    print(f"✅ Generated {len(df_monthly)} monthly records")
    print(f"   - Provinces: {df_monthly['Provinsi'].unique().tolist()}")
    print(f"   - Categories: {df_monthly['Jenis_Pendapatan'].unique().tolist()}")
    print(f"   - Years: {sorted(df_monthly['Tahun'].unique())}")
    
    # Step 3: Save processed data
    print("\n3️⃣  STEP 3: Saving Processed Data")
    print("-" * 70)
    
    # Ensure output directory exists
    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    output_file = DATA_PROCESSED_PATH / "revenue_consolidated.csv"
    df_monthly.to_csv(output_file, index=False)
    print(f"✅ Saved to: {output_file}")
    
    # Display sample
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
