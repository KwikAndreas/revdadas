/**
 * RevDadas Type Definitions
 * Matches the JSON structure from the pre-compute script.
 */

// ─── Historical Data ──────────────────────────────────────────────
export interface HistoricalRecord {
  Tanggal: string;
  Provinsi: string;
  Jenis_Pendapatan: string;
  Realisasi: number;
  Anggaran?: number;
  Persentase?: number;
  Bulan?: number;
  Tahun?: number;
  Kuartal?: number;
  Revenue_Growth?: number;
  Revenue_MA3?: number;
}

// ─── Forecast Data ────────────────────────────────────────────────
export interface ForecastRecord {
  Tanggal: string;
  Provinsi: string;
  Jenis_Pendapatan: string;
  Prediksi: number;
  Batas_Bawah: number;
  Batas_Atas: number;
  Metode?: string;
}

// ─── Anomaly Data ─────────────────────────────────────────────────
export interface AnomalyRecord {
  Tanggal: string;
  Provinsi: string;
  Jenis_Pendapatan: string;
  Realisasi: number;
  Anomaly: boolean;
  Tahun?: number;
  Bulan?: number;
  Anomaly_Score?: number;
  Severity?: string;
  Alasan?: string;
  Jenis_Fraud?: string;
  Deviasi?: number;              // selisih vs expected (moving average)
  Materiality?: string;          // 'Material' | 'Minor' | '-'
  Bulan_Nol_Sebelumnya?: number; // consecutive zero months before this row
  Gap_Data?: boolean;
}

// ─── Accuracy Data ────────────────────────────────────────────────
export interface AccuracyOverall {
  akurasi: number;
  n_reliable: number;
  n_series: number;
  median_wape: number;
}

export interface AccuracySeries {
  Provinsi: string;
  Jenis_Pendapatan: string;
  Akurasi: number;
  WAPE: number | null;
  sMAPE: number | null;
}

export interface AccuracyData {
  overall: AccuracyOverall | null;
  by_series: AccuracySeries[];
}

// ─── Business Recommendations ─────────────────────────────────────
export interface SectorScore {
  sektor: string;
  icon: string;
  skor: number;
  label: string;
  narasi: string;
  alasan: string;
}

export interface TopRecommendation {
  provinsi: string;
  top_sektor: string;
  top_icon: string;
  top_skor: number;
  top_label: string;
  top_alasan: string;
  hindari_sektor: string;
  hindari_skor: number;
  semua: SectorScore[];
}

export interface BusinessData {
  scored: Record<string, SectorScore[]>;
  top_recommendations: TopRecommendation[];
}

// ─── Policy Recommendations ───────────────────────────────────────
export interface PolicyRecommendation {
  judul: string;
  prioritas: "Tinggi" | "Sedang" | "Rendah";
  detail: string;
  kebijakan_existing?: string;
  kelebihan?: string[];
  kekurangan?: string[];
  indikator_dampak?: string;
  kaitan_bisnis?: string;
  perbandingan?: string;
  justifikasi?: string;
}

// ─── Metadata ─────────────────────────────────────────────────────
export interface Meta {
  generated_at: string;
  provinces: string[];
  tax_types: string[];
  forecast_periods: number[];
  date_range: {
    min: string;
    max: string;
  };
  total_rows: number;
}

// ─── Dashboard State ──────────────────────────────────────────────
export interface DashboardFilters {
  selectedProvinces: string[];
  selectedTaxType: string;
  forecastMonths: number;
  fraudPreventionPct: number;
  selectedYear: number;
}

// ─── Province Coordinates ─────────────────────────────────────────
export interface ProvinceCoord {
  name: string;
  lat: number;
  lng: number;
}
