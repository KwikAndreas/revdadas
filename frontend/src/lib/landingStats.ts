/**
 * Statistik landing page, dihitung saat build dari JSON di public/data
 * (hanya dipakai di Server Component). Logika anomali sama dengan KPI dashboard
 * (dashboard/page.tsx → kpiData) agar angka di landing selalu cocok.
 */
import fs from "node:fs";
import path from "node:path";
import { isolateAggregateAnomalies } from "./utils";
import type { AccuracyData, AnomalyRecord, BusinessData, Meta } from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

function readJSON<T>(file: string): T {
  return JSON.parse(fs.readFileSync(path.join(DATA_DIR, file), "utf-8")) as T;
}

const NON_REVENUE = ["Total Pendapatan Daerah", "Total Belanja Daerah", "Belanja Modal"];
const FEATURED_PROVINCES = ["DKI Jakarta", "Bali", "Jawa Barat"];

export interface SectorHighlight {
  provinsi: string;
  sektor: string;
  skor: number;
  label: string;
  alasan: string;
}

export interface LandingStats {
  provinces: number;
  observations: number;
  latestYear: number;
  accuracyPct: number;
  medianWape: number;
  reliableSeries: number;
  totalSeries: number;
  modelName: string;
  anomalyCount: number;
  anomalyDeviation: number;
  sectors: { sektor: string; narasi: string }[];
  sectorHighlights: SectorHighlight[];
}

export function getLandingStats(): LandingStats {
  const meta = readJSON<Meta>("meta.json");
  const accuracy = readJSON<AccuracyData>("accuracy.json");
  const anomalies = readJSON<AnomalyRecord[]>("anomalies.json");
  const business = readJSON<Record<string, BusinessData>>("business.json");

  const latestYear = parseInt(meta.date_range.max.substring(0, 4));
  const yearOf = (r: AnomalyRecord) => r.Tahun || parseInt(r.Tanggal.substring(0, 4));

  // Semua provinsi, semua pendapatan, tahun anggaran terakhir
  const flagged = isolateAggregateAnomalies(
    anomalies.filter(
      (r) =>
        r.Anomaly &&
        yearOf(r) === latestYear &&
        !NON_REVENUE.includes(r.Jenis_Pendapatan) &&
        !r.Jenis_Pendapatan.includes("Belanja")
    )
  );
  const anomalyDeviation = flagged.reduce((s, r) => s + Math.abs(r.Deviasi ?? 0), 0);


  const biz = business["12"] ?? Object.values(business)[0];
  const firstProv = Object.values(biz?.scored ?? {})[0] ?? [];
  const sectors = firstProv.map((s) => ({ sektor: s.sektor, narasi: s.narasi }));
  const sectorHighlights: SectorHighlight[] = (biz?.top_recommendations ?? [])
    .filter((r) => FEATURED_PROVINCES.includes(r.provinsi))
    .sort((a, b) => FEATURED_PROVINCES.indexOf(a.provinsi) - FEATURED_PROVINCES.indexOf(b.provinsi))
    .map((r) => ({
      provinsi: r.provinsi,
      sektor: r.top_sektor,
      skor: r.top_skor,
      label: r.top_label,
      alasan: r.top_alasan,
    }));

  const overall = accuracy.overall;
  return {
    provinces: meta.provinces.length,
    observations: meta.total_rows,
    latestYear,
    accuracyPct: overall?.akurasi ?? 0,
    medianWape: overall?.median_wape ?? 0,
    reliableSeries: overall?.n_reliable ?? 0,
    totalSeries: overall?.n_series ?? 0,
    modelName: overall?.model_name ?? meta.active_model_name ?? "Profil Serapan Berjangkar",
    anomalyCount: flagged.length,
    anomalyDeviation,
    sectors,
    sectorHighlights,
  };
}

/** Format angka gaya Indonesia (koma desimal, titik ribuan). */
export function idNumber(value: number, digits = 0): string {
  return value.toLocaleString("id-ID", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function idRupiahShort(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e12) return `Rp ${idNumber(value / 1e12, 1)} T`;
  if (abs >= 1e9) return `Rp ${idNumber(value / 1e9, 1)} M`;
  if (abs >= 1e6) return `Rp ${idNumber(value / 1e6, 1)} Jt`;
  return `Rp ${idNumber(value)}`;
}
