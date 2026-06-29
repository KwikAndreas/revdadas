/**
 * Data loading utilities for RevDadas dashboard.
 * Loads pre-computed JSON files from the public directory.
 */

import type {
  Meta,
  HistoricalRecord,
  ForecastRecord,
  AnomalyRecord,
  AccuracyData,
  BusinessData,
  PolicyRecommendation,
} from "./types";

const BASE_URL = "/data";

async function fetchJSON<T>(filename: string): Promise<T> {
  const res = await fetch(`${BASE_URL}/${filename}`);
  if (!res.ok) {
    throw new Error(`Failed to load ${filename}: ${res.statusText}`);
  }
  return res.json();
}

// ─── Cached Data Store ────────────────────────────────────────────
let cachedMeta: Meta | null = null;
let cachedHistorical: HistoricalRecord[] | null = null;
let cachedForecasts: Record<string, ForecastRecord[]> | null = null;
let cachedAnomalies: Record<string, AnomalyRecord[]> | null = null;
let cachedAccuracy: AccuracyData | null = null;
let cachedBusiness: Record<string, BusinessData> | null = null;
let cachedPolicy: Record<string, PolicyRecommendation[]> | null = null;

export async function loadMeta(): Promise<Meta> {
  if (!cachedMeta) cachedMeta = await fetchJSON<Meta>("meta.json");
  return cachedMeta;
}

export async function loadHistorical(): Promise<HistoricalRecord[]> {
  if (!cachedHistorical)
    cachedHistorical = await fetchJSON<HistoricalRecord[]>("historical.json");
  return cachedHistorical;
}

export async function loadForecasts(): Promise<Record<string, ForecastRecord[]>> {
  if (!cachedForecasts)
    cachedForecasts = await fetchJSON<Record<string, ForecastRecord[]>>("forecasts.json");
  return cachedForecasts;
}

export async function loadAnomalies(): Promise<Record<string, AnomalyRecord[]>> {
  if (!cachedAnomalies)
    cachedAnomalies = await fetchJSON<Record<string, AnomalyRecord[]>>("anomalies.json");
  return cachedAnomalies;
}

export async function loadAccuracy(): Promise<AccuracyData> {
  if (!cachedAccuracy)
    cachedAccuracy = await fetchJSON<AccuracyData>("accuracy.json");
  return cachedAccuracy;
}

export async function loadBusiness(): Promise<Record<string, BusinessData>> {
  if (!cachedBusiness)
    cachedBusiness = await fetchJSON<Record<string, BusinessData>>("business.json");
  return cachedBusiness;
}

export async function loadPolicy(): Promise<Record<string, PolicyRecommendation[]>> {
  if (!cachedPolicy)
    cachedPolicy = await fetchJSON<Record<string, PolicyRecommendation[]>>("policy.json");
  return cachedPolicy;
}

/**
 * Load all data at once for the dashboard.
 */
export async function loadAllData() {
  const [meta, historical, forecasts, anomalies, accuracy, business, policy] =
    await Promise.all([
      loadMeta(),
      loadHistorical(),
      loadForecasts(),
      loadAnomalies(),
      loadAccuracy(),
      loadBusiness(),
      loadPolicy(),
    ]);

  return { meta, historical, forecasts, anomalies, accuracy, business, policy };
}

export type AllData = Awaited<ReturnType<typeof loadAllData>>;
