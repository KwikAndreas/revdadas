"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { loadAllData, type AllData } from "@/lib/data";
import type {
  DashboardFilters,
  HistoricalRecord,
  ForecastRecord,
  AnomalyRecord,
  PolicyRecommendation,
} from "@/lib/types";
import {
  formatCurrency,
  DEFAULT_PROVINCES,
  getPriorityColors,
} from "@/lib/utils";
import Sidebar from "@/components/layout/Sidebar";
import Header from "@/components/layout/Header";
import KPICards from "@/components/dashboard/KPICards";
import ImpactCalculator from "@/components/dashboard/ImpactCalculator";
import AIInsights from "@/components/dashboard/AIInsights";
import RevenueChart from "@/components/charts/RevenueChart";
import ProportionChart from "@/components/charts/ProportionChart";
import DataTabs from "@/components/tables/DataTabs";
import { FileText, Sparkles, Map, Database, AlertCircle, CheckCircle, XCircle } from "lucide-react";
import dynamic from "next/dynamic";

// Dynamic import for map (requires browser APIs)
const HeatmapIndonesia = dynamic(
  () => import("@/components/maps/HeatmapIndonesia"),
  { ssr: false, loading: () => <div style={{ height: 400, background: "#f1f5f9", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", color: "#94a3b8" }}>Memuat peta...</div> }
);

export default function DashboardPage() {
  const [data, setData] = useState<AllData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showRecs, setShowRecs] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [filters, setFilters] = useState<DashboardFilters>({
    selectedProvinces: [],
    selectedTaxType: "Semua Pendapatan",
    forecastMonths: 9,
    fraudPreventionPct: 5,
  });

  // Load data on mount
  useEffect(() => {
    loadAllData()
      .then((d) => {
        setData(d);
        // Set default provinces from meta (intersect with DEFAULT_PROVINCES)
        const available = d.meta.provinces;
        const defaults = DEFAULT_PROVINCES.filter((p) => available.includes(p));
        setFilters((prev) => ({
          ...prev,
          selectedProvinces: defaults.length > 0 ? defaults : available.slice(0, 3),
        }));
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  // ─── Derived Data ────────────────────────────────────────────
  const filteredHistorical = useMemo<HistoricalRecord[]>(() => {
    if (!data) return [];
    const { selectedProvinces, selectedTaxType } = filters;
    return data.historical.filter((r) => {
      const provMatch = selectedProvinces.includes(r.Provinsi);
      const taxMatch =
        selectedTaxType === "Semua Pendapatan" ||
        r.Jenis_Pendapatan === selectedTaxType;
      return provMatch && taxMatch;
    });
  }, [data, filters]);

  const filteredForecast = useMemo<ForecastRecord[]>(() => {
    if (!data) return [];
    const period = filters.forecastMonths;
    const forecasts = data.forecasts[String(period)] || [];
    const { selectedProvinces, selectedTaxType } = filters;
    return forecasts.filter((r) => {
      const provMatch = selectedProvinces.includes(r.Provinsi);
      const taxMatch =
        selectedTaxType === "Semua Pendapatan" ||
        r.Jenis_Pendapatan === selectedTaxType;
      return provMatch && taxMatch;
    });
  }, [data, filters]);

  const filteredAnomalies = useMemo<AnomalyRecord[]>(() => {
    if (!data) return [];
    const period = filters.forecastMonths;
    const anomalies = data.anomalies[String(period)] || [];
    const { selectedProvinces, selectedTaxType } = filters;
    return anomalies.filter((r: any) => {
      const provMatch = selectedProvinces.includes(r.Provinsi);
      const taxMatch =
        selectedTaxType === "Semua Pendapatan" ||
        r.Jenis_Pendapatan === selectedTaxType;
      return provMatch && taxMatch;
    });
  }, [data, filters]);

  const bizData = useMemo(() => {
    if (!data) return { scored: {}, top_recommendations: [] };
    const period = filters.forecastMonths;
    const bizPeriodData = data.business[String(period)] || { scored: {}, top_recommendations: [] };

    const { selectedProvinces } = filters;
    const provSet = new Set(selectedProvinces);

    // Filter by selected provinces
    const scoredSubset: Record<string, any[]> = {};
    for (const prov of selectedProvinces) {
      if (bizPeriodData.scored[prov]) {
        scoredSubset[prov] = bizPeriodData.scored[prov];
      }
    }

    return {
      scored: scoredSubset,
      top_recommendations: bizPeriodData.top_recommendations.filter(
        (r: any) => provSet.has(r.Provinsi)
      ),
    };
  }, [data, filters]);

  const policyRecs = useMemo<PolicyRecommendation[]>(() => {
    if (!data) return [];
    // Snap to nearest pre-computed fraud prevention percentage
    const pctKeys = Object.keys(data.policy).map(Number).sort((a, b) => a - b);
    const nearest = pctKeys.reduce((prev, curr) =>
      Math.abs(curr - filters.fraudPreventionPct) <
        Math.abs(prev - filters.fraudPreventionPct)
        ? curr
        : prev
    );
    return data.policy[String(nearest)] || [];
  }, [data, filters.fraudPreventionPct]);

  // ─── KPI Calculations ───────────────────────────────────────
  const kpiData = useMemo(() => {
    const totalRevenue = filteredHistorical.reduce(
      (sum, r) => sum + r.Realisasi,
      0
    );
    const forecastTotal = filteredForecast.reduce(
      (sum, r) => sum + r.Prediksi,
      0
    );

    const anomaliesOnly = filteredAnomalies.filter((r) => r.Anomaly);
    const anomalyCount = anomaliesOnly.length;
    const potentialLoss = anomaliesOnly.reduce(
      (sum, r) => sum + r.Realisasi,
      0
    );
    const anomalyPct =
      totalRevenue > 0 ? (potentialLoss / totalRevenue) * 100 : 0;

    const savedRevenue = potentialLoss * (filters.fraudPreventionPct / 100);
    const simulatedDAU = totalRevenue * 1.5;
    const newPAD = totalRevenue + savedRevenue;
    const kemandirianFiskal = newPAD > 0 ? (newPAD / (newPAD + simulatedDAU)) * 100 : 0;

    return {
      totalRevenue,
      forecastTotal,
      anomalyCount,
      potentialLoss,
      anomalyPct,
      kemandirianFiskal,
      anomalies: anomaliesOnly,
    };
  }, [filteredHistorical, filteredForecast, filteredAnomalies, filters.fraudPreventionPct]);

  const accuracyText = useMemo(() => {
    if (!data || data.accuracy.by_series.length === 0) return "Predicted by AI Model";

    // Filter the series accuracy based on current selected provinces and tax types
    const { selectedProvinces, selectedTaxType } = filters;
    const filteredSeries = data.accuracy.by_series.filter((r) => {
      const provMatch = selectedProvinces.includes(r.Provinsi);
      const taxMatch = selectedTaxType === "Semua Pendapatan" || r.Jenis_Pendapatan === selectedTaxType;
      return provMatch && taxMatch;
    });

    if (filteredSeries.length === 0) return "Predicted by AI Model";

    // Calculate the average accuracy of the visible series to mimic Streamlit's dynamic calculation
    const totalAkurasi = filteredSeries.reduce((sum, r) => sum + r.Akurasi, 0);
    const avgAkurasi = totalAkurasi / filteredSeries.length;

    return `Akurasi backtest ${avgAkurasi.toFixed(0)}%`;
  }, [data, filters]);

  // ─── Insight Text ────────────────────────────────────────────
  const insightText = useMemo(() => {
    const anomaliesOnly = filteredAnomalies.filter((r) => r.Anomaly);
    if (anomaliesOnly.length === 0) {
      return "Sistem berjalan optimal. Tidak ada anomali signifikan.";
    }
    const top = anomaliesOnly[0];
    return `Terdeteksi diskrepansi data dan anomali pada pencatatan <b>${top.Jenis_Pendapatan}</b> di wilayah <b>${top.Provinsi}</b>.`;
  }, [filteredAnomalies]);

  // ─── Handlers ────────────────────────────────────────────────
  const handleFilterChange = useCallback(
    (partial: Partial<DashboardFilters>) => {
      setFilters((prev) => ({ ...prev, ...partial }));
    },
    []
  );

  const handleExportPDF = useCallback(() => {
    import("@/lib/pdf").then(({ generatePDF }) => {
      generatePDF(kpiData, policyRecs, filters);
    });
  }, [kpiData, policyRecs, filters]);

  // ─── Loading / Error States ──────────────────────────────────
  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        <p className="loading-text" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
          <Database size={16} /> Memuat data RevDadas...
        </p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="loading-screen">
        <p style={{ color: "#dc2626", fontSize: 16, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
          <AlertCircle size={20} /> Gagal memuat data: {error}
        </p>
        <p style={{ color: "#64748b", fontSize: 13 }}>
          Pastikan data sudah di-generate dengan menjalankan{" "}
          <code>python scripts/precompute.py</code>
        </p>
      </div>
    );
  }

  return (
    <div className="app-layout">
      {/* Mobile Sidebar Overlay */}
      <div 
        className={`sidebar-overlay ${sidebarOpen ? "open" : ""}`} 
        onClick={() => setSidebarOpen(false)} 
      />

      {/* ── Sidebar ────────────────────────────────────────── */}
      <Sidebar
        meta={data.meta}
        filters={filters}
        onFilterChange={handleFilterChange}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* ── Main Content ───────────────────────────────────── */}
      <main className="main-content">
        {/* Header */}
        <Header 
          onExportPDF={handleExportPDF} 
          onMenuClick={() => setSidebarOpen(true)}
        />

        {/* KPI Cards */}
        <KPICards
          totalRevenue={kpiData.totalRevenue}
          forecastTotal={kpiData.forecastTotal}
          anomalyPct={kpiData.anomalyPct}
          anomalyCount={kpiData.anomalyCount}
          potentialLoss={kpiData.potentialLoss}
          forecastMonths={filters.forecastMonths}
          accuracyText={accuracyText}
          kemandirianFiskal={kpiData.kemandirianFiskal}
          anomalies={kpiData.anomalies}
        />

        {/* Middle Row: Map + Impact Calculator */}
        <div className="middle-row">
          <div>
            <div className="map-header">
              <h3 className="section-title" style={{ margin: 0 }}>
                Heatmap Potensi Revenue & Risiko
              </h3>
              <div className="map-legend">
                <span className="map-legend-item map-legend-item--optimal">
                  OPTIMAL
                </span>
                <span className="map-legend-item map-legend-item--moderate">
                  MODERAT
                </span>
                <span className="map-legend-item map-legend-item--critical">
                  KRITIS
                </span>
              </div>
            </div>
            <div className="map-container">
              <HeatmapIndonesia
                historical={filteredHistorical}
                forecast={filteredForecast}
                anomalies={filteredAnomalies}
                selectedProvinces={filters.selectedProvinces}
              />
            </div>
          </div>

          <div>
            <ImpactCalculator
              potentialLoss={kpiData.potentialLoss}
              fraudPreventionPct={filters.fraudPreventionPct}
              onShowRecs={() => setShowRecs(true)}
            />
            <AIInsights insightText={insightText} />
          </div>
        </div>

        {/* Policy Recommendations (Moved up for UX Best Practice) */}
        {showRecs && (
          <div className="animate-fade-in-up" style={{ marginBottom: 32 }}>
            <h3 className="section-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Sparkles size={18} /> Rekomendasi Berbasis Algoritma
            </h3>
            <div style={{ display: "grid", gap: 12 }}>
              {policyRecs.map((rec, i) => {
                const colors = getPriorityColors(rec.prioritas);
                return (
                  <div key={i} className="insight-card" style={{ padding: "16px 20px" }}>
                    <div className="policy-header" style={{ marginBottom: 12 }}>
                      <span className="policy-title" style={{ fontSize: "1.1rem", fontWeight: 600 }}>{rec.judul}</span>
                      <span
                        className="policy-badge"
                        style={{
                          background: colors.bg,
                          color: colors.text,
                        }}
                      >
                        {rec.prioritas}
                      </span>
                    </div>
                    
                    <p className="policy-detail" style={{ marginBottom: 16 }}>{rec.detail}</p>
                    
                    {rec.kebijakan_existing && (
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                        <div style={{ background: "#f8fafc", padding: 12, borderRadius: 8, borderLeft: "3px solid #94a3b8" }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 4 }}>BASELINE (EXISTING)</div>
                          <div style={{ fontSize: 13, color: "#334155" }}>{rec.kebijakan_existing}</div>
                        </div>
                        <div style={{ background: "#f0fdfa", padding: 12, borderRadius: 8, borderLeft: "3px solid #14b8a6" }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: "#0d9488", marginBottom: 4 }}>REKOMENDASI AI</div>
                          <div style={{ fontSize: 13, color: "#0f766e" }}>{rec.perbandingan}</div>
                        </div>
                      </div>
                    )}
                    
                    {(rec.kelebihan || rec.kekurangan) && (
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16, fontSize: 13 }}>
                        <div>
                          <div style={{ fontWeight: 600, color: "#16a34a", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                            <CheckCircle size={14} /> Kelebihan
                          </div>
                          <ul style={{ paddingLeft: 20, color: "#334155", margin: 0, display: "flex", flexDirection: "column", gap: 4 }}>
                            {rec.kelebihan?.map((k, idx) => <li key={idx}>{k}</li>)}
                          </ul>
                        </div>
                        <div>
                          <div style={{ fontWeight: 600, color: "#dc2626", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                            <XCircle size={14} /> Risiko / Kekurangan
                          </div>
                          <ul style={{ paddingLeft: 20, color: "#334155", margin: 0, display: "flex", flexDirection: "column", gap: 4 }}>
                            {rec.kekurangan?.map((k, idx) => <li key={idx}>{k}</li>)}
                          </ul>
                        </div>
                      </div>
                    )}
                    
                    {(rec.kaitan_bisnis || rec.indikator_dampak || rec.justifikasi) && (
                      <div style={{ borderTop: "1px solid #e2e8f0", paddingTop: 12, marginTop: 12 }}>
                        {rec.justifikasi && (
                          <div style={{ marginBottom: 8, fontSize: 13 }}>
                            <span style={{ fontWeight: 600, color: "#475569", marginRight: 8 }}>Justifikasi:</span>
                            <span style={{ color: "#334155" }}>{rec.justifikasi}</span>
                          </div>
                        )}
                        <div style={{ display: "flex", gap: 16, fontSize: 12 }}>
                          {rec.indikator_dampak && (
                            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                              <span style={{ fontWeight: 600, color: "#64748b" }}>Indikator:</span>
                              <span style={{ color: "#0f172a" }}>{rec.indikator_dampak}</span>
                            </div>
                          )}
                          {rec.kaitan_bisnis && (
                            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                              <span style={{ fontWeight: 600, color: "#64748b" }}>Dampak:</span>
                              <span style={{ color: "#0f172a" }}>{rec.kaitan_bisnis}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <p className="table-caption" style={{ marginTop: 12 }}>
              Rekomendasi bersifat indikatif sebagai bahan diskusi kebijakan, bukan keputusan final.
            </p>
          </div>
        )}

        {/* Charts Row */}
        <div style={{ display: "flex", flexDirection: "column", gap: 24, marginBottom: 32 }}>
          <div className="chart-card animate-fade-in-up">
            <h3 className="section-title">
              Historical Revenue vs Forecast (Ensemble AI)
            </h3>
            <RevenueChart
              historical={filteredHistorical}
              forecast={filteredForecast}
            />
          </div>
          <div className="chart-card animate-fade-in-up" style={{ animationDelay: "100ms" }}>
            <h3 className="section-title">Proporsi Sumber Pendapatan</h3>
            <ProportionChart historical={filteredHistorical} />
          </div>
        </div>

        {/* Separator */}
        <hr className="separator" />

        {/* Detailed Data Logs */}
        <h3 className="section-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <FileText size={18} /> Detailed Data Logs
        </h3>
        <DataTabs
          forecast={filteredForecast}
          anomalies={filteredAnomalies}
          accuracy={data.accuracy}
          business={bizData}
          historical={filteredHistorical}
          selectedProvinces={filters.selectedProvinces}
          forecastMonths={filters.forecastMonths}
        />


      </main>
    </div>
  );
}
