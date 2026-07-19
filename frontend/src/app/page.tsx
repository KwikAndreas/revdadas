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
  toBillions,
} from "@/lib/utils";
import Sidebar from "@/components/layout/Sidebar";
import Header from "@/components/layout/Header";
import KPICards from "@/components/dashboard/KPICards";
import ImpactCalculator from "@/components/dashboard/ImpactCalculator";
import AIInsights from "@/components/dashboard/AIInsights";
import RevenueChart from "@/components/charts/RevenueChart";
import ProportionChart from "@/components/charts/ProportionChart";
import DataTabs from "@/components/tables/DataTabs";
import { FileText, Sparkles, Map as MapIcon, Database, AlertCircle, CheckCircle, XCircle } from "lucide-react";
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

  const historicalForProportion = useMemo<HistoricalRecord[]>(() => {
    if (!data) return [];
    return data.historical.filter((r) => filters.selectedProvinces.includes(r.Provinsi));
  }, [data, filters.selectedProvinces]);


  const filteredForecast = useMemo<ForecastRecord[]>(() => {
    if (!data) return [];
    const period = filters.forecastMonths;
    const forecasts = data.forecasts[String(period)] || [];
    const { selectedProvinces, selectedTaxType } = filters;
    return forecasts.filter((r) => {
      if (r.Jenis_Pendapatan.toLowerCase().includes("belanja")) return false;
      const provMatch = selectedProvinces.includes(r.Provinsi);
      const taxMatch =
        selectedTaxType === "Semua Pendapatan" ||
        r.Jenis_Pendapatan === selectedTaxType;
      return provMatch && taxMatch;
    });
  }, [data, filters]);

  const filteredAnomalies = useMemo<AnomalyRecord[]>(() => {
    if (!data) return [];
    // Anomaly data is now a flat array (no longer keyed by forecast period)
    const anomalies = data.anomalies;
    if (!Array.isArray(anomalies)) return []; // Guard against old cached object structure
    
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
    if (!data?.policy) return [];
    
    const firstProv = Object.keys(data.policy)[0];
    if (!firstProv || !data.policy[firstProv]) return [];
    
    const pctKeys = Object.keys(data.policy[firstProv]).map(Number).sort((a, b) => a - b);
    if (pctKeys.length === 0) return [];
    
    const nearest = pctKeys.reduce((prev, curr) =>
      Math.abs(curr - filters.fraudPreventionPct) < Math.abs(prev - filters.fraudPreventionPct) ? curr : prev
    );
    
    const nearestStr = String(nearest);
    let allRecs: PolicyRecommendation[] = [];
    
    for (const prov of filters.selectedProvinces) {
      if (data.policy[prov] && data.policy[prov][nearestStr]) {
        allRecs = allRecs.concat(data.policy[prov][nearestStr]);
      }
    }
    
    const order: Record<string, number> = { "Tinggi": 0, "Sedang": 1, "Rendah": 2 };
    allRecs.sort((a, b) => (order[a.prioritas] ?? 9) - (order[b.prioritas] ?? 9));
    
    return allRecs;
  }, [data?.policy, filters.fraudPreventionPct, filters.selectedProvinces]);

  // ─── KPI Calculations ───────────────────────────────────────
  const kpiData = useMemo(() => {
    const revenueToSum = filters.selectedTaxType === "Semua Pendapatan" ? "Total Pendapatan Daerah" : filters.selectedTaxType;
    const filteredRev = filteredHistorical.filter(r => r.Jenis_Pendapatan === revenueToSum);
    
    // Get latest year for Current Year KPIs
    const lastYear = filteredRev.length > 0 ? Math.max(...filteredRev.map(r => r.Tahun || parseInt(r.Tanggal.substring(0, 4)))) : 2025;
    const thisYearRecords = filteredRev.filter(r => (r.Tahun || parseInt(r.Tanggal.substring(0, 4))) === lastYear);
    const totalRevenue = thisYearRecords.reduce((sum, r) => sum + r.Realisasi, 0);

    const forecastTotal = filteredForecast.filter(r => r.Jenis_Pendapatan === revenueToSum).reduce(
      (sum, r) => sum + r.Prediksi,
      0
    );

    // Get target percentage if Anggaran is available
    let targetPercentage: number | undefined = undefined;
    if (thisYearRecords.length > 0) {
      const uniqueAnggaranMap = new Map<string, number>();
      thisYearRecords.forEach(r => {
        const key = `${r.Provinsi}-${r.Jenis_Pendapatan}`;
        if (!uniqueAnggaranMap.has(key) && r.Anggaran) {
          uniqueAnggaranMap.set(key, r.Anggaran);
        }
      });
      
      const totalAnggaran = Array.from(uniqueAnggaranMap.values()).reduce((a, b) => a + b, 0);
      const thisYearRealisasi = totalRevenue; // already calculated
      
      if (totalAnggaran > 0) {
        targetPercentage = (thisYearRealisasi / totalAnggaran) * 100;
      }
    }
    
    const AGGREGATE_TYPES = ["Total Pendapatan Daerah", "Total Belanja Daerah", "Belanja Modal"];
    const anomaliesOnly = filteredAnomalies.filter((r) => {
      if (!r.Anomaly) return false;
      if (filters.selectedTaxType === "Semua Pendapatan") {
        return !AGGREGATE_TYPES.includes(r.Jenis_Pendapatan) && !r.Jenis_Pendapatan.includes("Belanja");
      }
      return r.Jenis_Pendapatan === filters.selectedTaxType;
    });
    // Use ALL historical anomalies for the KPI to match the table
    const targetAnomalies = anomaliesOnly;
    const anomalyCount = targetAnomalies.length;
    // Use Deviasi (difference from expected) instead of total Realisasi
    const potentialLoss = targetAnomalies.reduce(
      (sum, r) => sum + Math.abs(r.Deviasi ?? 0),
      0
    );
    const anomalyPct =
      totalRevenue > 0 ? (potentialLoss / totalRevenue) * 100 : 0;

    const savedRevenue = potentialLoss * (filters.fraudPreventionPct / 100);
    
    // Kemandirian Fiskal = (PAD / Total Pendapatan Daerah) * 100
    const thisYearProportionRecords = historicalForProportion.filter(r => (r.Tahun || parseInt(r.Tanggal.substring(0, 4))) === lastYear);
    
    const totalPAD = thisYearProportionRecords
      .filter(r => r.Jenis_Pendapatan.toLowerCase().includes("pendapatan asli daerah"))
      .reduce((sum, r) => sum + r.Realisasi, 0);
      
    const actualTotalRevenue = thisYearProportionRecords
      .filter(r => r.Jenis_Pendapatan === "Total Pendapatan Daerah")
      .reduce((sum, r) => sum + r.Realisasi, 0);
    
    // Asumsikan fraud prevention menambah efisiensi PAD
    const newPAD = totalPAD + savedRevenue;
    const newTotalRevenue = (actualTotalRevenue > 0 ? actualTotalRevenue : totalRevenue) + savedRevenue;
    
    const kemandirianFiskal = newTotalRevenue > 0 ? (newPAD / newTotalRevenue) * 100 : 0;

    return {
      totalRevenue,
      targetPercentage,
      forecastTotal,
      anomalyCount,
      potentialLoss,
      anomalyPct,
      kemandirianFiskal,
      savedRevenue,
      anomalies: targetAnomalies,
      thisYearProportionRecords,
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

    // Calculate the median accuracy of the visible series
    const akurasiArray = filteredSeries.map(r => r.Akurasi).sort((a, b) => a - b);
    const mid = Math.floor(akurasiArray.length / 2);
    const medianAkurasi = akurasiArray.length % 2 !== 0 ? akurasiArray[mid] : (akurasiArray[mid - 1] + akurasiArray[mid]) / 2;

    return `Akurasi Model (Median) ${medianAkurasi.toFixed(0)}%`;
  }, [data, filters]);

  useEffect(() => {
    if (process.env.NODE_ENV !== 'development' || !data) return;
    
    let output = "================ DEBUG INFO ================\n";
    
    // 1. SUMMARY KPI
    output += "[SUMMARY] KINERJA UTAMA (KPI)\n";
    output += `Total Revenue                 : ${formatCurrency(kpiData.totalRevenue)}\n`;
    output += `Forecast 9 Bulan              : ${formatCurrency(kpiData.forecastTotal)}\n`;
    output += `Risiko Fraud/Anomali          : ${kpiData.anomalyPct.toFixed(1)}% (${kpiData.anomalyCount} records deteksi)\n`;
    output += `Revenue Loss Deteksi          : ${formatCurrency(kpiData.potentialLoss)}\n`;
    output += `Kemandirian Fiskal            : ${kpiData.kemandirianFiskal.toFixed(1)}%\n`;
    output += `Potensi Penyelamatan Arus Kas : ${formatCurrency(kpiData.savedRevenue)}\n\n`;

    // 2. PROPORSI SUMBER PENDAPATAN
    output += "[MODULE] PROPORSI SUMBER PENDAPATAN\n";
    const dataMap = new Map<string, number>();
    kpiData.thisYearProportionRecords.forEach((r) => {
      const name = r.Jenis_Pendapatan.toLowerCase();
      const val = toBillions(r.Realisasi);
      if (name === "pendapatan asli daerah (pad)") {
        dataMap.set("Pendapatan Asli Daerah (PAD)", (dataMap.get("Pendapatan Asli Daerah (PAD)") || 0) + val);
      } else if (name === "pendapatan transfer pemerintah pusat" || name === "pendapatan transfer antar daerah" || name === "tkdd") {
        dataMap.set("Pendapatan Transfer", (dataMap.get("Pendapatan Transfer") || 0) + val);
      } else if (name === "lain-lain pendapatan sesuai dengan ketentuan peraturan perundang-undangan" || name === "pendapatan hibah") {
        dataMap.set("Pendapatan Lainnya", (dataMap.get("Pendapatan Lainnya") || 0) + val);
      }
    });
    
    const propData = Array.from(dataMap.entries())
      .sort((a, b) => b[1] - a[1]);
    const totalProp = propData.reduce((acc, curr) => acc + curr[1], 0);

    propData.slice(0, 5).forEach((p, i) => {
      const pct = totalProp > 0 ? (p[1] / totalProp) * 100 : 0;
      const barLength = Math.floor(pct / 3);
      const bar = "[".padEnd(barLength + 1, "|").padEnd(21, " ") + "]";
      const label = p[0].substring(0, 27).padEnd(27, " ");
      const valStr = `Rp ${(p[1]/1000).toFixed(1)} T`.padStart(11, " ");
      output += `${i + 1}. ${label} : ${valStr} ${bar} ${pct.toFixed(1)}%\n`;
    });
    output += "\n";

    // 3. FORECAST DATA (Sample)
    output += "[MODULE] DETAIL DATA LOGS\n\n";
    output += "> TAB: FORECAST DATA (Sample Top 3)\n";
    output += "Tanggal    | Provinsi         | Jenis Pendapatan             | Prediksi       | Metode\n";
    output += "".padEnd(85, "-") + "\n";
    filteredForecast.slice(0, 3).forEach(r => {
      const d = r.Tanggal.split("T")[0];
      const prov = r.Provinsi.substring(0, 16).padEnd(16, " ");
      const jenis = r.Jenis_Pendapatan.substring(0, 28).padEnd(28, " ");
      const val = `Rp ${(toBillions(r.Prediksi)/1000).toFixed(1)} T`.padEnd(14, " ");
      output += `${d} | ${prov} | ${jenis} | ${val} | ${r.Metode || "-"}\n`;
    });
    output += "\n";

    // 4. ANOMALIES (Sample)
    output += "> TAB: ANOMALIES (Sample Top 3)\n";
    output += "Tanggal    | Provinsi         | Jenis Pendapatan             | Realisasi      | Severity\n";
    output += "".padEnd(85, "-") + "\n";
    const topAnomalies = filteredAnomalies.filter(a => a.Anomaly).sort((a,b) => b.Realisasi - a.Realisasi).slice(0, 3);
    if (topAnomalies.length > 0) {
      topAnomalies.forEach(r => {
        const d = r.Tanggal.split("T")[0];
        const prov = r.Provinsi.substring(0, 16).padEnd(16, " ");
        const jenis = r.Jenis_Pendapatan.substring(0, 28).padEnd(28, " ");
        const val = `Rp ${(toBillions(r.Realisasi)/1000).toFixed(1)} T`.padEnd(14, " ");
        output += `${d} | ${prov} | ${jenis} | ${val} | ${r.Severity}\n`;
      });
    } else {
      output += "Tidak ada deteksi anomali pada subset ini.\n";
    }
    output += "\n";

    // 5. ACCURACY
    output += "> TAB: AKURASI MODEL (WAPE) (Sample Top 3)\n";
    output += "Provinsi         | Jenis Pendapatan             | Akurasi | WAPE    | Keandalan\n";
    output += "".padEnd(85, "-") + "\n";
    const accuracySeries = data.accuracy?.by_series?.filter(a => 
      filters.selectedProvinces.includes(a.Provinsi) && 
      (filters.selectedTaxType === "Semua Pendapatan" || a.Jenis_Pendapatan === filters.selectedTaxType)
    ) || [];
    
    accuracySeries.slice(0, 3).forEach(r => {
      const prov = r.Provinsi.substring(0, 16).padEnd(16, " ");
      const jenis = r.Jenis_Pendapatan.substring(0, 28).padEnd(28, " ");
      const ak = `${r.Akurasi.toFixed(1)}%`.padEnd(7, " ");
      const wa = `${r.WAPE ? r.WAPE.toFixed(1) : "-"}%`.padEnd(7, " ");
      let ke = "🔴 Lemah";
      if (r.WAPE !== null) {
        if (r.WAPE < 30) ke = "🟢 Andal";
        else if (r.WAPE < 50) ke = "🟡 Cukup";
      }
      output += `${prov} | ${jenis} | ${ak} | ${wa} | ${ke}\n`;
    });
    output += "\n";

    output += "================ Pengaturan ================\n";
    output += `Jenis Pendapatan : ${filters.selectedTaxType}\n`;
    output += `Provinsi Target  : ${filters.selectedProvinces.length > 3 ? filters.selectedProvinces.slice(0,3).join(", ") + " dll" : filters.selectedProvinces.join(", ")}\n`;
    output += `Periode Prediksi : ${filters.forecastMonths} Bulan\n`;
    output += `Pencegahan Fraud : ${filters.fraudPreventionPct}%\n`;
    output += "============================================";
    
    console.log(output);
  }, [kpiData, filters, historicalForProportion, filteredForecast, filteredAnomalies, data]);

  // ─── Insight Text ────────────────────────────────────────────
  const insightText = useMemo(() => {
    const anomaliesOnly = kpiData.anomalies;
    if (anomaliesOnly.length === 0) {
      return "Sistem berjalan optimal. Tidak ada anomali signifikan.";
    }
    const top = anomaliesOnly[0];
    return `Terdeteksi diskrepansi data dan anomali pada pencatatan <b>${top.Jenis_Pendapatan}</b> di wilayah <b>${top.Provinsi}</b>.`;
  }, [kpiData.anomalies]);

  // ─── Handlers ────────────────────────────────────────────────
  const handleFilterChange = useCallback(
    (partial: Partial<DashboardFilters>) => {
      setFilters((prev) => ({ ...prev, ...partial }));
    },
    []
  );

  const handleExportPDF = useCallback(() => {
    import("@/lib/pdf").then(({ generatePDF }) => {
      generatePDF(kpiData, policyRecs, filters, bizData);
    });
  }, [kpiData, policyRecs, filters, bizData]);

  // ─── Debug Logging ──────────────────────────────────────────
  useEffect(() => {
    if (process.env.NODE_ENV === "development" && !loading && data) {
      console.log(
        "================ DEBUG INFO ================\n" +
        `Total Revenue                 : Rp ${(kpiData.totalRevenue / 1e12).toFixed(1)} T\n` +
        `Forecast ${filters.forecastMonths} Bulan              : Rp ${(kpiData.forecastTotal / 1e12).toFixed(1)} T\n` +
        `Risiko Fraud/Anomali          : ${kpiData.anomalyPct.toFixed(1)}% (${kpiData.anomalyCount} records deteksi)\n` +
        `Revenue Loss Deteksi          : Rp ${(kpiData.potentialLoss / 1e12).toFixed(1)} T\n` +
        `Kemandirian Fiskal            : ${kpiData.kemandirianFiskal.toFixed(1)}%\n` +
        `Potensi Penyelamatan Arus Kas : Rp ${(kpiData.savedRevenue / 1e12).toFixed(1)} T\n` +
        "--- Pengaturan ---\n" +
        `Jenis Pendapatan : ${filters.selectedTaxType}\n` +
        `Provinsi Target  : ${filters.selectedProvinces.join(", ") || "Semua"}\n` +
        `Periode Prediksi : ${filters.forecastMonths} Bulan\n` +
        `Pencegahan Fraud : ${filters.fraudPreventionPct}%\n` +
        "============================================"
      );
    }
  }, [kpiData, filters, loading, data]);

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
          targetPercentage={kpiData.targetPercentage}
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
              onShowRecs={() => {
                setShowRecs(true);
                setTimeout(() => {
                  document.getElementById('rekomendasi-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 85);
              }}
            />
            <AIInsights insightText={insightText} />
          </div>
        </div>

        {/* Policy Recommendations (Moved up for UX Best Practice) */}
        {showRecs && (
          <div id="rekomendasi-section" className="animate-fade-in-up" style={{ marginBottom: 32 }}>
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
                      <div className="policy-grid">
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
                      <div className="policy-grid" style={{ fontSize: 13 }}>
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
                          <div style={{ marginBottom: 12, fontSize: 13 }}>
                            <span style={{ fontWeight: 600, color: "#475569", marginRight: 8 }}>Justifikasi:</span>
                            <span style={{ color: "#334155" }}>{rec.justifikasi}</span>
                          </div>
                        )}
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, fontSize: 12 }}>
                          {rec.indikator_dampak && (
                            <div style={{ display: "flex", alignItems: "flex-start", gap: 4, flex: "1 1 200px" }}>
                              <span style={{ fontWeight: 600, color: "#64748b" }}>Indikator:</span>
                              <span style={{ color: "#0f172a" }}>{rec.indikator_dampak}</span>
                            </div>
                          )}
                          {rec.kaitan_bisnis && (
                            <div style={{ display: "flex", alignItems: "flex-start", gap: 4, flex: "1 1 200px" }}>
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
              historical={filteredHistorical.filter(r => r.Jenis_Pendapatan === (filters.selectedTaxType === "Semua Pendapatan" ? "Total Pendapatan Daerah" : filters.selectedTaxType))}
              forecast={filteredForecast.filter(r => r.Jenis_Pendapatan === (filters.selectedTaxType === "Semua Pendapatan" ? "Total Pendapatan Daerah" : filters.selectedTaxType))}
            />
          </div>
          <div className="chart-card animate-fade-in-up" style={{ animationDelay: "100ms" }}>
            <h3 className="section-title">Proporsi Sumber Pendapatan</h3>
            <ProportionChart historical={kpiData.thisYearProportionRecords} />
          </div>
        </div>

        {/* Separator */}
        <hr className="separator" />

        {/* Detailed Data Logs */}
        <div id="data-logs">
          <h3 className="section-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <FileText size={18} /> Detailed Data Logs
          </h3>
          <DataTabs
            forecast={filteredForecast}
          anomalies={kpiData.anomalies}
          accuracy={data.accuracy}
          business={bizData}
          historical={filteredHistorical}
          selectedProvinces={filters.selectedProvinces}
          forecastMonths={filters.forecastMonths}
          />
        </div>

      </main>
    </div>
  );
}
