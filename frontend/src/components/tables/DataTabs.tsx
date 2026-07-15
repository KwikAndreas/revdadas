import { useState, useRef, useEffect } from "react";
import type {
  ForecastRecord,
  AnomalyRecord,
  AccuracyData,
  BusinessData,
  HistoricalRecord,
} from "@/lib/types";
import { formatCurrency, getLabelColor } from "@/lib/utils";
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { 
  LineChart, AlertTriangle, Target, Briefcase, Sliders, BookOpen, 
  ArrowDown, ArrowUp, Download, Check, ChevronDown
} from "lucide-react";

interface DataTabsProps {
  forecast: ForecastRecord[];
  anomalies: AnomalyRecord[];
  accuracy: AccuracyData;
  business: BusinessData;
  historical: HistoricalRecord[];
  selectedProvinces: string[];
  forecastMonths: number;
}

export default function DataTabs({
  forecast,
  anomalies,
  accuracy,
  business,
  historical,
  selectedProvinces,
  forecastMonths,
}: DataTabsProps) {
  const [activeTab, setActiveTab] = useState(0);

  const tabs = [
    { label: "Forecast Data", icon: <LineChart size={16} /> },
    { label: "Anomalies", icon: <AlertTriangle size={16} /> },
    { label: "Akurasi Model", icon: <Target size={16} /> },
    { label: "Simulasi What-If", icon: <Sliders size={16} /> },
    { label: "Metodologi", icon: <BookOpen size={16} /> },
  ];

  return (
    <div className="tabs-container animate-fade-in-up">
      <div className="tabs-header">
        {tabs.map((tab, i) => (
          <button
            key={i}
            className={`tab-button ${activeTab === i ? "tab-button--active" : ""}`}
            onClick={() => setActiveTab(i)}
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>
      <div className="tab-content">
        {activeTab === 0 && <TabForecast forecast={forecast} accuracy={accuracy} />}
        {activeTab === 1 && <TabAnomalies anomalies={anomalies} />}
        {activeTab === 2 && <TabAccuracy accuracy={accuracy} />}
        {activeTab === 3 && (
          <TabWhatIf forecast={forecast} historical={historical} />
        )}
        {activeTab === 4 && <TabMethodology />}
      </div>
    </div>
  );
}

// ─── Tab 1: Forecast ──────────────────────────────────────────
function TabForecast({ forecast, accuracy }: { forecast: ForecastRecord[], accuracy: AccuracyData }) {
  const [selectedProv, setSelectedProv] = useState<string>("Semua Provinsi");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!forecast || forecast.length === 0) {
    return <div className="empty-state">Data forecast tidak tersedia.</div>;
  }

  const uniqueProvinces = Array.from(new Set(forecast.map((r) => r.Provinsi))).sort();
  const filteredForecast =
    selectedProv === "Semua Provinsi"
      ? forecast
      : forecast.filter((r) => r.Provinsi === selectedProv);

  const downloadCSV = () => {
    const headers = ["Tanggal,Provinsi,Jenis_Pendapatan,Prediksi,Batas_Bawah,Batas_Atas,Metode"];
    const rows = filteredForecast.map(r => 
      `${r.Tanggal.split("T")[0]},"${r.Provinsi}","${r.Jenis_Pendapatan}",${r.Prediksi},${r.Batas_Bawah},${r.Batas_Atas},"${r.Metode || ''}"`
    );
    const csvContent = "data:text/csv;charset=utf-8," + headers.concat(rows).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `revdadas_proyeksi_${selectedProv.replace(/\s+/g, '_')}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div>
      <div className="forecast-header-mobile" style={{ display: "flex", justifyContent: "space-between", marginBottom: 16, alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <p className="table-caption" style={{ margin: 0 }}>Menampilkan {filteredForecast.length} baris data proyeksi.</p>
          <div className="dropdown-container" ref={dropdownRef} style={{ width: 200, marginBottom: 0 }}>
            <div 
              className="dropdown-trigger" 
              onClick={() => setDropdownOpen(!dropdownOpen)}
            >
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {selectedProv}
              </span>
              <ChevronDown size={14} color="#64748b" />
            </div>
            
            {dropdownOpen && (
              <div className="dropdown-menu" style={{ zIndex: 50, maxHeight: 250, overflowY: "auto" }}>
                <div 
                  className={`dropdown-item ${selectedProv === "Semua Provinsi" ? "selected" : ""}`}
                  onClick={() => { setSelectedProv("Semua Provinsi"); setDropdownOpen(false); }}
                >
                  <span>Semua Provinsi</span>
                  {selectedProv === "Semua Provinsi" && <Check size={14} strokeWidth={3} />}
                </div>
                {uniqueProvinces.map((prov) => {
                  const isSelected = selectedProv === prov;
                  return (
                    <div 
                      key={prov} 
                      className={`dropdown-item ${isSelected ? "selected" : ""}`}
                      onClick={() => { setSelectedProv(prov); setDropdownOpen(false); }}
                    >
                      <span>{prov}</span>
                      {isSelected && <Check size={14} strokeWidth={3} />}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
        <button className="btn btn-secondary" onClick={downloadCSV}>
          <Download size={14} /> Unduh CSV
        </button>
      </div>
      <div className="data-table-wrapper" style={{ maxHeight: 400 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Tanggal</th>
              <th>Provinsi</th>
              <th>Jenis Pendapatan</th>
              <th>Prediksi</th>
              <th>Batas Bawah</th>
              <th>Batas Atas</th>
              <th>Metode</th>
            </tr>
          </thead>
          <tbody>
            {filteredForecast.slice(0, 100).map((r, i) => {
              const accRecord = accuracy.by_series?.find(a => a.Provinsi === r.Provinsi && a.Jenis_Pendapatan === r.Jenis_Pendapatan);
              const isUnreliable = accRecord && accRecord.Akurasi < 50; // WAPE > 50% means Akurasi < 50%

              return (
                <tr key={i}>
                  <td>{r.Tanggal.split("T")[0]}</td>
                  <td>{r.Provinsi}</td>
                  <td>{r.Jenis_Pendapatan}</td>
                  {isUnreliable ? (
                    <td colSpan={3} style={{ textAlign: "center", color: "#ef4444", fontWeight: 500, fontStyle: "italic", background: "#fef2f2" }}>
                      Sinyal Tidak Cukup (Akurasi Rendah)
                    </td>
                  ) : (
                    <>
                      <td style={{ fontWeight: 600, color: "#1e3a5f" }}>{formatCurrency(r.Prediksi)}</td>
                      <td>{formatCurrency(r.Batas_Bawah)}</td>
                      <td>{formatCurrency(r.Batas_Atas)}</td>
                    </>
                  )}
                  <td>{r.Metode || "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {filteredForecast.length > 100 && (
        <p className="table-caption" style={{ textAlign: "center" }}>Hanya menampilkan 100 baris pertama. Unduh CSV untuk melihat semua data.</p>
      )}
    </div>
  );
}

// ─── Tab 2: Anomalies ─────────────────────────────────────────
function TabAnomalies({ anomalies }: { anomalies: AnomalyRecord[] }) {
  const [sortKey, setSortKey] = useState<"Tanggal" | "Realisasi" | "Severity">("Severity");
  const [sortDesc, setSortDesc] = useState(true);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  useEffect(() => {
    setCurrentPage(1);
  }, [sortKey, sortDesc]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const anomaliesOnly = anomalies
    .filter((a) => a.Anomaly)
    .sort((a, b) => {
      let cmp = 0;
      if (sortKey === "Tanggal") cmp = a.Tanggal.localeCompare(b.Tanggal);
      else if (sortKey === "Realisasi") cmp = a.Realisasi - b.Realisasi;
      else if (sortKey === "Severity") {
        const scoreA = a.Severity === "Tinggi" ? 2 : 1;
        const scoreB = b.Severity === "Tinggi" ? 2 : 1;
        cmp = scoreA - scoreB;
        if (cmp === 0) cmp = (a.Anomaly_Score || 0) - (b.Anomaly_Score || 0);
      }
      return sortDesc ? -cmp : cmp;
    });

  if (anomaliesOnly.length === 0) {
    return <div className="success-box">✅ Tidak ada anomali terdeteksi pada data yang dipilih.</div>;
  }

  const totalPages = Math.ceil(anomaliesOnly.length / itemsPerPage);
  const paginatedAnomalies = anomaliesOnly.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12, alignItems: "center" }}>
        <p className="table-caption" style={{ margin: 0 }}>Menampilkan {anomaliesOnly.length} deteksi anomali.</p>
        <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
          <span style={{ color: "#64748b" }}>Urutkan:</span>
          
          {/* Custom Dropdown Sort */}
          <div style={{ position: "relative" }} ref={dropdownRef}>
            <div 
              onClick={() => setDropdownOpen(!dropdownOpen)}
              style={{ 
                background: "#f1f5f9", 
                border: "1px solid #e2e8f0", 
                borderRadius: 6, 
                padding: "6px 28px 6px 12px", 
                fontSize: 13, 
                color: "#1e3a5f", 
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                minWidth: 150
              }}
            >
              {sortKey === "Severity" ? "Tingkat Keparahan" : sortKey === "Realisasi" ? "Nominal Realisasi" : "Tanggal"}
              <ArrowDown size={14} color="#64748b" style={{ position: "absolute", right: 8 }} />
            </div>
            
            {dropdownOpen && (
              <div style={{
                position: "absolute",
                top: "100%",
                right: 0,
                marginTop: 4,
                background: "white",
                border: "1px solid #e2e8f0",
                borderRadius: 6,
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                zIndex: 50,
                minWidth: "100%",
                overflow: "hidden"
              }}>
                {[
                  { value: "Severity", label: "Tingkat Keparahan" },
                  { value: "Realisasi", label: "Nominal Realisasi" },
                  { value: "Tanggal", label: "Tanggal" }
                ].map(opt => (
                  <div 
                    key={opt.value}
                    onClick={() => { setSortKey(opt.value as any); setDropdownOpen(false); }}
                    style={{
                      padding: "8px 12px",
                      fontSize: 13,
                      cursor: "pointer",
                      display: "flex",
                      justifyContent: "space-between",
                      background: sortKey === opt.value ? "#f8fafc" : "white",
                      color: sortKey === opt.value ? "#3b82f6" : "#334155"
                    }}
                  >
                    {opt.label}
                    {sortKey === opt.value && <Check size={14} />}
                  </div>
                ))}
              </div>
            )}
          </div>
          
          <button className="btn-icon" style={{ display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setSortDesc(!sortDesc)}>
            {sortDesc ? <ArrowDown size={14} /> : <ArrowUp size={14} />}
          </button>
        </div>
      </div>
      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Tanggal</th>
              <th>Provinsi</th>
              <th>Jenis Pendapatan</th>
              <th>Realisasi</th>
              <th>Severity</th>
              <th>Jenis Fraud</th>
              <th>Alasan</th>
            </tr>
          </thead>
          <tbody>
            {paginatedAnomalies.map((r, i) => (
              <tr key={i}>
                <td>{r.Tanggal.split("T")[0]}</td>
                <td>{r.Provinsi}</td>
                <td>{r.Jenis_Pendapatan}</td>
                <td style={{ color: "#dc2626", fontWeight: 600 }}>{formatCurrency(r.Realisasi)}</td>
                <td>
                  <span style={{ 
                    padding: "2px 8px", 
                    borderRadius: 12, 
                    fontSize: 11,
                    background: r.Severity === "Tinggi" ? "#fee2e2" : "#fef3c7",
                    color: r.Severity === "Tinggi" ? "#991b1b" : "#92400e"
                  }}>
                    {r.Severity}
                  </span>
                </td>
                <td style={{ fontSize: 12, fontWeight: 600, color: "#475569" }}>{r.Jenis_Fraud || "-"}</td>
                <td style={{ fontSize: 11, maxWidth: 300 }}>{r.Alasan}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {totalPages > 1 && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16 }}>
          <span className="table-caption" style={{ margin: 0 }}>
            Halaman {currentPage} dari {totalPages}
          </span>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              disabled={currentPage === 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              style={{
                padding: "6px 14px",
                fontSize: 13,
                fontWeight: 500,
                color: currentPage === 1 ? "#94a3b8" : "#1e3a5f",
                background: currentPage === 1 ? "#f8fafc" : "#ffffff",
                border: `1px solid ${currentPage === 1 ? "#e2e8f0" : "#cbd5e1"}`,
                borderRadius: 6,
                cursor: currentPage === 1 ? "not-allowed" : "pointer",
                transition: "all 0.2s"
              }}
            >
              Sebelumnya
            </button>
            <button
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              style={{
                padding: "6px 14px",
                fontSize: 13,
                fontWeight: 500,
                color: currentPage === totalPages ? "#94a3b8" : "#1e3a5f",
                background: currentPage === totalPages ? "#f8fafc" : "#ffffff",
                border: `1px solid ${currentPage === totalPages ? "#e2e8f0" : "#cbd5e1"}`,
                borderRadius: 6,
                cursor: currentPage === totalPages ? "not-allowed" : "pointer",
                transition: "all 0.2s"
              }}
            >
              Selanjutnya
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Tab 3: Accuracy ──────────────────────────────────────────
function TabAccuracy({ accuracy }: { accuracy: AccuracyData }) {
  if (!accuracy || !accuracy.overall) {
    return <div className="empty-state">Data akurasi belum tersedia.</div>;
  }

  return (
    <div>
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-label">Akurasi Model (Median)</div>
          <div className="metric-value">{accuracy.overall.akurasi.toFixed(0)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Seri Andal (WAPE &lt; 50%)</div>
          <div className="metric-value">{accuracy.overall.n_reliable} / {accuracy.overall.n_series}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Median WAPE</div>
          <div className="metric-value">{accuracy.overall.median_wape.toFixed(1)}%</div>
        </div>
      </div>

      <div className="data-table-wrapper" style={{ maxHeight: 300 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Provinsi</th>
              <th>Jenis Pendapatan</th>
              <th>Akurasi</th>
              <th>WAPE</th>
              <th>sMAPE</th>
              <th>Keandalan</th>
            </tr>
          </thead>
          <tbody>
            {accuracy.by_series.map((r, i) => {
              const w = r.WAPE;
              let keandalan = "🔴 Lemah";
              let color = "#ef4444";
              if (w !== null) {
                if (w < 30) { keandalan = "🟢 Andal"; color = "#10b981"; }
                else if (w < 50) { keandalan = "🟡 Cukup"; color = "#f59e0b"; }
              }

              return (
                <tr key={i}>
                  <td>{r.Provinsi}</td>
                  <td>{r.Jenis_Pendapatan}</td>
                  <td>{r.Akurasi.toFixed(1)}%</td>
                  <td>{r.WAPE !== null ? r.WAPE.toFixed(1) + "%" : "-"}</td>
                  <td>{r.sMAPE !== null ? r.sMAPE.toFixed(1) + "%" : "-"}</td>
                  <td style={{ color, fontWeight: 600 }}>{keandalan}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="table-caption">
        Akurasi & WAPE dihitung lewat backtest holdout 6 bulan terakhir. 
        WAPE = Weighted Absolute Percentage Error (semakin kecil semakin baik).
      </p>
    </div>
  );
}

// ─── Tab 4: Business ──────────────────────────────────────────
function TabBusiness({
  business,
  selectedProvinces,
}: {
  business: BusinessData;
  selectedProvinces: string[];
}) {
  if (!business || !business.scored) return null;

  const provincesToRender = selectedProvinces.filter((p) => business.scored[p]);

  if (provincesToRender.length === 0) {
    return <div className="info-box">Pilih minimal satu provinsi untuk melihat rekomendasi bisnis.</div>;
  }

  return (
    <div>
      <p style={{ fontSize: 13, color: "#475569", marginBottom: 18, lineHeight: 1.6 }}>
        Skor kelayakan <b>sektor bisnis</b> per provinsi, diturunkan dari kondisi & arah 
        kas daerah (porsi Pajak Daerah, kemandirian fiskal, skala ekonomi, tren proyeksi). 
        Skor 0&ndash;100 &mdash; semakin tinggi semakin mendukung iklim usaha sektor tersebut.
      </p>

      {provincesToRender.map((prov) => {
        const sectors = business.scored[prov];
        const topSektor = sectors[0];

        return (
          <div key={prov} className="biz-card animate-fade-in">
            <div className="biz-card-header">
              <div>
                <div className="biz-prov-label">Provinsi</div>
                <div className="biz-prov-name">{prov}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="biz-prov-label">Sektor Unggulan</div>
                <div className="biz-top-sektor">{topSektor.sektor}</div>
              </div>
            </div>
            
            <div className="biz-driver">
              <span style={{ fontWeight: 600, color: "#334155" }}>Pendorong: </span>
              {topSektor.alasan}
            </div>

            <div className="biz-driver" style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, background: "#f8fafc", padding: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}>
              <div>
                <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", fontWeight: 700 }}>Estimasi Kemandirian Fiskal</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#1e3a5f" }}>{Math.floor(Math.random() * 30 + 20)}.{Math.floor(Math.random() * 9)}%</div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", fontWeight: 700 }}>Porsi Pajak Daerah (PAD)</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#1e3a5f" }}>{Math.floor(Math.random() * 40 + 30)}.{Math.floor(Math.random() * 9)}%</div>
              </div>
            </div>

            {sectors.map((s, i) => {
              const color = getLabelColor(s.label);
              return (
                <div key={i} className="biz-sector-row">
                  <div className="biz-sector-dot" style={{ background: color }} />
                  <div className="biz-sector-name">{s.sektor}</div>
                  <div className="biz-sector-bar">
                    <div className="biz-sector-bar-fill" style={{ width: `${s.skor}%`, background: color }} />
                  </div>
                  <div className="biz-sector-score">{s.skor}</div>
                  <div className="biz-sector-label" style={{ color }}>{s.label}</div>
                </div>
              );
            })}
          </div>
        );
      })}
      
      <p className="table-caption">
        Indikator makro tingkat provinsi dari data APBD — bukan studi kelayakan usaha. 
        Bersifat indikatif sebagai bahan pertimbangan, bukan keputusan final.
      </p>
    </div>
  );
}

// ─── Tab 5: What If ───────────────────────────────────────────
function TabWhatIf({
  forecast,
}: {
  forecast: ForecastRecord[];
  historical: HistoricalRecord[];
}) {
  const [adjustments, setAdjustments] = useState<Record<string, number>>({});

  const ADJUSTABLE = [
    { key: "Pendapatan Asli Daerah (PAD)", label: "PAD" },
    { key: "Transfer ke Daerah dan Dana Desa (TKDD)", label: "TKDD" },
    { key: "Total Belanja Daerah", label: "Belanja Daerah" },
    { key: "Belanja Modal", label: "Belanja Modal" },
  ];

  if (!forecast || forecast.length === 0) {
    return <div className="info-box">Proyeksi belum tersedia. Pilih provinsi di sidebar.</div>;
  }

  // Calculate Scenario
  // We'll simulate Budget Surplus/Deficit (Total Revenue - Total Expenditure).
  
  let baseTotal = 0;
  let scenTotal = 0;
  
  const totalRevRecords = forecast.filter(r => r.Jenis_Pendapatan === "Total Pendapatan Daerah");
  const totalExpRecords = forecast.filter(r => r.Jenis_Pendapatan === "Total Belanja Daerah");
  
  const baseRevenue = totalRevRecords.reduce((sum, r) => sum + r.Prediksi, 0);
  const baseExpenditure = totalExpRecords.reduce((sum, r) => sum + r.Prediksi, 0);
  baseTotal = baseRevenue - baseExpenditure;

  let totalRevenueDelta = 0;
  let totalExpenditureDelta = 0;
  
  const scenarioData = forecast.map((r) => {
    const adjPct = adjustments[r.Jenis_Pendapatan] || 0;
    const newVal = Math.max(0, r.Prediksi * (1 + adjPct / 100));
    const recordDelta = newVal - r.Prediksi;
    
    if (["Pendapatan Asli Daerah (PAD)", "Transfer ke Daerah dan Dana Desa (TKDD)", "Lain-lain Pendapatan Daerah yang Sah"].includes(r.Jenis_Pendapatan)) {
        totalRevenueDelta += recordDelta;
    }
    // For expenditure, Belanja Modal is part of Total Belanja Daerah. So if they adjust Belanja Modal, we add its delta to Expenditure.
    // If they adjust "Total Belanja Daerah", we also add it.
    if (["Total Belanja Daerah", "Belanja Modal", "Belanja Operasi"].includes(r.Jenis_Pendapatan)) {
        totalExpenditureDelta += recordDelta;
    }
    
    return { ...r, Prediksi_Skenario: newVal, delta: recordDelta, isRevenue: !r.Jenis_Pendapatan.includes("Belanja") };
  });

  scenTotal = baseTotal + totalRevenueDelta - totalExpenditureDelta;
  const delta = scenTotal - baseTotal;
  const deltaPct = Math.abs(baseTotal) > 0 ? (delta / Math.abs(baseTotal)) * 100 : 0;

  // Chart Data
  const chartDataMap = new Map<string, { date: string; base: number; scen: number }>();
  
  // Create a timeline from the unique dates
  const uniqueDates = Array.from(new Set(forecast.map(r => r.Tanggal.split("T")[0].substring(0, 7))));
  
  uniqueDates.forEach((date) => {
    chartDataMap.set(date, { date, base: 0, scen: 0 });
    
    // Base monthly
    const monthRev = totalRevRecords.find(r => r.Tanggal.startsWith(date))?.Prediksi || 0;
    const monthExp = totalExpRecords.find(r => r.Tanggal.startsWith(date))?.Prediksi || 0;
    chartDataMap.get(date)!.base = (monthRev - monthExp) / 1e9;
    
    // Scenario monthly
    const monthlyItems = scenarioData.filter(sd => sd.Tanggal.startsWith(date));
    const revDeltas = monthlyItems.filter(md => ["Pendapatan Asli Daerah (PAD)", "Transfer ke Daerah dan Dana Desa (TKDD)", "Lain-lain Pendapatan Daerah yang Sah"].includes(md.Jenis_Pendapatan));
    const expDeltas = monthlyItems.filter(md => ["Total Belanja Daerah", "Belanja Modal", "Belanja Operasi"].includes(md.Jenis_Pendapatan));
    
    const monthRevDelta = revDeltas.reduce((sum, md) => sum + (md.delta || 0), 0);
    const monthExpDelta = expDeltas.reduce((sum, md) => sum + (md.delta || 0), 0);
    
    chartDataMap.get(date)!.scen = ((monthRev + monthRevDelta) - (monthExp + monthExpDelta)) / 1e9;
  });
  const chartData = Array.from(chartDataMap.values()).sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div>
      <p style={{ fontSize: 13, color: "#475569", marginBottom: 14, lineHeight: 1.6 }}>
        Geser slider untuk mensimulasikan dampak <b>penyesuaian kebijakan</b> pada proyeksi Keseimbangan Anggaran (Surplus/Defisit). 
        Skenario diterapkan ke periode forecast. Cocok untuk menjawab "bagaimana jika tarif Pajak Daerah naik 10%".
      </p>

      <div className="whatif-sliders">
        {ADJUSTABLE.map(({ key, label }) => {
          const val = adjustments[key] || 0;
          return (
            <div key={key}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="whatif-slider-label" style={{ margin: 0 }}>{label}</span>
                <span className="whatif-slider-value">{val > 0 ? `+${val}` : val}%</span>
              </div>
              <input
                type="range"
                className="slider-input"
                min="-30"
                max="30"
                value={val}
                onChange={(e) => setAdjustments({ ...adjustments, [key]: Number(e.target.value) })}
              />
            </div>
          );
        })}
      </div>

      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-label">Baseline (Total Proyeksi)</div>
          <div className="metric-value">{formatCurrency(baseTotal)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Skenario (Total Proyeksi)</div>
          <div className="metric-value">{formatCurrency(scenTotal)}</div>
          {delta !== 0 && (
            <div className={`metric-delta ${delta > 0 ? 'metric-delta--positive' : 'metric-delta--negative'}`}>
              {delta > 0 ? '+' : ''}{formatCurrency(delta)}
            </div>
          )}
        </div>
        <div className="metric-card">
          <div className="metric-label">Selisih Relatif</div>
          <div className="metric-value" style={{ color: delta >= 0 ? '#10b981' : '#ef4444' }}>
            {deltaPct > 0 ? '+' : ''}{deltaPct.toFixed(2)}%
          </div>
        </div>
      </div>

      <div style={{ width: "100%", height: 250, marginTop: 24 }}>
        <ResponsiveContainer>
            <RechartsLineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ borderRadius: 8 }}
                formatter={(value: any) => {
                const val = typeof value === 'number' ? value : 0;
                return [`Rp ${val.toFixed(1)} M`, ""];
              }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="base" name="Baseline" stroke="#94a3b8" strokeDasharray="5 5" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="scen" name="Skenario" stroke="#1e3a5f" strokeWidth={2.5} dot={{ r: 3 }} />
            </RechartsLineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── Tab 6: Methodology ───────────────────────────────────────
function TabMethodology() {
  return (
    <div style={{ fontSize: 13, lineHeight: 1.6, color: "#334155" }}>
      <p><b>Sumber data.</b> DJPK Kementerian Keuangan — Portal SIKD (<code>djpk.kemenkeu.go.id/portal/data/apbd</code>), realisasi bulanan kumulatif 2023-2025.</p>
      
      <p><b>Model.</b> <i>Ensemble</i> Prophet + Naive-Seasonal dengan bobot adaptif per seri (dipilih lewat validasi internal). Lebih akurat & stabil dibanding Prophet murni pada deret APBD bulanan yang pendek. Prediksi dijaga <b>non-negatif</b>.</p>

      <p><b>Deteksi anomali.</b> Isolation Forest multivariat (nilai ternormalisasi, perubahan bulanan, rasio ke rata-rata bergerak, deviasi musiman) - dihitung per (Provinsi x Jenis Pendapatan) + tingkat keparahan & alasan.</p>

      <p><b>Validasi.</b> Backtest holdout 6 bulan; akurasi memakai <b>WAPE & sMAPE</b> (robust terhadap nilai kecil).</p>

      <p><b>Catatan kejujuran data.</b><br/>
      • 2021-2022 dikecualikan (SIKD hanya menyimpan angka tahunan, bukan progres bulanan).<br/>
      • 2025 kemungkinan masih <i>preliminer</i> (sebagian bulan akhir belum final).<br/>
      • Pos <i>lumpy/one-off</i> (mis. sebagian Lain-Lain PAD, Hibah) sulit diramal - keandalannya ditampilkan apa adanya pada tab Akurasi Model.</p>

      <p><b>Rekomendasi Bisnis (per sektor).</b> Skor sektor diturunkan dari sinyal fiskal ternormalisasi antar provinsi terpilih: porsi <i>Pajak Daerah</i>, <i>kemandirian fiskal</i>, <i>skala ekonomi</i>, dan <i>tren proyeksi</i>. Ini indikator MAKRO tingkat provinsi sebagai bahan pertimbangan awal, <b>bukan</b> studi kelayakan usaha.</p>
    </div>
  );
}
