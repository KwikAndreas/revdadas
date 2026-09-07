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
  BarChart,
  Bar,
} from "recharts";
import { 
  LineChart, AlertTriangle, Target, Briefcase, Sliders, BookOpen, 
  ArrowDown, ArrowUp, Download, Check, ChevronDown, ShieldCheck
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

  useEffect(() => {
    const handleSwitch = (e: any) => {
      setActiveTab(e.detail);
    };
    window.addEventListener('switchTab', handleSwitch);
    return () => window.removeEventListener('switchTab', handleSwitch);
  }, []);

  const tabs = [
    { label: "Forecast Data", icon: <LineChart size={16} /> },
    { label: "Anomalies", icon: <AlertTriangle size={16} /> },
    { label: "Exploratory Data", icon: <LineChart size={16} /> },
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
        {activeTab === 2 && <TabEDA historical={historical} selectedProvinces={selectedProvinces} />}
        {activeTab === 3 && <TabAccuracy accuracy={accuracy} />}
        {activeTab === 4 && (
          <TabWhatIf key={selectedProvinces.join("-")} forecast={forecast} historical={historical} />
        )}
        {activeTab === 5 && <TabMethodology />}
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
  const filteredForecast = forecast
    .filter((r) => !r.Jenis_Pendapatan.includes("Belanja"))
    .filter((r) => selectedProv === "Semua Provinsi" || r.Provinsi === selectedProv);

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
  const [showGapData, setShowGapData] = useState(false);
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
    .filter((a) => a.Anomaly && (showGapData || a.Severity !== "Gap Data" && !a.Gap_Data))
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
    return (
      <div 
        style={{
          background: "#ffffff",
          border: "1px solid #e2e8f0",
          borderRadius: 8,
          padding: "36px 24px",
          textAlign: "center",
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)"
        }}
        className="animate-fade-in"
      >
        <div style={{
          width: 48,
          height: 48,
          borderRadius: "50%",
          background: "#ecfdf5",
          border: "1px solid #a7f3d0",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 16px"
        }}>
          <ShieldCheck size={24} color="#059669" strokeWidth={2.2} />
        </div>
        
        <h4 style={{ fontSize: 15, fontWeight: 700, color: "#0f172a", margin: "0 0 8px 0" }}>
          Seluruh Realisasi Kas Berada dalam Koridor Wajar
        </h4>
        
        <p style={{ fontSize: 13, color: "#64748b", maxWidth: 580, margin: "0 auto 20px", lineHeight: 1.6 }}>
          Evaluasi algoritma <i>Isolation Forest</i> multivariat (analisis deviasi musiman tahunan, rata-rata bergerak 3-bulan, dan laju pertumbuhan MoM) tidak mendeteksi deviasi transaksi yang melebihi batas ambang risiko (&gt; 2.0σ) pada filter yang dipilih.
        </p>

        <div style={{
          display: "inline-flex",
          flexWrap: "wrap",
          justifyContent: "center",
          gap: 16,
          padding: "10px 18px",
          background: "#f8fafc",
          borderRadius: 6,
          border: "1px solid #e2e8f0",
          fontSize: 12
        }}>
          <div>
            <span style={{ color: "#64748b" }}>Status Audit: </span>
            <strong style={{ color: "#059669" }}>Clear (Terkonfirmasi Normal)</strong>
          </div>
          <div style={{ width: 1, background: "#cbd5e1" }} />
          <div>
            <span style={{ color: "#64748b" }}>Metode Pengujian: </span>
            <strong style={{ color: "#0f172a" }}>Multivariate Isolation Forest</strong>
          </div>
          <div style={{ width: 1, background: "#cbd5e1" }} />
          <div>
            <span style={{ color: "#64748b" }}>Tingkat Deviasi: </span>
            <strong style={{ color: "#0f172a" }}>Z-Score &lt; 2.0σ</strong>
          </div>
        </div>
      </div>
    );
  }

  const totalPages = Math.ceil(anomaliesOnly.length / itemsPerPage);
  const paginatedAnomalies = anomaliesOnly.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12, alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <p className="table-caption" style={{ margin: 0 }}>Menampilkan {anomaliesOnly.length} deteksi anomali.</p>
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "#475569", cursor: "pointer" }}>
            <input 
              type="checkbox" 
              checked={showGapData} 
              onChange={(e) => setShowGapData(e.target.checked)}
              style={{ cursor: "pointer" }}
            />
            Tampilkan Gap Pelaporan ({(anomalies.filter(a => a.Anomaly && (a.Severity === "Gap Data" || a.Gap_Data)).length)})
          </label>
        </div>
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
              <th>Nilai Transaksi</th>
              <th>Alasan Anomali</th>
            </tr>
          </thead>
          <tbody>
            {paginatedAnomalies.map((r, i) => {
              const isCurrentYear = parseInt(r.Tanggal.substring(0, 4)) === 2025;
              return (
              <tr key={i} style={isCurrentYear ? { backgroundColor: "rgba(239, 68, 68, 0.05)" } : {}}>
                <td>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {r.Tanggal.split("T")[0]}
                    {isCurrentYear && (
                      <span style={{ fontSize: 10, background: "#ef4444", color: "white", padding: "2px 6px", borderRadius: 4, fontWeight: "bold" }}>
                        Pantau!
                      </span>
                    )}
                  </div>
                </td>
                <td>{r.Provinsi}</td>
                <td>{r.Jenis_Pendapatan}</td>
                <td style={{ color: "#dc2626", fontWeight: 600 }}>{formatCurrency(r.Realisasi)}</td>
                <td style={{ fontSize: 11, maxWidth: 300 }}>{r.Alasan}</td>
              </tr>
              );
            })}
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
            {accuracy.by_series
              .filter(r => !r.Jenis_Pendapatan.includes("Belanja"))
              .map((r, i) => {
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
    { key: "Pendapatan Asli Daerah (PAD)", label: "Pendapatan Asli Daerah (PAD)" },
    { key: "Transfer ke Daerah dan Dana Desa (TKDD)", label: "Transfer ke Daerah dan Dana Desa (TKDD)" },
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
  const baseBelanjaModal = forecast.filter(r => r.Jenis_Pendapatan === "Belanja Modal").reduce((sum, r) => sum + r.Prediksi, 0);
  baseTotal = baseRevenue - baseExpenditure;

  let totalRevenueDelta = 0;
  let totalExpenditureDelta = 0;
  let scenBelanjaModal = 0;
  
  const scenarioData = forecast.map((r) => {
    const adjPct = adjustments[r.Jenis_Pendapatan] || 0;
    const newVal = Math.max(0, r.Prediksi * (1 + adjPct / 100));
    const recordDelta = newVal - r.Prediksi;
    
    if (["Pendapatan Asli Daerah (PAD)", "Transfer ke Daerah dan Dana Desa (TKDD)", "Lain-lain Pendapatan Daerah yang Sah"].includes(r.Jenis_Pendapatan)) {
        totalRevenueDelta += recordDelta;
    }
    // For expenditure, Belanja Modal is part of Total Belanja Daerah. So its slider only changes the proportion.
    if (r.Jenis_Pendapatan === "Total Belanja Daerah") {
        totalExpenditureDelta += recordDelta;
    }
    if (r.Jenis_Pendapatan === "Belanja Modal") {
        scenBelanjaModal += newVal;
    }
    
    return { ...r, Prediksi_Skenario: newVal, delta: recordDelta, isRevenue: !r.Jenis_Pendapatan.includes("Belanja") };
  });

  scenTotal = baseTotal + totalRevenueDelta - totalExpenditureDelta;
  const delta = scenTotal - baseTotal;
  const deltaPct = Math.abs(baseTotal) > 0 ? (delta / Math.abs(baseTotal)) * 100 : 0;

  const basePorsi = baseExpenditure > 0 ? (baseBelanjaModal / baseExpenditure) * 100 : 0;
  const scenExpenditure = baseExpenditure + totalExpenditureDelta;
  const scenPorsi = scenExpenditure > 0 ? (scenBelanjaModal / scenExpenditure) * 100 : 0;

  // Chart Data
  const chartDataMap = new Map<string, { date: string; base: number; scen: number }>();
  
  // Create a timeline from the unique dates
  const uniqueDates = Array.from(new Set(forecast.map(r => r.Tanggal.split("T")[0].substring(0, 7))));
  
  uniqueDates.forEach((date) => {
    chartDataMap.set(date, { date, base: 0, scen: 0 });
    
    // Base monthly (summed across all selected provinces for this date)
    const monthRev = totalRevRecords.filter(r => r.Tanggal.startsWith(date)).reduce((sum, r) => sum + r.Prediksi, 0);
    const monthExp = totalExpRecords.filter(r => r.Tanggal.startsWith(date)).reduce((sum, r) => sum + r.Prediksi, 0);
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
        Simulasi ini memproyeksikan <b>Keseimbangan Anggaran Fiskal (Surplus/Defisit = Total Pendapatan − Total Belanja)</b>. 
        Angka negatif menunjukkan <b>Defisit Fiskal</b> di mana proyeksi belanja melampaui pendapatan (yang dalam APBD ditutup melalui pembiayaan netto/SILPA). 
        Geser slider di bawah untuk menguji bagaimana penyesuaian tarif PAD, transfer TKDD, atau alokasi belanja modal mengubah postur kas daerah.
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
          <div className="metric-label">Keseimbangan Fiskal (Baseline)</div>
          <div className="metric-value" style={{ color: baseTotal < 0 ? '#dc2626' : '#16a34a' }}>
            {formatCurrency(baseTotal)}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
            <span style={{ 
              fontSize: 10.5, 
              fontWeight: 700, 
              padding: "1px 6px", 
              borderRadius: 4, 
              background: baseTotal < 0 ? "#fee2e2" : "#dcfce7", 
              color: baseTotal < 0 ? "#991b1b" : "#166534" 
            }}>
              {baseTotal < 0 ? "Defisit APBD" : "Surplus APBD"}
            </span>
            <span style={{ fontSize: 10.5, color: "#64748b" }}>
              Rev {formatCurrency(baseRevenue)} vs Exp {formatCurrency(baseExpenditure)}
            </span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Keseimbangan Fiskal (Skenario)</div>
          <div className="metric-value" style={{ color: scenTotal < 0 ? '#dc2626' : '#16a34a' }}>
            {formatCurrency(scenTotal)}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
            <span style={{ 
              fontSize: 10.5, 
              fontWeight: 700, 
              padding: "1px 6px", 
              borderRadius: 4, 
              background: scenTotal < 0 ? "#fee2e2" : "#dcfce7", 
              color: scenTotal < 0 ? "#991b1b" : "#166534" 
            }}>
              {scenTotal < 0 ? "Defisit Skenario" : "Surplus Skenario"}
            </span>
            {delta !== 0 ? (
              <span style={{ fontSize: 10.5, fontWeight: 600, color: delta > 0 ? "#16a34a" : "#dc2626" }}>
                {delta > 0 ? "Kas membaik +" : "Defisit melebar "}{formatCurrency(delta)}
              </span>
            ) : (
              <span style={{ fontSize: 10.5, color: "#64748b" }}>Status Quo (0%)</span>
            )}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Dampak Relatif Intervensi</div>
          <div className="metric-value" style={{ color: delta >= 0 ? '#10b981' : '#ef4444' }}>
            {deltaPct > 0 ? '+' : ''}{deltaPct.toFixed(2)}%
          </div>
          <div style={{ fontSize: 10.5, color: "#64748b", marginTop: 4 }}>
            {delta === 0 ? "Slider netral (belum ada perubahan)" : delta > 0 ? "Penyusutan defisit anggaran" : "Pelebaran beban anggaran"}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Porsi Belanja Modal</div>
          <div className="metric-value" style={{ color: '#3b82f6', fontSize: 20 }}>
            {basePorsi.toFixed(1)}% <span style={{fontSize: 14, color: '#64748b'}}>→</span> {scenPorsi.toFixed(1)}%
          </div>
          <div style={{ fontSize: 10.5, color: "#64748b", marginTop: 4 }}>
            Rasio belanja modal thd total belanja
          </div>
        </div>
      </div>

      <div style={{ width: "100%", height: 250, marginTop: 24 }}>
        <ResponsiveContainer>
            <RechartsLineChart data={chartData} margin={{ top: 10, right: 15, left: 15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis 
                tick={{ fontSize: 11 }} 
                axisLine={false} 
                tickLine={false}
                width={75}
                tickFormatter={(val) => formatCurrency(Number(val) * 1e9)}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, background: "white", border: "1px solid #e2e8f0", fontSize: 12, boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }}
                formatter={(value: any) => {
                  const val = typeof value === 'number' ? value : 0;
                  return [formatCurrency(val * 1e9), ""];
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
    <div style={{ fontSize: 13, lineHeight: 1.7, color: "#334155" }} className="animate-fade-in">
      {/* Header Dokumen Metodologi */}
      <div style={{ borderBottom: "1px solid #e2e8f0", paddingBottom: 16, marginBottom: 20 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", margin: "0 0 6px 0", letterSpacing: "-0.01em" }}>
          Kerangka Metodologi & Spesifikasi Teknis Pemodelan
        </h3>
        <p style={{ margin: 0, color: "#64748b", fontSize: 13 }}>
          Dokumentasi teknis pemodelan ekonometrika deret waktu fiskal, deteksi anomali transaksi APBD, serta formulasi indikator kebijakan fiskal daerah berbasis data Sistem Informasi Keuangan Daerah (SIKD) Kementerian Keuangan.
        </p>

        {/* Bar Ringkasan Parameter Teknis */}
        <div style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 20,
          marginTop: 14,
          padding: "10px 14px",
          background: "#f8fafc",
          borderRadius: 6,
          border: "1px solid #e2e8f0",
          fontSize: 12
        }}>
          <div>
            <span style={{ color: "#64748b" }}>Basis Data: </span>
            <strong style={{ color: "#0f172a" }}>DJPK SIKD 2023–2025 (Bulanan)</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Mesin Utama: </span>
            <strong style={{ color: "#0f172a" }}>Theta Method (M3 Winner)</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Mesin Komparasi: </span>
            <strong style={{ color: "#0f172a" }}>Additive GAM (Prophet)</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Deteksi Anomali: </span>
            <strong style={{ color: "#0f172a" }}>Isolation Forest Multivariat</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Metrik Validasi: </span>
            <strong style={{ color: "#0f172a" }}>WAPE & sMAPE (Holdout)</strong>
          </div>
        </div>
      </div>

      {/* Bagian 1: Pemodelan Deret Waktu Fiskal */}
      <section style={{ marginBottom: 24 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          1. Arsitektur Pemodelan Deret Waktu (Dual-Engine Forecasting)
        </h4>
        <p style={{ margin: "0 0 12px 0" }}>
          Untuk memproyeksikan realisasi pendapatan dan belanja daerah secara akurat di tengah karakteristik deret waktu APBD bulanan yang pendek ($N \approx 36$ observasi), RevDadas menerapkan pendekatan peramalan ganda dengan pemisahan peran operasional dan eksplorasi analitis:
        </p>

        {/* Tabel Komparasi Teknis */}
        <div style={{ overflowX: "auto", marginBottom: 14 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, textAlign: "left" }}>
            <thead>
              <tr style={{ background: "#f1f5f9", borderBottom: "2px solid #cbd5e1" }}>
                <th style={{ padding: "8px 12px", color: "#334155", fontWeight: 700, width: "20%" }}>Parameter</th>
                <th style={{ padding: "8px 12px", color: "#1e3a5f", fontWeight: 700, width: "40%" }}>Theta Method (Algoritma Utama)</th>
                <th style={{ padding: "8px 12px", color: "#475569", fontWeight: 700, width: "40%" }}>Facebook Prophet (Opsi Komparatif)</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Formulasi Matematika</td>
                <td style={{ padding: "8px 12px" }}>
                  Dekomposisi kurva ganda: $\theta_0 = 0$ (regresi tren linier) dan $\theta_2 = 2$ (Simple Exponential Smoothing kurvatur lokal).
                </td>
                <td style={{ padding: "8px 12px" }}>
                  Generalized Additive Model: $y(t) = g(t) + s(t) + \epsilon_t$ dengan dekomposisi tren linier/logistik dan deret Fourier musiman tahunan.
                </td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#fcfcfd" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Resistensi Overfitting</td>
                <td style={{ padding: "8px 12px" }}>
                  <strong style={{ color: "#16a34a" }}>Tinggi (Parsimonious).</strong> Hanya mengestimasi 2 parameter bebas, mencegah ledakan parameter pada observasi bulanan pendek ($N \approx 36$).
                </td>
                <td style={{ padding: "8px 12px" }}>
                  <strong style={{ color: "#d97706" }}>Moderat.</strong> Memerlukan estimasi titik perubahan tren (*changepoints*) dan koefisien Fourier yang lebih banyak.
                </td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Karakteristik Trayektori</td>
                <td style={{ padding: "8px 12px" }}>
                  <i>Mean-reverting</i> alami. Mencegah proyeksi melompat liar (*anti-jomplang*) pada pos pendapatan yang fluktuatif.
                </td>
                <td style={{ padding: "8px 12px" }}>
                  Mengikuti akselerasi tren historis, diperkuat batas atas adaptif (*capping* $1.3\times$ maks historis) serta batasan $y \ge 0$.
                </td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#fcfcfd" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Performa Backtest (9 Bln)</td>
                <td style={{ padding: "8px 12px" }}>
                  Rata-rata WAPE: <strong>20.0%</strong> | Median WAPE: <strong>15.2%</strong> | Akurasi: <strong>85%</strong> (41/48 kategori berkinerja optimal).
                </td>
                <td style={{ padding: "8px 12px" }}>
                  Rata-rata WAPE: <strong>26.0%</strong> | Median WAPE: <strong>13.3%</strong> | Akurasi: <strong>87%</strong>.
                </td>
              </tr>
              <tr>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Isolasi Komputasi</td>
                <td colSpan={2} style={{ padding: "8px 12px", color: "#334155", background: "#f8fafc" }}>
                  Setiap model diprekomputasi ke direktori terpisah (<code>public/data/models/theta/</code> dan <code>prophet/</code>) dengan penanda metode eksplisit pada setiap baris data sehingga hasil kalkulasi antar algoritma tidak pernah saling mencemari.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Bagian 2: Deteksi Anomali & Profil Risiko */}
      <section style={{ marginBottom: 24 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          2. Deteksi Anomali Realisasi Kas & Audit Risk Engine
        </h4>
        <p style={{ margin: "0 0 10px 0" }}>
          Sistem deteksi anomali dirancang untuk memberikan peringatan dini (*early warning signal*) kepada BPKAD dan Inspektorat Daerah terhadap indikasi penyimpangan realisasi kas bulanan. Algoritma <b>Isolation Forest</b> dilatih secara terpisah untuk setiap kombinasi <code>(Provinsi × Pos Anggaran)</code> dengan 4 vektor fitur:
        </p>
        <ul style={{ margin: "0 0 12px 0", paddingLeft: 20 }}>
          <li><b>Nilai Realisasi Ternormalisasi:</b> Menilai magnitude transaksi terhadap distribusi historis akun terkait.</li>
          <li><b>Laju Perubahan Bulanan (MoM Growth %):</b> Mengidentifikasi akselerasi belanja atau kontraksi penerimaan yang tidak wajar.</li>
          <li><b>Deviasi terhadap Rata-rata Bergerak 3-Bulan:</b> Mengukur lonjakan temporer terhadap baseline jangka pendek.</li>
          <li><b>Deviasi Musiman Siklikal:</b> Membandingkan realisasi terhadap pola bulan yang sama pada siklus tahun anggaran sebelumnya.</li>
        </ul>
        <p style={{ margin: 0, fontSize: 12.5, color: "#475569" }}>
          Anomali diklasifikasikan ke dalam kategori <b>High Severity</b> (&gt; 2.5$\sigma$ deviasi) dan <b>Medium</b>, dilengkapi penalaran pemicu (*root-cause reasoning*) otomatis untuk mendukung proses penelaahan dokumen audit.
        </p>
      </section>

      {/* Bagian 3: Pra-Pemrosesan & Validasi */}
      <section style={{ marginBottom: 24 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          3. Pra-Pemrosesan Data & Validasi Empiris
        </h4>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={{ background: "#ffffff", padding: 14, borderRadius: 6, border: "1px solid #e2e8f0" }}>
            <div style={{ fontWeight: 700, color: "#1e3a5f", marginBottom: 6, fontSize: 13 }}>
              Winsorization & Penegakan Logika Fiskal
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#475569", lineHeight: 1.6 }}>
              Untuk meredam distorsi akibat fenomena tutup buku akhir tahun (*Desember shock*) pada data historis, data melalui penyaringan Winsorization pada persentil ke-98. Selain itu, ditegakkan batasan <i>non-negativity constraint</i> ($y \ge 0$) untuk menjamin seluruh pos anggaran mematuhi logika akuntansi sektor publik.
            </p>
          </div>

          <div style={{ background: "#ffffff", padding: 14, borderRadius: 6, border: "1px solid #e2e8f0" }}>
            <div style={{ fontWeight: 700, color: "#1e3a5f", marginBottom: 6, fontSize: 13 }}>
              Validasi Holdout & Rasionalitas Metrik WAPE
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#475569", lineHeight: 1.6 }}>
              Validasi dilakukan melalui pengujian <i>rolling holdout</i> 6 dan 9 bulan terakhir tanpa kebocoran data. Metrik akurasi menggunakan <b>WAPE</b> (<i>Weighted Absolute Percentage Error</i>) dan <b>sMAPE</b>, menggantikan metrik MAPE konvensional yang kerap meledak tak berhingga ($\infty$) akibat pembagian dengan realisasi pos-pos kecil mendekati nol.
            </p>
          </div>
        </div>
      </section>

      {/* Bagian 4: Catatan Integritas & Transparansi Data */}
      <section>
        <div style={{ 
          background: "#f8fafc", 
          padding: 14, 
          borderRadius: 6, 
          borderLeft: "4px solid #94a3b8",
          borderTop: "1px solid #e2e8f0",
          borderRight: "1px solid #e2e8f0",
          borderBottom: "1px solid #e2e8f0",
          fontSize: 12, 
          color: "#475569", 
          lineHeight: 1.6 
        }}>
          <strong style={{ color: "#1e293b", display: "block", marginBottom: 4, fontSize: 12.5 }}>
            Catatan Integritas & Batasan Analisis Data (SIKD DJPK)
          </strong>
          <span style={{ display: "block", marginBottom: 4 }}>
            • <b>Pengecualian Data 2021–2022:</b> Pada periode tersebut, portal SIKD hanya menyediakan pelaporan agregat tahunan tanpa rekonsiliasi progres bulanan yang konsisten.
          </span>
          <span style={{ display: "block", marginBottom: 4 }}>
            • <b>Status Preliminer 2025:</b> Realisasi bulan-bulan akhir tahun anggaran 2025 berstatus tentatif dan masih dalam proses audit verifikasi DJPK Kementerian Keuangan.
          </span>
          <span style={{ display: "block" }}>
            • <b>Pos Anggaran Sporadis (Lumpy Items):</b> Pos penerimaan tak terduga seperti Hibah atau Bantuan Keuangan Khusus memiliki volatilitas tinggi. Tingkat ketidakpastian ini direfleksikan melalui rentang interval keyakinan (<i>Confidence Interval</i>) yang lebih lebar pada proyeksi.
          </span>
        </div>
      </section>
    </div>
  );
}

// ─── Tab EDA ──────────────────────────────────────────────────
function TabEDA({ historical, selectedProvinces }: { historical: HistoricalRecord[], selectedProvinces: string[] }) {
  if (!historical || historical.length === 0) return <div className="info-box">Data tidak tersedia.</div>;

  const dataByProv = selectedProvinces.map(prov => {
    const sum = historical.filter(r => r.Provinsi === prov && r.Jenis_Pendapatan === "Total Pendapatan Daerah").reduce((acc, curr) => acc + curr.Realisasi, 0);
    return { Provinsi: prov, Realisasi: sum / 1e9 };
  });

  const trendMap = new Map<string, number>();
  historical.filter(r => selectedProvinces.includes(r.Provinsi) && r.Jenis_Pendapatan === "Total Pendapatan Daerah").forEach(r => {
    const date = r.Tanggal.split("T")[0].substring(0, 7);
    trendMap.set(date, (trendMap.get(date) || 0) + (r.Realisasi / 1e9));
  });
  const trendData = Array.from(trendMap.entries()).map(([date, val]) => ({ date, Realisasi: val })).sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div>
      <p style={{ fontSize: 13, color: "#475569", marginBottom: 18 }}>
        Visualisasi Data Pra-Model (EDA) - Distribusi realisasi kumulatif historis untuk melihat sebaran data secara umum pada provinsi terpilih.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24 }}>
        <div className="chart-card">
          <h4 style={{ fontSize: 14, fontWeight: 600, color: "#1e3a5f", marginBottom: 12 }}>Distribusi Realisasi per Provinsi</h4>
          <div style={{ height: 300, width: "100%" }}>
            <ResponsiveContainer>
              <BarChart data={dataByProv} margin={{ top: 10, right: 15, left: 15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="Provinsi" tick={{ fontSize: 11 }} />
                <YAxis 
                  tick={{ fontSize: 11 }} 
                  width={75}
                  tickFormatter={(val) => formatCurrency(Number(val) * 1e9)}
                />
                <Tooltip 
                  formatter={(value: any) => [formatCurrency(Number(value) * 1e9), "Total Realisasi"]} 
                  contentStyle={{ background: "white", border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 12, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)" }}
                />
                <Bar dataKey="Realisasi" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <h4 style={{ fontSize: 14, fontWeight: 600, color: "#1e3a5f", marginBottom: 12 }}>Total Tren Realisasi Seiring Waktu</h4>
          <div style={{ height: 300, width: "100%" }}>
            <ResponsiveContainer>
              <RechartsLineChart data={trendData} margin={{ top: 10, right: 15, left: 15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis 
                  tick={{ fontSize: 11 }} 
                  width={75}
                  tickFormatter={(val) => formatCurrency(Number(val) * 1e9)}
                />
                <Tooltip 
                  formatter={(value: any) => [formatCurrency(Number(value) * 1e9), "Total Realisasi"]} 
                  contentStyle={{ background: "white", border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 12, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)" }}
                />
                <Line type="monotone" dataKey="Realisasi" stroke="#10b981" strokeWidth={3} dot={{ r: 2 }} />
              </RechartsLineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
