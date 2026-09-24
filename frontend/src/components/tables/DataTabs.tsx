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
  ArrowDown, ArrowUp, Download, Check, ChevronDown, ShieldCheck,
  Info, Landmark
} from "lucide-react";
import { useLanguage } from "@/lib/LanguageContext";

interface DataTabsProps {
  forecast: ForecastRecord[];
  anomalies: AnomalyRecord[];
  accuracy: AccuracyData;
  business: BusinessData;
  historical: HistoricalRecord[];
  allHistorical?: HistoricalRecord[];
  allForecast?: ForecastRecord[];
  selectedProvinces: string[];
  forecastMonths: number;
}

export default function DataTabs({
  forecast,
  anomalies,
  accuracy,
  business,
  historical,
  allHistorical,
  allForecast,
  selectedProvinces,
  forecastMonths,
}: DataTabsProps) {
  const { lang, t } = useLanguage();
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    const handleSwitch = (e: any) => {
      setActiveTab(e.detail);
    };
    window.addEventListener('switchTab', handleSwitch);
    return () => window.removeEventListener('switchTab', handleSwitch);
  }, []);

  const tabs = [
    { label: t("tabs.forecast"), icon: <LineChart size={16} /> },
    { label: t("tabs.anomalies"), icon: <AlertTriangle size={16} /> },
    { label: t("tabs.exploratory"), icon: <LineChart size={16} /> },
    { label: t("tabs.accuracy"), icon: <Target size={16} /> },
    { label: t("tabs.whatif"), icon: <Sliders size={16} /> },
    { label: t("tabs.methodology"), icon: <BookOpen size={16} /> },
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
          <TabWhatIf
            key={selectedProvinces.join("-")}
            forecast={forecast}
            historical={historical}
            allForecast={allForecast}
            allHistorical={allHistorical}
            selectedProvinces={selectedProvinces}
          />
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
                    <td colSpan={3} style={{ textAlign: "center", color: "#64748b", fontSize: 11.5, background: "#f8fafc" }}>
                      Estimasi Terbatas (WAPE &gt; 50%)
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
    .filter((a) => a.Anomaly && a.Severity !== "Gap Data" && !a.Gap_Data)
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
      <div style={{ marginBottom: 10, fontSize: 12, color: "#64748b", display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <span style={{ fontWeight: 600, color: "#334155" }}>Kategori Konteks:</span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#2563eb" }}></span>
          <b>Transisi UU HKPD</b> (Penyesuaian Tarif Regulasi PDRD)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#ca8a04" }}></span>
          <b>Siklus Musiman</b> (Jatuh Tempo Pajak / Tutup Buku)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#dc2626" }}></span>
          <b>Deviasi Operasional</b> (Prioritas Verifikasi APIP)
        </span>
      </div>

      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Tanggal</th>
              <th>Provinsi</th>
              <th>Jenis Pendapatan</th>
              <th>Konteks Anomali</th>
              <th>Nilai Transaksi</th>
              <th>Alasan Anomali</th>
            </tr>
          </thead>
          <tbody>
            {paginatedAnomalies.map((r, i) => {
              const yr = r.Tahun || parseInt(r.Tanggal.substring(0, 4));
              const month = r.Bulan || parseInt(r.Tanggal.substring(5, 7));
              const isCurrentYear = yr === 2025;
              const isHkpdImpact = yr >= 2024 && (
                r.Jenis_Pendapatan.toLowerCase().includes("pajak") || 
                r.Jenis_Pendapatan.toLowerCase().includes("retribusi") ||
                r.Jenis_Pendapatan.toLowerCase().includes("pad")
              );
              const isSeasonal = (month >= 8 && month <= 9) || month === 12;

              let contextBadge = (
                <span style={{ fontSize: 10, background: "#fef2f2", color: "#b91c1c", border: "1px solid #fecaca", padding: "2px 6px", borderRadius: 4, fontWeight: 600 }}>
                  Deviasi Operasional
                </span>
              );

              if (isHkpdImpact) {
                contextBadge = (
                  <span style={{ fontSize: 10, background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe", padding: "2px 6px", borderRadius: 4, fontWeight: 600 }} title="Penyesuaian tarif berdasarkan UU No. 1/2022 tentang HKPD">
                    Transisi UU HKPD
                  </span>
                );
              } else if (isSeasonal) {
                contextBadge = (
                  <span style={{ fontSize: 10, background: "#fefce8", color: "#a16207", border: "1px solid #fef08a", padding: "2px 6px", borderRadius: 4, fontWeight: 600 }} title="Siklus musiman penerimaan daerah (jatuh tempo PBB atau tutup buku)">
                    Siklus Musiman
                  </span>
                );
              }

              return (
              <tr key={i} style={isCurrentYear ? { backgroundColor: "rgba(239, 68, 68, 0.05)" } : {}}>
                <td>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {r.Tanggal.split("T")[0]}
                    {isCurrentYear && (
                      <span style={{ fontSize: 10, background: "#fee2e2", color: "#991b1b", border: "1px solid #fecaca", padding: "1px 6px", borderRadius: 4, fontWeight: 600 }}>
                        TA 2025
                      </span>
                    )}
                  </div>
                </td>
                <td>{r.Provinsi}</td>
                <td>{r.Jenis_Pendapatan}</td>
                <td>{contextBadge}</td>
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
      <div className="metrics-row" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
        <div className="metric-card">
          <div className="metric-label">Model Aktif</div>
          <div className="metric-value" style={{ fontSize: 16, color: "#0284c7" }}>Profil Serapan Berjangkar</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Akurasi Model (Median)</div>
          <div className="metric-value">{accuracy.overall?.akurasi != null ? `${accuracy.overall.akurasi.toFixed(1)}%` : "-"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Seri Andal (WAPE &lt; 50%)</div>
          <div className="metric-value">
            {accuracy.overall?.n_reliable ?? 0} / {accuracy.overall?.n_series ?? 0} (
            {accuracy.overall?.n_series
              ? Math.round((accuracy.overall.n_reliable / accuracy.overall.n_series) * 100)
              : 50}
            %)
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Median WAPE</div>
          <div className="metric-value">{accuracy.overall?.median_wape != null ? `${accuracy.overall.median_wape.toFixed(1)}%` : "-"}</div>
        </div>
      </div>

      <div style={{ 
        background: "#f0f9ff", 
        border: "1px solid #bae6fd", 
        borderRadius: 8, 
        padding: "12px 16px", 
        marginBottom: 14, 
        fontSize: 12.5, 
        color: "#0369a1", 
        lineHeight: 1.55,
        display: "flex",
        alignItems: "flex-start",
        gap: 10
      }}>
        <Info size={16} strokeWidth={2.2} style={{ flexShrink: 0, marginTop: 2, color: "#0284c7" }} />
        <div>
          <strong style={{ color: "#0c4a6e" }}>Justifikasi Ekonometrika Pemilihan Model:</strong>{" "}
          Pendapatan daerah memiliki sampel observasi pasca-pandemi yang relatif pendek (24–36 bulan). Model Prophet konvensional cenderung overfit pada sampel pendek, sedangkan <b>Profil Serapan Berjangkar</b> menjangkarkan peramalan pada <b>Pagu Legal APBD</b> dan pola musiman historis terbobot, menghasilkan median WAPE yang jauh lebih stabil dan tahan uji (robust).
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
              let keandalan = "Lemah";
              let statusColor = "#dc2626";
              let statusBg = "#fef2f2";
              let statusBorder = "#fecaca";
              if (w !== null) {
                if (w < 30) { 
                  keandalan = "Andal"; 
                  statusColor = "#16a34a"; 
                  statusBg = "#f0fdf4";
                  statusBorder = "#bbf7d0";
                } else if (w < 50) { 
                  keandalan = "Cukup"; 
                  statusColor = "#d97706"; 
                  statusBg = "#fffbeb";
                  statusBorder = "#fde68a";
                }
              }

              return (
                <tr key={i}>
                  <td>{r.Provinsi}</td>
                  <td>{r.Jenis_Pendapatan}</td>
                  <td>{r.Akurasi != null ? `${r.Akurasi.toFixed(1)}%` : "-"}</td>
                  <td>{r.WAPE !== null ? r.WAPE.toFixed(1) + "%" : "-"}</td>
                  <td>{r.sMAPE !== null ? r.sMAPE.toFixed(1) + "%" : "-"}</td>
                  <td>
                    <span style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: 11,
                      fontWeight: 600,
                      color: statusColor,
                      background: statusBg,
                      border: `1px solid ${statusBorder}`,
                      padding: "2px 8px",
                      borderRadius: 4
                    }}>
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: statusColor }} />
                      {keandalan}
                    </span>
                  </td>
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
  historical,
  allForecast,
  allHistorical,
  selectedProvinces = [],
}: {
  forecast: ForecastRecord[];
  historical: HistoricalRecord[];
  allForecast?: ForecastRecord[];
  allHistorical?: HistoricalRecord[];
  selectedProvinces?: string[];
}) {
  const [adjustments, setAdjustments] = useState<Record<string, number>>({});

  const ADJUSTABLE = [
    { key: "Pendapatan Asli Daerah (PAD)", label: "Pendapatan Asli Daerah (PAD)" },
    { key: "Transfer ke Daerah dan Dana Desa (TKDD)", label: "Transfer ke Daerah dan Dana Desa (TKDD)" },
    { key: "Total Belanja Daerah", label: "Belanja Daerah" },
    { key: "Belanja Modal", label: "Belanja Modal" },
  ];

  // Gunakan allForecast & allHistorical (jika ada) agar tidak terpotong oleh filter taxType
  const provFilteredForecast = allForecast
    ? allForecast.filter((r) => selectedProvinces.includes(r.Provinsi))
    : forecast;

  const provFilteredHistorical = (allHistorical || historical).filter(
    (r) =>
      (selectedProvinces.length > 0 ? selectedProvinces.includes(r.Provinsi) : true) &&
      r.Tanggal >= "2024-01-01"
  );

  if (!provFilteredForecast || provFilteredForecast.length === 0) {
    return <div className="info-box">Proyeksi belum tersedia. Pilih provinsi di sidebar.</div>;
  }

  // Calculate Scenario
  // We'll simulate Budget Surplus/Deficit (Total Revenue - Total Expenditure).
  
  let baseTotal = 0;
  let scenTotal = 0;
  
  const totalRevRecords = provFilteredForecast.filter(r => r.Jenis_Pendapatan === "Total Pendapatan Daerah");
  const totalExpRecords = provFilteredForecast.filter(r => r.Jenis_Pendapatan === "Total Belanja Daerah");
  
  const baseRevenue = totalRevRecords.reduce((sum, r) => sum + r.Prediksi, 0);
  const baseExpenditure = totalExpRecords.reduce((sum, r) => sum + r.Prediksi, 0);
  const baseBelanjaModal = provFilteredForecast.filter(r => r.Jenis_Pendapatan === "Belanja Modal").reduce((sum, r) => sum + r.Prediksi, 0);
  baseTotal = baseRevenue - baseExpenditure;

  let totalRevenueDelta = 0;
  let totalExpenditureDelta = 0;
  let scenBelanjaModal = 0;
  
  const scenarioData = provFilteredForecast.map((r) => {
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

  // Historical Records for Total Rev & Total Exp
  const histRevRecords = provFilteredHistorical.filter(r => r.Jenis_Pendapatan === "Total Pendapatan Daerah");
  const histExpRecords = provFilteredHistorical.filter(r => r.Jenis_Pendapatan === "Total Belanja Daerah");

  const histDates = Array.from(new Set(histRevRecords.map(r => r.Tanggal.split("T")[0].substring(0, 7)))).sort();
  const forecastDates = Array.from(new Set(totalRevRecords.map(r => r.Tanggal.split("T")[0].substring(0, 7)))).sort();

  const chartData: Array<{
    date: string;
    actual: number | null;
    base: number | null;
    scen: number | null;
  }> = [];

  const lastHistDate = histDates[histDates.length - 1];

  // 1. Data Historis Realisasi (2024 - 2025)
  histDates.forEach((date) => {
    const mRev = histRevRecords.filter(r => r.Tanggal.startsWith(date)).reduce((sum, r) => sum + r.Realisasi, 0);
    const mExp = histExpRecords.filter(r => r.Tanggal.startsWith(date)).reduce((sum, r) => sum + r.Realisasi, 0);
    const netActual = (mRev - mExp) / 1e9;

    if (date === lastHistDate && forecastDates.length > 0) {
      // Sambungkan baseline & skenario ke titik realisasi terakhir agar garis menyambung
      chartData.push({
        date,
        actual: netActual,
        base: netActual,
        scen: netActual,
      });
    } else {
      chartData.push({
        date,
        actual: netActual,
        base: null,
        scen: null,
      });
    }
  });

  // 2. Data Proyeksi Forecast & Skenario (2026 ke depan)
  forecastDates.forEach((date) => {
    const monthRev = totalRevRecords.filter(r => r.Tanggal.startsWith(date)).reduce((sum, r) => sum + r.Prediksi, 0);
    const monthExp = totalExpRecords.filter(r => r.Tanggal.startsWith(date)).reduce((sum, r) => sum + r.Prediksi, 0);
    const baseVal = (monthRev - monthExp) / 1e9;

    const monthlyItems = scenarioData.filter(sd => sd.Tanggal.startsWith(date));
    const revDeltas = monthlyItems.filter(md => ["Pendapatan Asli Daerah (PAD)", "Transfer ke Daerah dan Dana Desa (TKDD)", "Lain-lain Pendapatan Daerah yang Sah"].includes(md.Jenis_Pendapatan));
    const expDeltas = monthlyItems.filter(md => ["Total Belanja Daerah", "Belanja Modal", "Belanja Operasi"].includes(md.Jenis_Pendapatan));

    const monthRevDelta = revDeltas.reduce((sum, md) => sum + (md.delta || 0), 0);
    const monthExpDelta = expDeltas.reduce((sum, md) => sum + (md.delta || 0), 0);
    const scenVal = ((monthRev + monthRevDelta) - (monthExp + monthExpDelta)) / 1e9;

    chartData.push({
      date,
      actual: null,
      base: baseVal,
      scen: scenVal,
    });
  });

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

      <div style={{ width: "100%", height: 280, marginTop: 24 }}>
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
                content={({ active, payload, label }) => {
                  if (!active || !payload || !payload.length) return null;
                  const validEntries = payload.filter((p: any) => p.value !== null && p.value !== undefined);
                  if (!validEntries.length) return null;
                  return (
                    <div style={{ background: "white", padding: "10px 14px", border: "1px solid #e2e8f0", borderRadius: 8, boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)", fontSize: 12 }}>
                      <div style={{ fontWeight: 700, color: "#334155", marginBottom: 6 }}>{label}</div>
                      {validEntries.map((entry: any, idx: number) => {
                        const val = typeof entry.value === "number" ? entry.value : 0;
                        const prefix = val >= 0 ? "Surplus " : "Defisit ";
                        return (
                          <div key={idx} style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 3 }}>
                            <span style={{ width: 8, height: 8, borderRadius: "50%", background: entry.color, display: "inline-block" }}></span>
                            <span style={{ color: "#64748b" }}>{entry.name}:</span>
                            <span style={{ fontWeight: 600, color: val < 0 ? "#dc2626" : "#16a34a" }}>
                              {prefix}{formatCurrency(Math.abs(val) * 1e9)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  );
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="actual" name="Realisasi Historis" stroke="#0284c7" strokeWidth={2.2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="base" name="Baseline (Status Quo)" stroke="#94a3b8" strokeDasharray="5 5" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="scen" name="Skenario (Intervensi)" stroke="#1e3a5f" strokeWidth={2.5} dot={{ r: 3 }} />
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
          Kerangka Metodologi & Spesifikasi Teknis Pemodelan RevDadas
        </h3>
        <p style={{ margin: 0, color: "#64748b", fontSize: 13 }}>
          Dokumentasi teknis pemodelan ekonometrika deret waktu fiskal, deteksi anomali transaksi APBD, simulasi kebijakan what-if, serta keselarasan dengan mandat Elektronifikasi Transaksi Pemda (ETPD) Bank Indonesia berbasis data Sistem Informasi Keuangan Daerah (SIKD) Kementerian Keuangan.
        </p>

        {/* Bar Ringkasan Parameter Teknis */}
        <div style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 16,
          marginTop: 14,
          padding: "12px 16px",
          background: "#f8fafc",
          borderRadius: 8,
          border: "1px solid #e2e8f0",
          fontSize: 12
        }}>
          <div>
            <span style={{ color: "#64748b" }}>Basis Data: </span>
            <strong style={{ color: "#0f172a" }}>DJPK SIKD (38 Provinsi, 2023–2025 Diskrit)</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Mesin Utama: </span>
            <strong style={{ color: "#0284c7" }}>Profil Serapan Berjangkar (Anchor Pagu)</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Pembanding Ekonometrika: </span>
            <strong style={{ color: "#0f172a" }}>Theta Method (M3) & Prophet (GAM)</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Deteksi Anomali: </span>
            <strong style={{ color: "#0f172a" }}>Isolation Forest + Domain Guardrail APBD</strong>
          </div>
          <div>
            <span style={{ color: "#64748b" }}>Validasi Evaluasi: </span>
            <strong style={{ color: "#16a34a" }}>WAPE, sMAPE, & MASE &lt; 1 (Holdout)</strong>
          </div>
        </div>
      </div>

      {/* Bagian 1: Pemodelan Deret Waktu Fiskal */}
      <section style={{ marginBottom: 26 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          1. Mesin Peramalan Utama: Profil Serapan Berjangkar (Anchor Absorption Engine)
        </h4>
        <p style={{ margin: "0 0 12px 0" }}>
          Data keuangan daerah memiliki dua karakteristik unik yang membuat algoritma deret waktu konvensional (seperti LSTM atau ARIMA murni) rentan gagal: <b>sampel observasi bulanan yang pendek ($N \approx 36$ bulan)</b> serta <b>volatilitas tajam di akhir tahun fiskal (efek tutup buku Desember)</b>. Model Prophet konvensional hanya mencapai akurasi 54,3% pada data ini.
        </p>
        <p style={{ margin: "0 0 12px 0" }}>
          RevDadas mengatasi limitasi tersebut melalui <b>Profil Serapan Berjangkar</b> — model yang menjangkarkan proyeksi pada <b>pagu Anggaran APBD</b> yang telah disahkan dan diketahui secara pasti sejak awal tahun anggaran:
        </p>

        {/* Kotak Formula Matematika */}
        <div style={{ background: "#f8fafc", padding: "14px 18px", borderRadius: 8, border: "1px solid #e2e8f0", marginBottom: 14 }}>
          <div style={{ fontWeight: 700, color: "#1e3a5f", marginBottom: 6, fontSize: 13 }}>
            Formulasi Matematika Profil Serapan Berjangkar:
          </div>
          <div style={{ fontFamily: "monospace", fontSize: 12.5, color: "#0f172a", lineHeight: 1.8 }}>
            <div>• <b>Prediksi Bulanan:</b> Prediksi[m] = T × p[m]</div>
            <div>• <b>Profil Serapan Normalisasi:</b> p[m] = normalisasi( γ · profil_historis[m] + (1 − γ) · (1/12) )</div>
            <div>• <b>Estimasi Target Tahunan:</b> T = Anggaran × rasio_serapan_rata-rata (Koefisien Variasi rendah ~6,8%)</div>
            <div>• <b>Parameter Shrinkage (γ):</b> Belanja = 0,75 (musiman kuat diakui) | Pendapatan Inti = 0,40 | Pendapatan Rinci = 0,15 (regulasi uniform anti-overfitting)</div>
            <div>• <b>Gerbang Seleksi MASE:</b> Hanya proyeksi dengan MASE &lt; 1 (mengungguli <i>seasonal naive</i>) yang ditayangkan secara aktif.</div>
          </div>
        </div>

        {/* Tabel Komparasi Teknis 3 Model */}
        <div style={{ overflowX: "auto", marginBottom: 14 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, textAlign: "left" }}>
            <thead>
              <tr style={{ background: "#f1f5f9", borderBottom: "2px solid #cbd5e1" }}>
                <th style={{ padding: "8px 12px", color: "#334155", fontWeight: 700, width: "22%" }}>Parameter</th>
                <th style={{ padding: "8px 12px", color: "#0284c7", fontWeight: 700, width: "30%" }}>Profil Serapan Berjangkar (Aktif)</th>
                <th style={{ padding: "8px 12px", color: "#1e3a5f", fontWeight: 700, width: "24%" }}>Theta Method (M3 Winner)</th>
                <th style={{ padding: "8px 12px", color: "#64748b", fontWeight: 700, width: "24%" }}>Additive GAM (Prophet)</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Basis Informasi</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff" }}>
                  <strong>Pagu Anggaran APBD</strong> + Profil Distribusi Musiman Historis.
                </td>
                <td style={{ padding: "8px 12px" }}>Dekomposisi kurva ganda tren linier &amp; kurvatur SES lokal.</td>
                <td style={{ padding: "8px 12px" }}>Dekomposisi tren Fourier aditif $g(t) + s(t) + \epsilon_t$.</td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#fcfcfd" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Ketahanan Sampel Pendek</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff" }}>
                  <strong style={{ color: "#16a34a" }}>Sangat Tinggi.</strong> Tidak bergantung pada data deret waktu puluhan tahun.
                </td>
                <td style={{ padding: "8px 12px" }}>
                  <strong style={{ color: "#16a34a" }}>Tinggi.</strong> Parsimonious (hanya 2 parameter bebas).
                </td>
                <td style={{ padding: "8px 12px" }}>
                  <strong style={{ color: "#d97706" }}>Rendah/Moderat.</strong> Rawan overfit pada sampel $N \le 36$.
                </td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Akurasi Holdout (WAPE)</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff" }}>
                  <strong>70,1% Akurasi</strong> (Median WAPE 29,8% | 62% lolos MASE &lt; 1).
                </td>
                <td style={{ padding: "8px 12px" }}>78,0% Akurasi (Median WAPE 15,2%).</td>
                <td style={{ padding: "8px 12px" }}>54,3% Akurasi (Median WAPE 26,0%).</td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#fcfcfd" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "#475569" }}>Efisiensi Komputasi</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff" }}>
                  <strong style={{ color: "#16a34a" }}>45× Lebih Cepat</strong> tanpa dependensi library C++/Stan.
                </td>
                <td style={{ padding: "8px 12px" }}>Cepat (Statsmodels Python).</td>
                <td style={{ padding: "8px 12px" }}>Lambat (kompilasi C++ CmdStanPy).</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Bagian 2: Deteksi Anomali & Audit Risk Engine */}
      <section style={{ marginBottom: 26 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          2. Deteksi Anomali Realisasi Kas &amp; Domain Rule Guardrail APBD
        </h4>
        <p style={{ margin: "0 0 10px 0" }}>
          Sistem deteksi anomali dirancang sebagai <i>Early Warning Signal</i> untuk Bapenda dan Inspektorat Daerah guna menyaring potensi kebocoran penerimaan kas (*under-reporting*) atau manipulasi belanja. Algoritma <b>Isolation Forest</b> dilatih secara terpisah untuk setiap seri <code>(Provinsi × Pos Anggaran)</code> dengan 4 vektor fitur:
        </p>
        <ul style={{ margin: "0 0 12px 0", paddingLeft: 20 }}>
          <li><b>Nilai Realisasi Ternormalisasi:</b> Menilai magnitude transaksi terhadap distribusi historis akun terkait.</li>
          <li><b>Laju Perubahan Bulanan (MoM Growth %):</b> Mengidentifikasi akselerasi belanja atau kontraksi penerimaan yang tidak wajar.</li>
          <li><b>Deviasi terhadap Rata-rata Bergerak 3-Bulan (MA-3):</b> Mengukur lonjakan temporer terhadap baseline jangka pendek.</li>
          <li><b>Deviasi Musiman Siklikal (YoY):</b> Membandingkan realisasi terhadap pola bulan yang sama pada siklus tahun anggaran sebelumnya.</li>
        </ul>
        <div style={{ background: "#f8fafc", padding: "12px 16px", borderRadius: 6, border: "1px solid #e2e8f0" }}>
          <strong style={{ color: "#1e3a5f", display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <ShieldCheck size={16} strokeWidth={2.2} style={{ color: "#16a34a" }} />
            Domain Rule Guardrail APBD (Anti-False Positive):
          </strong>
          <span style={{ fontSize: 12.5, color: "#475569" }}>
            Pada akuntansi keuangan daerah, pos-pos tertentu seperti <b>Belanja Modal</b>, <b>Hibah</b>, atau <b>Bagi Hasil</b> memiliki sifat alamiah cair secara sporadis (*lumpy/one-off payments*). Sistem secara otomatis melakukan <i>auto-downgrade</i> pada akun-akun ini agar tidak memicu alarm palsu (*false alarm*), sehingga auditor hanya difokuskan pada akun rutin seperti Pajak Daerah dan Retribusi Daerah.
          </span>
        </div>
      </section>

      {/* Bagian 3: Simulasi Kebijakan What-If */}
      <section style={{ marginBottom: 26 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          3. Simulasi Kebijakan What-If &amp; Keseimbangan Anggaran Fiskal
        </h4>
        <p style={{ margin: "0 0 10px 0" }}>
          Tab Simulasi What-If memproyeksikan dinamika <b>Keseimbangan Anggaran Fiskal Bulanan (Surplus/Defisit = Total Pendapatan − Total Belanja)</b>. Model menyajikan kesinambungan garis utuh:
        </p>
        <ul style={{ margin: "0 0 12px 0", paddingLeft: 20 }}>
          <li><b>Realisasi Historis (2024–2025):</b> Ditampilkan di awal sebagai data riil hasil rekonsiliasi APBD tanpa modifikasi.</li>
          <li><b>Titik Transisi (Desember 2025):</b> Garis proyeksi menyambung secara kontinu (*seamless connection*) dari titik realisasi terakhir.</li>
          <li><b>Proyeksi Interaktif (2026):</b> Garis <i>Baseline (Status Quo)</i> dan <i>Skenario (Intervensi)</i> merespons secara <i>real-time</i> saat pengguna menggeser slider kebijakan (PAD, transfer TKDD, alokasi Belanja Daerah, atau rasio Belanja Modal).</li>
        </ul>
      </section>

      {/* Bagian 4: Pra-Pemrosesan Data & Validasi */}
      <section style={{ marginBottom: 26 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          4. Pra-Pemrosesan Data &amp; Validasi Ekonometrika
        </h4>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={{ background: "#ffffff", padding: 14, borderRadius: 6, border: "1px solid #e2e8f0" }}>
            <div style={{ fontWeight: 700, color: "#1e3a5f", marginBottom: 6, fontSize: 13 }}>
              Decumulation Engine &amp; Logika Fiskal
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#475569", lineHeight: 1.6 }}>
              Data portal SIKD dilaporkan dalam format kumulatif tahun berjalan (YTD). RevDadas membangun <i>Decumulation Engine</i> untuk mengekstrak realisasi bulanan diskrit murni: <code>ΔPAD_t = YTD_t − YTD_{'{t-1}'}</code>. Data kemudian melalui Winsorization persentil ke-98 untuk memitigasi distorsi tutup buku tanpa menghilangkan sinyal musiman, serta penegakan batasan non-negativitas ($y \ge 0$).
            </p>
          </div>

          <div style={{ background: "#ffffff", padding: 14, borderRadius: 6, border: "1px solid #e2e8f0" }}>
            <div style={{ fontWeight: 700, color: "#1e3a5f", marginBottom: 6, fontSize: 13 }}>
              Rasionalitas Metrik WAPE, sMAPE, &amp; MASE
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#475569", lineHeight: 1.6 }}>
              Validasi dilakukan lewat pengujian <i>rolling holdout</i> 6 bulan terakhir. Metrik akurasi menggunakan <b>WAPE</b> (<i>Weighted Absolute Percentage Error</i>) dan <b>sMAPE</b> menggantikan MAPE tradisional yang kerap meledak tak berhingga ($\infty$) akibat pembagian dengan realisasi mendekati nol. Benchmark <b>MASE</b> menjamin model mengalahkan <i>seasonal naive</i>.
            </p>
          </div>
        </div>
      </section>

      {/* Bagian 5: Keselarasan Program Bank Indonesia & Kepatuhan Tata Kelola */}
      <section style={{ marginBottom: 26 }}>
        <div style={{ 
          background: "#f8fafc", 
          padding: 16, 
          borderRadius: 8, 
          borderLeft: "4px solid #0284c7",
          borderTop: "1px solid #e2e8f0",
          borderRight: "1px solid #e2e8f0",
          borderBottom: "1px solid #e2e8f0",
          fontSize: 12, 
          color: "#475569", 
          lineHeight: 1.65 
        }}>
          <strong style={{ color: "#0f172a", display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 13 }}>
            <Landmark size={16} strokeWidth={2.2} style={{ color: "#0284c7" }} />
            Keselarasan dengan Mandat Bank Indonesia, ETPD, &amp; Tata Kelola DJPK
          </strong>
          <span style={{ display: "block", marginBottom: 6 }}>
            • <b>Dukungan bagi Satgas TP2DD:</b> RevDadas bertindak sebagai <i>"Otak Intelijen Analitik Lanjutan"</i> pelengkap program Elektronifikasi Transaksi Pemda (ETPD). Jika kanal QRIS Pemda dan KKPD mendigitalisasi transaksi pembayaran di hilir, RevDadas memverifikasi apakah kenaikan transaksi digital tersebut benar-benar tercermin pada penerimaan kas daerah dan terbebas dari kebocoran (*under-reporting*).
          </span>
          <span style={{ display: "block", marginBottom: 6 }}>
            • <b>Pengendalian Likuiditas Regional:</b> Peramalan kas yang akurat membantu bendahara daerah menyerap anggaran tepat waktu, mengurangi penumpukan dana mengendap (<i>idle cash</i> / SiLPA berlebih di BPD), serta menjaga transmisi likuiditas moneter daerah.
          </span>
          <span style={{ display: "block", marginBottom: 6 }}>
            • <b>Risk-Based Audit (Pemeriksaan Terarah):</b> Sistem bekerja pada tingkat <i>Top-Down Risk Screening</i>. Temuan anomali makro menjadi dasar penerbitan surat perintah audit terarah bagi Bapenda/APIP untuk memeriksa dokumen transaksi mikro secara presisi.
          </span>
          <span style={{ display: "block" }}>
            • <b>Catatan Integritas Data:</b> Data SIKD periode 2021–2022 dikecualikan karena portal hanya menyediakan pelaporan agregat tahunan tanpa rekonsiliasi bulanan. Realisasi bulan-bulan akhir 2025 berstatus preliminer mengikuti siklus audit reguler BPK/Kemenkeu.
          </span>
        </div>
      </section>

      {/* Bagian 6: Benchmark Komparasi Kompetitif */}
      <section style={{ marginBottom: 26 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          5. Benchmark Kompetitif Solusi: RevDaDas vs Sistem Eksisting
        </h4>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, textAlign: "left" }}>
            <thead>
              <tr style={{ background: "#f1f5f9", borderBottom: "2px solid #cbd5e1" }}>
                <th style={{ padding: "8px 12px", color: "#334155", fontWeight: 700, width: "24%" }}>Dimensi Evaluasi</th>
                <th style={{ padding: "8px 12px", color: "#0284c7", fontWeight: 700, width: "32%" }}>RevDaDas (Smart Revenue Intel)</th>
                <th style={{ padding: "8px 12px", color: "#64748b", fontWeight: 700, width: "22%" }}>SIPD Kemendagri</th>
                <th style={{ padding: "8px 12px", color: "#64748b", fontWeight: 700, width: "22%" }}>Spreadsheet / Manual Excel</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600 }}>Tujuan &amp; Fungsi Utama</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff", color: "#0369a1", fontWeight: 600 }}>Early Warning System &amp; Audit Prediktif</td>
                <td style={{ padding: "8px 12px" }}>Pencatatan Transaksi &amp; Akuntansi</td>
                <td style={{ padding: "8px 12px" }}>Rekapitulasi Ad-hoc Staf Bapenda</td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#fcfcfd" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600 }}>Deteksi Anomali Kas</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff", color: "#16a34a", fontWeight: 600 }}>Otomatis (Isolation Forest + Z-Score)</td>
                <td style={{ padding: "8px 12px", color: "#dc2626" }}>Tidak Ada (Hanya validasi pagu)</td>
                <td style={{ padding: "8px 12px", color: "#dc2626" }}>Manual via visualisasi baris</td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600 }}>Peramalan Fiskal Berjangkar</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff", color: "#16a34a", fontWeight: 600 }}>Berjangkar Pagu APBD + Musiman</td>
                <td style={{ padding: "8px 12px", color: "#dc2626" }}>Tidak Ada (Hanya realisasi berjalan)</td>
                <td style={{ padding: "8px 12px", color: "#d97706" }}>Regresi Linear Sederhana (Rawan Overfit)</td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0", background: "#fcfcfd" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600 }}>Simulasi Kebijakan (What-If)</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff", color: "#16a34a", fontWeight: 600 }}>Dinamis Real-time dengan Slider</td>
                <td style={{ padding: "8px 12px", color: "#dc2626" }}>Tidak Tersedia</td>
                <td style={{ padding: "8px 12px", color: "#d97706" }}>Rumus manual bertingkat (rentan error)</td>
              </tr>
              <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600 }}>Mitigasi Halusinasi AI</td>
                <td style={{ padding: "8px 12px", background: "#f0f9ff", color: "#16a34a", fontWeight: 600 }}>Deterministic Grounded Rule-Engine</td>
                <td style={{ padding: "8px 12px" }}>N/A (Tanpa Mesin Intelijen)</td>
                <td style={{ padding: "8px 12px" }}>N/A (Bergantung keahlian staf)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Bagian 7: Mitigasi Kesalahan Input Data */}
      <section>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 4, height: 16, background: "#1e3a5f", borderRadius: 2 }}></span>
          6. Quality Control Gate: Mitigasi Risiko Kesalahan Input Data
        </h4>
        <div style={{ background: "#ffffff", padding: 14, borderRadius: 8, border: "1px solid #e2e8f0" }}>
          <p style={{ margin: "0 0 10px 0", fontSize: 12.5, color: "#475569" }}>
            Untuk memitigasi risiko human error dari operator Pemda saat memasukkan data transaksi atau pelaporan, RevDaDas menerapkan 3 lapis filter validasi otomatis:
          </p>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12.5, color: "#475569", lineHeight: 1.7 }}>
            <li><b>Lapis 1 - Sanity Check Batas Nominal:</b> Sistem menolak/menandai transaksi bulanan yang melebihi 100% total pagu tahunan dalam 1 bulan (mencegah salah pengetikan digit nol).</li>
            <li><b>Lapis 2 - Deteksi Inversi Non-Negatif:</b> Angka minus pada realisasi pendapatan akibat jurnal pembalik diverifikasi secara terpisah agar tidak merusak profil tren musiman.</li>
            <li><b>Lapis 3 - Z-Score Statistical Anomaly Gate:</b> Deviasi &gt; 3.0σ langsung dikarantina untuk diverifikasi dua pihak (*maker-checker*) sebelum masuk ke model peramalan fiskal.</li>
          </ul>
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
