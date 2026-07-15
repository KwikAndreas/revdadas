import { formatCurrency } from "@/lib/utils";
import { Wallet, TrendingUp, AlertTriangle, ShieldAlert, Percent } from "lucide-react";
import type { AnomalyRecord } from "@/lib/types";
import { useState } from "react";

interface KPICardsProps {
  totalRevenue: number;
  forecastTotal: number;
  anomalyPct: number;
  anomalyCount: number;
  potentialLoss: number;
  forecastMonths: number;
  accuracyText: string;
  kemandirianFiskal: number;
  anomalies: AnomalyRecord[];
}

export default function KPICards({
  totalRevenue,
  forecastTotal,
  anomalyPct,
  anomalyCount,
  potentialLoss,
  forecastMonths,
  accuracyText,
  kemandirianFiskal,
  anomalies,
}: KPICardsProps) {
  const [showModal, setShowModal] = useState(false);

  return (
    <div className="kpi-grid">
      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <Wallet size={14} /> TOTAL REVENUE (AKTUAL)
        </div>
        <div className="kpi-value kpi-value--dark">
          {formatCurrency(totalRevenue)}
        </div>
        <div className="kpi-sub kpi-sub--green">Real Data</div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <TrendingUp size={14} /> FORECAST {forecastMonths} BULAN
        </div>
        <div className="kpi-value kpi-value--red">
          {formatCurrency(forecastTotal)}
        </div>
        <div className="kpi-sub kpi-sub--gray">{accuracyText}</div>
      </div>

      <div 
        className="kpi-card animate-fade-in-up" 
        style={{ cursor: "pointer" }}
        onClick={() => setShowModal(true)}
      >
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <AlertTriangle size={14} /> SKOR RISIKO AUDIT (ANOMALI)
        </div>
        <div className="kpi-value kpi-value--orange">
          {anomalyPct.toFixed(1)}%
        </div>
        <div className="kpi-sub kpi-sub--blue" style={{ textDecoration: "underline" }}>
          {anomalyCount} records deteksi (Klik detail)
        </div>
      </div>

      {/* Modal Overlay for Anomaly Records */}
      {showModal && anomalies.length > 0 && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: "rgba(15, 23, 42, 0.5)",
          backdropFilter: "blur(4px)",
          zIndex: 9999,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 20
        }} onClick={() => setShowModal(false)}>
          <div style={{
            background: "white",
            borderRadius: 12,
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
            width: "100%",
            maxWidth: 600,
            maxHeight: "80vh",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden"
          }} onClick={(e) => e.stopPropagation()}>
            <div style={{ 
              padding: "16px 20px", 
              borderBottom: "1px solid #e2e8f0", 
              display: "flex", 
              justifyContent: "space-between",
              alignItems: "center",
              background: "#f8fafc"
            }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#1e293b", display: "flex", alignItems: "center", gap: 8 }}>
                <AlertTriangle size={18} color="#f59e0b" />
                Detail Anomali Terdeteksi ({anomalyCount} Records)
              </div>
              <button 
                onClick={() => setShowModal(false)}
                style={{
                  background: "transparent", border: "none", cursor: "pointer", 
                  fontSize: 20, color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center",
                  width: 32, height: 32, borderRadius: 16
                }}
                onMouseOver={(e) => e.currentTarget.style.background = "#e2e8f0"}
                onMouseOut={(e) => e.currentTarget.style.background = "transparent"}
              >
                &times;
              </button>
            </div>
            
            <div style={{ padding: 20, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, background: "#f1f5f9" }}>
              {anomalies.map((r, i) => (
                <div key={i} style={{ 
                  background: "white", 
                  padding: 16, 
                  borderRadius: 8, 
                  borderLeft: `4px solid ${r.Severity === 'Tinggi' ? '#ef4444' : '#f59e0b'}`,
                  boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <div style={{ fontWeight: 700, color: "#334155", fontSize: 14 }}>{r.Provinsi}</div>
                    <span style={{ 
                      fontSize: 11, 
                      fontWeight: 600, 
                      padding: "2px 8px", 
                      borderRadius: 12,
                      background: r.Severity === 'Tinggi' ? '#fee2e2' : '#fef3c7',
                      color: r.Severity === 'Tinggi' ? '#991b1b' : '#92400e'
                    }}>
                      {r.Severity}
                    </span>
                  </div>
                  
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 13 }}>
                    <div>
                      <div style={{ color: "#64748b", fontSize: 11, marginBottom: 2 }}>Jenis Pendapatan</div>
                      <div style={{ fontWeight: 500, color: "#0f172a" }}>{r.Jenis_Pendapatan}</div>
                    </div>
                    <div>
                      <div style={{ color: "#64748b", fontSize: 11, marginBottom: 2 }}>Kategori Anomali</div>
                      <div style={{ fontWeight: 600, color: r.Severity === 'Tinggi' ? '#dc2626' : '#d97706' }}>{r.Jenis_Fraud || 'Anomali'}</div>
                    </div>
                    <div style={{ gridColumn: "1 / -1" }}>
                      <div style={{ color: "#64748b", fontSize: 11, marginBottom: 2 }}>Alasan Deteksi</div>
                      <div style={{ color: "#475569" }}>{r.Alasan}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <ShieldAlert size={14} /> NILAI TRANSAKSI UNTUK DITINJAU
        </div>
        <div className="kpi-value kpi-value--red">
          {formatCurrency(potentialLoss)}
        </div>
        <div className="kpi-sub kpi-sub--red">Perlu Verifikasi Manual</div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <Percent size={14} /> KEMANDIRIAN FISKAL
        </div>
        <div className="kpi-value kpi-value--blue">
          {kemandirianFiskal.toFixed(1)}%
        </div>
        <div className="kpi-sub kpi-sub--green">Simulasi UU HKPD</div>
      </div>
    </div>
  );
}
