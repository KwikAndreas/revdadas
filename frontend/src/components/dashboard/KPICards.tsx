import { formatCurrency } from "@/lib/utils";
import { Wallet, TrendingUp, AlertTriangle, ShieldAlert, Percent } from "lucide-react";
import type { AnomalyRecord } from "@/lib/types";

interface KPICardsProps {
  totalRevenue: number;
  totalAnggaran?: number;
  targetPercentage?: number;
  forecastTotal: number;
  anomalyPct: number;
  anomalyCount: number;
  potentialLoss: number;
  forecastMonths: number;
  accuracyText: string;
  kemandirianFiskal: number;
  anomalies: AnomalyRecord[];
  selectedYear: number;
}

export default function KPICards({
  totalRevenue,
  totalAnggaran,
  targetPercentage,
  forecastTotal,
  anomalyPct,
  anomalyCount,
  potentialLoss,
  forecastMonths,
  accuracyText,
  kemandirianFiskal,
  anomalies,
  selectedYear,
}: KPICardsProps) {
  return (
    <div className="kpi-grid">
      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center", textTransform: "uppercase" }}>
          <Wallet size={14} /> TOTAL REVENUE KUMULATIF (AKTUAL)
        </div>
        <div className="kpi-value kpi-value--dark">
          {formatCurrency(totalRevenue)}
        </div>
        <div className="kpi-sub kpi-sub--green">
          {targetPercentage !== undefined 
            ? `${targetPercentage.toFixed(1)}% dari Target Tahunan` 
            : 'Real Data'}
        </div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <TrendingUp size={14} /> TOTAL FORECAST ({forecastMonths} BULAN KE DEPAN)
        </div>
        <div className="kpi-value kpi-value--red">
          {formatCurrency(forecastTotal)}
        </div>
        <div className="kpi-sub kpi-sub--gray">{accuracyText}</div>
      </div>

      <div 
        className="kpi-card animate-fade-in-up" 
        style={{ cursor: "pointer" }}
        onClick={() => {
          const detailElement = document.getElementById("anomali-details-section");
          if (detailElement) detailElement.scrollIntoView({ behavior: "smooth" });
        }}
      >
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center", textTransform: "uppercase" }}>
          <AlertTriangle size={14} /> RISIKO ANOMALI
        </div>
        <div className="kpi-value kpi-value--orange">
          {anomalyPct.toFixed(1)}%
        </div>
        <div className="kpi-sub kpi-sub--blue" style={{ textDecoration: "underline" }}>
          {anomalyCount} records dianalisis (Lihat Tabel)
        </div>
      </div>

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
