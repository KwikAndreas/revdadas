import { formatCurrency } from "@/lib/utils";
import { Wallet, TrendingUp, AlertTriangle, ShieldAlert } from "lucide-react";

interface KPICardsProps {
  totalRevenue: number;
  forecastTotal: number;
  anomalyPct: number;
  anomalyCount: number;
  potentialLoss: number;
  forecastMonths: number;
  accuracyText: string;
}

export default function KPICards({
  totalRevenue,
  forecastTotal,
  anomalyPct,
  anomalyCount,
  potentialLoss,
  forecastMonths,
  accuracyText,
}: KPICardsProps) {
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
        <div className="kpi-sub kpi-sub--blue">{accuracyText}</div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <AlertTriangle size={14} /> RISIKO FRAUD/ANOMALI
        </div>
        <div className="kpi-value kpi-value--orange">
          {anomalyPct.toFixed(1)}%
        </div>
        <div className="kpi-sub kpi-sub--gray">
          {anomalyCount} records deteksi
        </div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title" style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <ShieldAlert size={14} /> REVENUE LOSS DETEKSI
        </div>
        <div className="kpi-value kpi-value--red">
          {formatCurrency(potentialLoss)}
        </div>
        <div className="kpi-sub kpi-sub--red">Tindakan Diperlukan</div>
      </div>
    </div>
  );
}
