import { formatCurrency } from "@/lib/utils";
import { Wallet, TrendingUp, AlertTriangle, ShieldAlert, Percent } from "lucide-react";
import type { AnomalyRecord } from "@/lib/types";
import { useLanguage } from "@/lib/LanguageContext";

interface KPICardsProps {
  revenueLabel: string;
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
  revenueLabel,
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
  const { lang, t } = useLanguage();

  return (
    <div className="kpi-grid">
      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title">
          <span
            title={revenueLabel}
            style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}
          >
            {revenueLabel}
          </span>
          <Wallet size={14} color="#0284c7" style={{ flexShrink: 0 }} />
        </div>
        <div className="kpi-value kpi-value--dark">
          {formatCurrency(totalRevenue)}
        </div>
        <div className="kpi-sub kpi-sub--green">
          {targetPercentage !== undefined 
            ? t("kpi.target_annual", { pct: targetPercentage.toFixed(1) })
            : t("kpi.actual_realization")}
        </div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title">
          <span>{t("kpi.proyeksi", { months: forecastMonths })}</span>
          <TrendingUp size={14} color="#0284c7" />
        </div>
        <div className="kpi-value kpi-value--dark">
          {formatCurrency(forecastTotal)}
        </div>
        <div className="kpi-sub kpi-sub--gray">{accuracyText}</div>
      </div>

      <div 
        className="kpi-card animate-fade-in-up" 
        style={{ cursor: anomalyCount > 0 ? "pointer" : "default" }}
        title={anomalyCount > 0 
          ? (lang === "en" ? "Click to view anomaly details table" : "Klik untuk melihat rincian tabel anomali") 
          : (lang === "en" ? "No critical anomalies on this filter" : "Tidak ada anomali kritis pada filter ini")}
        onClick={() => {
          if (anomalyCount > 0) {
            window.dispatchEvent(new CustomEvent("switchTab", { detail: 1 }));
            const detailElement = document.getElementById("data-logs") || document.getElementById("anomali-details-section");
            if (detailElement) detailElement.scrollIntoView({ behavior: "smooth" });
          }
        }}
      >
        <div className="kpi-title">
          <span>{t("kpi.risiko_deviasi")}</span>
          <AlertTriangle size={14} color={anomalyCount > 0 ? "#d97706" : "#16a34a"} />
        </div>
        <div className={`kpi-value ${anomalyCount > 0 ? "kpi-value--orange" : "kpi-value--green"}`}>
          {anomalyCount > 0 ? `${anomalyPct.toFixed(1)}%` : "0.0%"}
        </div>
        <div 
          className={`kpi-sub ${anomalyCount > 0 ? "kpi-sub--blue" : "kpi-sub--green"}`} 
          style={anomalyCount > 0 ? { textDecoration: "underline" } : undefined}
        >
          {anomalyCount > 0 
            ? t("kpi.deviasi_view", { count: anomalyCount }) 
            : t("kpi.normal_status")}
        </div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title">
          <span>{t("kpi.transaksi_tinjauan")}</span>
          <ShieldAlert size={14} color={potentialLoss > 0 ? "#dc2626" : "#16a34a"} />
        </div>
        <div className={`kpi-value ${potentialLoss > 0 ? "kpi-value--red" : "kpi-value--green"}`}>
          {formatCurrency(potentialLoss)}
        </div>
        <div className={`kpi-sub ${potentialLoss > 0 ? "kpi-sub--red" : "kpi-sub--green"}`}>
          {potentialLoss > 0 ? t("kpi.manual_apip") : t("kpi.kas_aman")}
        </div>
      </div>

      <div className="kpi-card animate-fade-in-up">
        <div className="kpi-title">
          <span>{t("kpi.kemandirian_fiskal")}</span>
          <Percent size={14} color="#0284c7" />
        </div>
        <div className="kpi-value kpi-value--dark">
          {kemandirianFiskal.toFixed(1)}%
        </div>
        <div className="kpi-sub kpi-sub--green">{t("kpi.simulasi_hkpd")}</div>
      </div>
    </div>
  );
}
