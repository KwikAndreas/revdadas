import { useMemo } from "react";
import { formatCurrency } from "@/lib/utils";
import { Compass, ArrowRight, ShieldAlert, CheckCircle2, AlertTriangle, Calculator } from "lucide-react";
import type { FiscalInsightData } from "./AIInsights";
import { useLanguage } from "@/lib/LanguageContext";

interface FiscalIntelligencePanelProps {
  potentialLoss: number;
  fraudPreventionPct: number;
  onShowRecs: () => void;
  insightData?: FiscalInsightData;
}

export default function FiscalIntelligencePanel({
  potentialLoss,
  fraudPreventionPct,
  onShowRecs,
  insightData,
}: FiscalIntelligencePanelProps) {
  const { lang, t } = useLanguage();
  const isOptimal = insightData?.isOptimal ?? true;
  const savedRevenue = potentialLoss * (fraudPreventionPct / 100);

  return (
    <div
      className="animate-fade-in-up"
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: "16px 18px",
        boxShadow: "0 1px 3px rgba(15, 23, 42, 0.03)",
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      {/* ── Header: Regulatory Context & Status Indicator ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          paddingBottom: 10,
          borderBottom: "1px solid #f1f5f9",
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ color: "#0284c7", display: "flex", alignItems: "center" }}>
            <Compass size={16} />
          </div>
          <div>
            <h4 style={{ margin: 0, fontWeight: 700, fontSize: 13.5, color: "#0f172a" }}>
              {t("intel.title")}
            </h4>
            <span style={{ fontSize: 11, color: "#64748b" }}>
              {t("intel.subtitle")}
            </span>
          </div>
        </div>

        {/* Minimal Status Dot (Antislop rule: real state, no pulsing glow) */}
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 5,
            fontSize: 11,
            fontWeight: 600,
            color: isOptimal ? "#166534" : "#b45309",
            background: isOptimal ? "#f0fdf4" : "#fffbeb",
            padding: "3px 8px",
            borderRadius: 4,
            border: `1px solid ${isOptimal ? "#bbf7d0" : "#fef3c7"}`,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: isOptimal ? "#16a34a" : "#d97706",
            }}
          />
          {isOptimal ? t("intel.controlled") : t("intel.audit_attention")}
        </span>
      </div>

      {/* ── Scenario Recovery & Action Bar ── */}
      <div
        style={{
          background: "#f8fafc",
          border: "1px solid #e2e8f0",
          borderRadius: 8,
          padding: "12px 14px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 2 }}>
            {t("intel.est_recovery", { pct: fraudPreventionPct })}
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: 20, fontWeight: 800, color: "#0f172a", fontVariantNumeric: "tabular-nums" }}>
              {formatCurrency(savedRevenue)}
            </span>
            {potentialLoss > 0 && (
              <span style={{ fontSize: 11.5, color: "#64748b" }}>
                {t("intel.from_risk")} {formatCurrency(potentialLoss)}
              </span>
            )}
          </div>
        </div>

        <button
          onClick={onShowRecs}
          style={{
            background: "#1e3a5f",
            color: "#ffffff",
            border: "none",
            borderRadius: 6,
            padding: "7px 12px",
            fontSize: 11.5,
            fontWeight: 600,
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            transition: "background 0.15s ease",
            flexShrink: 0,
          }}
          onMouseOver={(e) => (e.currentTarget.style.background = "#2a4f7f")}
          onMouseOut={(e) => (e.currentTarget.style.background = "#1e3a5f")}
        >
          <span>{t("intel.review_recs")}</span>
          <ArrowRight size={13} />
        </button>
      </div>

      {/* ── Diagnosis & Strategic Policy Guidance ── */}
      {insightData && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Diagnosis */}
          <div style={{ fontSize: 12, lineHeight: 1.5, color: "#334155" }}>
            <span style={{ fontWeight: 700, color: "#0f172a", marginRight: 4 }}>
              {t("intel.diagnosis")}
            </span>
            {insightData.kondisi}
          </div>

          {/* Strategic Action Briefing */}
          <div
            style={{
              background: "#ffffff",
              border: `1px solid ${isOptimal ? "#e2e8f0" : "#fed7aa"}`,
              borderLeft: `3px solid ${isOptimal ? "#0284c7" : "#ea580c"}`,
              borderRadius: 6,
              padding: "10px 12px",
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: isOptimal ? "#0369a1" : "#9a3412",
                marginBottom: 3,
              }}
            >
              {t("intel.guidance")}
            </div>
            <p style={{ margin: 0, fontSize: 11.5, color: "#475569", lineHeight: 1.5 }}>
              {insightData.rekomendasi}
            </p>
          </div>
        </div>
      )}

      {/* ── Quiet Institutional Footnote ── */}
      <div
        style={{
          fontSize: 10.5,
          color: "#94a3b8",
          display: "flex",
          alignItems: "center",
          gap: 5,
          paddingTop: 4,
        }}
      >
        <CheckCircle2 size={12} color="#0284c7" />
        <span>{t("intel.footnote")}</span>
      </div>
    </div>
  );
}
