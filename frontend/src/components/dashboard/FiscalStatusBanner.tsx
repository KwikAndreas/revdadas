import { useState } from "react";
import { 
  ShieldCheck, AlertCircle, AlertTriangle, Users, ChevronDown, ChevronUp, 
  Building2, ShieldAlert, Landmark, Target 
} from "lucide-react";
import { useLanguage } from "@/lib/LanguageContext";

interface FiscalStatusBannerProps {
  anomalyPct: number;
  anomalyCount: number;
  selectedProvinces: string[];
  selectedYear: number;
}

export default function FiscalStatusBanner({
  anomalyPct,
  anomalyCount,
  selectedProvinces,
  selectedYear,
}: FiscalStatusBannerProps) {
  const { lang, t } = useLanguage();
  const [showPersonaGuide, setShowPersonaGuide] = useState(false);

  // Indikator Status & Pemberitahuan Fiskal
  let statusColor = "#16a34a";
  let statusBorder = "#86efac";
  let statusTitle = t("ews.normal_title");
  let statusDesc = t("ews.normal_desc");
  let StatusIcon = ShieldCheck;

  if (anomalyPct >= 15 || anomalyCount >= 10) {
    statusColor = "#dc2626";
    statusBorder = "#fca5a5";
    statusTitle = t("ews.critical_title");
    statusDesc = t("ews.critical_desc");
    StatusIcon = ShieldAlert;
  } else if (anomalyPct >= 5 || anomalyCount > 0) {
    statusColor = "#d97706";
    statusBorder = "#fcd34d";
    statusTitle = t("ews.warning_title");
    statusDesc = t("ews.warning_desc");
    StatusIcon = AlertCircle;
  }

  const provText = selectedProvinces.length === 1 
    ? selectedProvinces[0] 
    : `${selectedProvinces.length} Wilayah`;

  return (
    <div style={{ marginBottom: 16 }}>
      {/* Sleek Executive EWS Strip */}
      <div 
        style={{
          background: "#ffffff",
          border: "1px solid #e2e8f0",
          borderLeft: `4px solid ${statusColor}`,
          borderRadius: 8,
          padding: "10px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 12,
          boxShadow: "0 1px 3px rgba(15, 23, 42, 0.03)"
        }}
        className="animate-fade-in"
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "1 1 320px", minWidth: 0 }}>
          <div style={{
            color: statusColor,
            display: "flex",
            alignItems: "center",
            flexShrink: 0
          }}>
            <StatusIcon size={18} strokeWidth={2.4} />
          </div>
          <div style={{ minWidth: 0, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span style={{ 
              fontSize: 12, 
              fontWeight: 700, 
              color: statusColor, 
              background: `${statusColor}14`, 
              padding: "2px 8px", 
              borderRadius: 4,
              letterSpacing: 0.2
            }}>
              Pemberitahuan: {statusTitle}
            </span>
            <span style={{ fontSize: 12, color: "#475569", lineHeight: 1.4 }}>
              {statusDesc}
            </span>
          </div>
        </div>

        {/* Minimal Trigger Panduan Persona */}
        <button
          onClick={() => setShowPersonaGuide(!showPersonaGuide)}
          style={{
            background: "transparent",
            border: "1px solid #cbd5e1",
            borderRadius: 6,
            padding: "6px 12px",
            fontSize: 11.5,
            fontWeight: 600,
            color: "#334155",
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            transition: "all 0.15s ease",
            flexShrink: 0
          }}
          aria-expanded={showPersonaGuide}
          onMouseOver={(e) => {
            e.currentTarget.style.borderColor = "#94a3b8";
            e.currentTarget.style.background = "#f8fafc";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.borderColor = "#cbd5e1";
            e.currentTarget.style.background = "transparent";
          }}
        >
          <Users size={13} color="#0284c7" />
          <span>{lang === "en" ? `Persona Guide (${provText})` : `Panduan Persona (${provText})`}</span>
          {showPersonaGuide ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
      </div>

      {/* Panel Ekspansi: Persona Pengguna (Ringkas & Tenang) */}
      {showPersonaGuide && (
        <div 
          style={{
            marginTop: 8,
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
            padding: "14px 18px",
            fontSize: 12,
            boxShadow: "0 2px 8px rgba(0,0,0,0.03)"
          }}
          className="animate-fade-in"
        >
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 700, color: "#0f172a", marginBottom: 10, fontSize: 12.5 }}>
            <Target size={14} color="#0284c7" />
            <span>{lang === "en" ? "RevDadas Use Cases & Institutional Personas:" : "Momen Penggunaan & Persona RevDaDas:"}</span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))", gap: 12 }}>
            {/* Persona 1: Bapenda */}
            <div style={{ padding: "10px 12px", borderRadius: 6, background: "#f8fafc", border: "1px solid #f1f5f9" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 700, color: "#0369a1", fontSize: 12, marginBottom: 4 }}>
                <Building2 size={13} />
                <span>{lang === "en" ? "Bapenda / Revenue Agency" : "Bapenda / Pengelola PAD"}</span>
              </div>
              <p style={{ margin: 0, color: "#475569", lineHeight: 1.45, fontSize: 11.5 }}>
                <b>{lang === "en" ? "Monthly Monitoring:" : "Monitoring Bulanan:"}</b> {lang === "en" ? "Early shortfall detection before year-end and accelerating slow tax/retribution collection." : "Deteksi shortfall dini sebelum tutup tahun dan akselerasi penagihan pos pajak/retribusi yang melambat."}
              </p>
            </div>

            {/* Persona 2: APIP / Inspektorat */}
            <div style={{ padding: "10px 12px", borderRadius: 6, background: "#f8fafc", border: "1px solid #f1f5f9" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 700, color: "#b91c1c", fontSize: 12, marginBottom: 4 }}>
                <ShieldAlert size={13} />
                <span>{lang === "en" ? "Regional Inspectorate / APIP" : "Inspektorat Daerah / APIP"}</span>
              </div>
              <p style={{ margin: 0, color: "#475569", lineHeight: 1.45, fontSize: 11.5 }}>
                <b>{lang === "en" ? "Risk-Based Audit:" : "Risk-Based Audit:"}</b> {lang === "en" ? "Targeted audits based on statistical deviations (Z > 2.0σ) avoiding random manual checks across thousands of files." : "Audit terarah berbasis deviasi statistik (Z > 2.0σ) tanpa pemeriksaan manual acak ribuan berkas."}
              </p>
            </div>

            {/* Persona 3: Bank Indonesia / TP2DD */}
            <div style={{ padding: "10px 12px", borderRadius: 6, background: "#f8fafc", border: "1px solid #f1f5f9" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 700, color: "#15803d", fontSize: 12, marginBottom: 4 }}>
                <Landmark size={13} />
                <span>{lang === "en" ? "Bank Indonesia & TP2DD Task Force" : "Bank Indonesia & Satgas TP2DD"}</span>
              </div>
              <p style={{ margin: 0, color: "#475569", lineHeight: 1.45, fontSize: 11.5 }}>
                <b>{lang === "en" ? "ETPD Evaluation:" : "Evaluasi ETPD:"}</b> {lang === "en" ? "Verify QRIS/KKPD digitalization matches real revenue, and control idle cash in regional banks." : "Verifikasi digitalisasi QRIS/KKPD berbanding lurus dengan kas PAD riil, dan kendalikan idle cash di BPD."}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
