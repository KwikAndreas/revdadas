import { Compass, CheckCircle2, AlertCircle } from "lucide-react";

export interface FiscalInsightData {
  isOptimal: boolean;
  kondisi: string;
  rekomendasi: string;
  metricLabel?: string;
  metricValue?: string;
  recoveryValue?: string;
}

interface AIInsightsProps {
  insightText?: string;
  insightData?: FiscalInsightData;
}

export default function AIInsights({ insightText, insightData }: AIInsightsProps) {
  const isOptimal = insightData 
    ? insightData.isOptimal 
    : (insightText?.toLowerCase().includes("optimal") ?? true);

  return (
    <div 
      className="insight-card animate-fade-in"
      style={{
        background: "#ffffff",
        borderRadius: 8,
        border: "1px solid #e2e8f0",
        padding: "16px 20px",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        marginTop: 16
      }}
    >
      {/* Header Institusional */}
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "space-between", 
        marginBottom: 14,
        paddingBottom: 10,
        borderBottom: "1px solid #f1f5f9" 
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 28,
            height: 28,
            borderRadius: 6,
            background: "#f1f5f9",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#1e3a5f"
          }}>
            <Compass size={16} />
          </div>
          <div>
            <span style={{ fontWeight: 700, fontSize: 13, color: "#0f172a", letterSpacing: "-0.01em" }}>
              Radar Kebijakan & Intelijen Fiskal
            </span>
            <span style={{ display: "block", fontSize: 11, color: "#64748b" }}>
              Briefing Strategis Pemerintah Daerah (B2G)
            </span>
          </div>
        </div>

        {/* Status Indicator */}
        <span style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 5,
          padding: "3px 10px",
          borderRadius: 12,
          fontSize: 11,
          fontWeight: 600,
          background: isOptimal ? "#ecfdf5" : "#fff7ed",
          border: isOptimal ? "1px solid #a7f3d0" : "1px solid #fed7aa",
          color: isOptimal ? "#065f46" : "#9a3412"
        }}>
          {isOptimal ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
          {isOptimal ? "Koridor Fiskal Normal" : "Perhatian Risiko Audit"}
        </span>
      </div>

      {/* Content */}
      {insightData ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Diagnosis Block */}
          <div style={{ fontSize: 12.5, lineHeight: 1.6, color: "#334155" }}>
            <span style={{ fontWeight: 700, color: "#0f172a", marginRight: 6 }}>
              Diagnosis Realisasi:
            </span>
            {insightData.kondisi}
          </div>

          {/* Strategic Action Block */}
          <div style={{
            background: isOptimal ? "#f8fafc" : "#fefce8",
            borderLeft: isOptimal ? "3px solid #0284c7" : "3px solid #d97706",
            borderTop: "1px solid #e2e8f0",
            borderRight: "1px solid #e2e8f0",
            borderBottom: "1px solid #e2e8f0",
            borderRadius: "0 6px 6px 0",
            padding: "10px 14px"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4, flexWrap: "wrap", gap: 8 }}>
              <span style={{
                fontSize: 11,
                fontWeight: 700,
                color: isOptimal ? "#0369a1" : "#854d0e",
                letterSpacing: "0.5px",
                textTransform: "uppercase"
              }}>
                Rekomendasi BPKAD & Inspektorat Daerah
              </span>
              {insightData.recoveryValue && (
                <span style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: "#166534",
                  background: "#dcfce7",
                  padding: "1px 6px",
                  borderRadius: 4
                }}>
                  Potensi Pemulihan: {insightData.recoveryValue}
                </span>
              )}
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#475569", lineHeight: 1.55 }}>
              {insightData.rekomendasi}
            </p>
          </div>
        </div>
      ) : (
        <div 
          style={{ fontSize: 12.5, lineHeight: 1.6, color: "#334155" }}
          dangerouslySetInnerHTML={{ 
            __html: (insightText || "")
              .replace(/🎯/g, "")
              .replace(/🚨/g, "")
              .replace(/<b>🎯/g, "<b>")
              .replace(/<b>🚨/g, "<b>")
          }} 
        />
      )}
    </div>
  );
}
