import { Compass, CheckCircle2, AlertCircle, ShieldCheck } from "lucide-react";

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
      {/* Header Institusional (Clean & Calmed) */}
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "space-between", 
        marginBottom: 12,
        paddingBottom: 8,
        borderBottom: "1px solid #f1f5f9",
        flexWrap: "wrap",
        gap: 8
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            color: "#0284c7",
            display: "flex",
            alignItems: "center"
          }}>
            <Compass size={16} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, color: "#0f172a" }}>
              Radar Intelijen Fiskal &amp; Regulasi
            </div>
            <span style={{ fontSize: 11, color: "#64748b" }}>
              Rekomendasi Berbasis UU HKPD No. 1/2022 &amp; PP 35/2023
            </span>
          </div>
        </div>

        {/* Minimal Status Dot */}
        <span style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 5,
          fontSize: 11,
          fontWeight: 600,
          color: isOptimal ? "#166534" : "#9a3412"
        }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: isOptimal ? "#16a34a" : "#ea580c" }} />
          {isOptimal ? "Koridor Fiskal Terkendali" : "Perhatian Khusus Audit"}
        </span>
      </div>

      {/* Content */}
      {insightData ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Diagnosis Block */}
          <div style={{ fontSize: 12, lineHeight: 1.55, color: "#334155" }}>
            <span style={{ fontWeight: 700, color: "#0f172a", marginRight: 4 }}>
              Diagnosis:
            </span>
            {insightData.kondisi}
          </div>

          {/* Strategic Action Block */}
          <div style={{
            background: "#f8fafc",
            borderLeft: `3px solid ${isOptimal ? "#0284c7" : "#d97706"}`,
            borderRadius: "0 6px 6px 0",
            padding: "8px 12px"
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: isOptimal ? "#0369a1" : "#854d0e", marginBottom: 3 }}>
              Arahan BPKAD &amp; Inspektorat Daerah:
            </div>
            <p style={{ margin: 0, fontSize: 12, color: "#475569", lineHeight: 1.5 }}>
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

      <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px dashed #e2e8f0", fontSize: 11, color: "#64748b", display: "flex", alignItems: "center", gap: 6 }}>
        <ShieldCheck size={13} color="#059669" />
        <span>Rekomendasi dipetakan secara deterministik berdasar katalog regulasi fiskal resmi (menjamin 100% konsistensi tanpa halusinasi AI).</span>
      </div>
    </div>
  );
}
