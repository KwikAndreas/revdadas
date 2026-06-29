import { formatCurrency } from "@/lib/utils";
import { Calculator, ArrowRightCircle } from "lucide-react";

interface ImpactCalculatorProps {
  potentialLoss: number;
  fraudPreventionPct: number;
  onShowRecs: () => void;
}

export default function ImpactCalculator({
  potentialLoss,
  fraudPreventionPct,
  onShowRecs,
}: ImpactCalculatorProps) {
  const potensiTambahan = potentialLoss * (fraudPreventionPct / 100);

  return (
    <div style={{ position: "relative" }}>
      <div style={{
        background: "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
        border: "1px solid #e2e8f0",
        borderRadius: 16,
        padding: 24,
        boxShadow: "0 10px 30px -10px rgba(15, 23, 42, 0.08)",
        position: "relative",
        overflow: "hidden"
      }}>
        {/* Decorative background element */}
        <div style={{
          position: "absolute",
          top: -30,
          right: -30,
          width: 150,
          height: 150,
          background: "radial-gradient(circle, rgba(59, 130, 246, 0.08) 0%, transparent 70%)",
          borderRadius: "50%"
        }} />

        <div style={{ display: "flex", alignItems: "center", gap: 10, color: "#1e3a5f", fontWeight: 700, marginBottom: 20 }}>
          <div style={{ background: "#eff6ff", padding: 8, borderRadius: 8, color: "#3b82f6", display: "flex" }}>
            <Calculator size={18} strokeWidth={2.5} />
          </div>
          <span style={{ fontSize: 15 }}>Kalkulator Pemulihan Kas</span>
        </div>
        
        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase" }}>
          Potensi Penyelamatan Arus Kas
        </div>
        
        <div style={{ fontSize: 32, fontWeight: 800, color: "#0f172a", margin: "4px 0 16px 0", letterSpacing: -0.5 }}>
          {formatCurrency(potensiTambahan)}
        </div>
        
        <div style={{ 
          background: "#f0fdf4", 
          borderLeft: "3px solid #22c55e", 
          padding: "12px 14px", 
          borderRadius: "0 8px 8px 0",
          fontSize: 12,
          color: "#166534",
          lineHeight: 1.5,
          marginBottom: 20
        }}>
          <b>Tindakan Proaktif:</b> Dengan mereduksi tingkat kebocoran/anomali sebesar <b>{fraudPreventionPct}%</b>, daerah dapat mengamankan likuiditas ekstra untuk belanja modal & infrastruktur.
        </div>

        <button 
          onClick={onShowRecs} 
          style={{ 
            width: "100%", 
            background: "#1e3a5f", 
            color: "white", 
            border: "none", 
            padding: "12px 16px", 
            borderRadius: 8, 
            fontWeight: 600, 
            display: "flex", 
            alignItems: "center", 
            justifyContent: "center", 
            gap: 8,
            cursor: "pointer",
            transition: "background 0.2s"
          }}
          onMouseOver={(e) => e.currentTarget.style.background = "#2a4f7f"}
          onMouseOut={(e) => e.currentTarget.style.background = "#1e3a5f"}
        >
          Tinjau Rekomendasi Strategis
          <ArrowRightCircle size={16} />
        </button>
      </div>
    </div>
  );
}
