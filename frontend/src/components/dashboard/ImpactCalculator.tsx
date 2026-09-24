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
    <div>
      <div style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: 20,
        boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)"
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#0f172a", fontWeight: 700, fontSize: 13.5 }}>
            <Calculator size={16} color="#0284c7" />
            <span>Kalkulator Skenario Audit</span>
          </div>
          <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b", background: "#f1f5f9", padding: "2px 8px", borderRadius: 4 }}>
            Asumsi {fraudPreventionPct}%
          </span>
        </div>
        
        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600 }}>
          Estimasi Pemulihan Likuiditas:
        </div>
        
        <div style={{ fontSize: 26, fontWeight: 800, color: "#0f172a", margin: "2px 0 12px 0" }}>
          {formatCurrency(potensiTambahan)}
        </div>
        
        <p style={{ 
          margin: "0 0 16px 0",
          fontSize: 11.5,
          color: "#475569",
          lineHeight: 1.5
        }}>
          Proyeksi nilai kas yang dapat diamankan jika <b>{fraudPreventionPct}%</b> transaksi anomali diselesaikan melalui audit terarah oleh Inspektorat/APIP.
        </p>

        <button 
          onClick={onShowRecs} 
          style={{ 
            width: "100%", 
            background: "#1e3a5f", 
            color: "white", 
            border: "none", 
            padding: "10px 14px", 
            borderRadius: 6, 
            fontSize: 12,
            fontWeight: 600, 
            display: "flex", 
            alignItems: "center", 
            justifyContent: "center", 
            gap: 6,
            cursor: "pointer",
            transition: "background 0.15s ease"
          }}
          onMouseOver={(e) => e.currentTarget.style.background = "#2a4f7f"}
          onMouseOut={(e) => e.currentTarget.style.background = "#1e3a5f"}
        >
          <span>Tinjau Rekomendasi Strategis</span>
          <ArrowRightCircle size={14} />
        </button>
      </div>
    </div>
  );
}
