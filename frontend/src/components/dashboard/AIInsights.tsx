import { Lightbulb, AlertTriangle, CheckCircle } from "lucide-react";

export default function AIInsights({ insightText }: { insightText: string }) {
  const isOptimal = insightText.toLowerCase().includes("optimal");

  return (
    <div className="insight-card">
      <div className="insight-header">
        <Lightbulb size={18} color="#eab308" strokeWidth={2.5} /> 
        <span>Sistem Deteksi Anomali</span>
      </div>
      <div className="warning-box" style={{ 
        borderLeftColor: isOptimal ? "#22c55e" : "#f97316",
        background: isOptimal ? "#f0fdf4" : undefined 
      }}>
        <div className="warning-icon" style={{ color: isOptimal ? "#22c55e" : "#f97316" }}>
          {isOptimal ? <CheckCircle size={20} /> : <AlertTriangle size={20} />}
        </div>
        <div>
          <div className="warning-label" style={{ color: isOptimal ? "#166534" : undefined }}>
            {isOptimal ? "STATUS AMAN" : "PERINGATAN ANOMALI"}
          </div>
          <div
            className="warning-text"
            dangerouslySetInnerHTML={{ __html: insightText }}
          />
        </div>
      </div>
    </div>
  );
}
