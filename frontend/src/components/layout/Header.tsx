import { RefreshCcw, Download, Menu } from "lucide-react";

export default function Header({ onExportPDF, onMenuClick }: { onExportPDF: () => void, onMenuClick?: () => void }) {
  return (
    <header className="app-header">
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {onMenuClick && (
          <button className="mobile-only btn-icon" onClick={onMenuClick} style={{ padding: 4 }}>
            <Menu size={24} />
          </button>
        )}
        <div className="header-center">
          AI-Driven Revenue Forecasting & Anomaly Detection
        </div>
      </div>
      <div className="header-actions">
        <button
          className="btn btn-secondary"
          onClick={() => window.location.reload()}
        >
          <RefreshCcw size={14} /> Refresh
        </button>
        <button className="btn btn-primary" onClick={onExportPDF}>
          <Download size={14} /> Export PDF
        </button>
      </div>
    </header>
  );
}
