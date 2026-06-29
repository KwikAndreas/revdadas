import { RefreshCcw, Download } from "lucide-react";

export default function Header({ onExportPDF }: { onExportPDF: () => void }) {
  return (
    <header className="app-header">
      <div className="header-center">
        AI-Driven Revenue Forecasting & Fraud Detection
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
