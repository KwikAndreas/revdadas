import { useState, useRef, useEffect } from "react";
import { RefreshCcw, Download, Menu, ChevronDown, FileText, FileSpreadsheet, File } from "lucide-react";

export type ExportFormat = "pdf" | "xlsx" | "docx";

export default function Header({
  onExport,
  onMenuClick,
}: {
  onExport: (format: ExportFormat) => void;
  onMenuClick?: () => void;
}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (format: ExportFormat) => {
    setDropdownOpen(false);
    onExport(format);
  };

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

        {/* Unified Export Dropdown */}
        <div className="export-dropdown-wrapper" ref={dropdownRef} style={{ position: "relative" }}>
          <button 
            className="btn btn-primary" 
            onClick={() => setDropdownOpen(!dropdownOpen)}
            style={{ display: "flex", alignItems: "center", gap: 6 }}
            aria-haspopup="true"
            aria-expanded={dropdownOpen}
          >
            <Download size={14} /> 
            <span>Export</span>
            <ChevronDown 
              size={13} 
              style={{ 
                transition: "transform 0.2s ease", 
                transform: dropdownOpen ? "rotate(180deg)" : "rotate(0deg)" 
              }} 
            />
          </button>

          {dropdownOpen && (
            <div 
              style={{
                position: "absolute",
                top: "calc(100% + 6px)",
                right: 0,
                width: 280,
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                boxShadow: "0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1)",
                padding: 6,
                zIndex: 100,
                animation: "fadeIn 0.15s ease-out"
              }}
            >
              <div style={{ 
                padding: "6px 10px 4px 10px", 
                fontSize: 10.5, 
                fontWeight: 700, 
                color: "#94a3b8", 
                textTransform: "uppercase", 
                letterSpacing: "0.05em" 
              }}>
                Pilih Format Ekspor
              </div>

              {/* Option 1: PDF */}
              <button
                type="button"
                onClick={() => handleSelect("pdf")}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 10,
                  width: "100%",
                  padding: "8px 10px",
                  background: "transparent",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "background 0.15s"
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f8fafc")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <div style={{
                  padding: 6,
                  borderRadius: 6,
                  background: "#fee2e2",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginTop: 2
                }}>
                  <FileText size={16} color="#dc2626" />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>PDF Document (.pdf)</div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Laporan Eksekutif Dossier B2G Resmi</div>
                </div>
              </button>

              {/* Option 2: Excel */}
              <button
                type="button"
                onClick={() => handleSelect("xlsx")}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 10,
                  width: "100%",
                  padding: "8px 10px",
                  background: "transparent",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "background 0.15s"
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f8fafc")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <div style={{
                  padding: 6,
                  borderRadius: 6,
                  background: "#dcfce7",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginTop: 2
                }}>
                  <FileSpreadsheet size={16} color="#16a34a" />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>Excel Spreadsheet (.xlsx)</div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Lembar Kerja Multitab & Dataset Lengkap</div>
                </div>
              </button>

              {/* Option 3: Word */}
              <button
                type="button"
                onClick={() => handleSelect("docx")}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 10,
                  width: "100%",
                  padding: "8px 10px",
                  background: "transparent",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "background 0.15s"
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f8fafc")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <div style={{
                  padding: 6,
                  borderRadius: 6,
                  background: "#dbeafe",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginTop: 2
                }}>
                  <File size={16} color="#2563eb" />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>Word Document (.docx)</div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Dokumen Telaah & Nota Dinas Kebijakan</div>
                </div>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

