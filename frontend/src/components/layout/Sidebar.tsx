"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import type { Meta, DashboardFilters } from "@/lib/types";
import { ChevronDown, Check, Activity, Search } from "lucide-react";

interface SidebarProps {
  meta: Meta;
  filters: DashboardFilters;
  onFilterChange: (partial: Partial<DashboardFilters>) => void;
  isOpen?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ meta, filters, onFilterChange, isOpen, onClose }: SidebarProps) {
  const allTaxTypes = ["Semua Pendapatan", ...meta.tax_types];
  const [provDropdownOpen, setProvDropdownOpen] = useState(false);
  const [taxDropdownOpen, setTaxDropdownOpen] = useState(false);
  const [yearDropdownOpen, setYearDropdownOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);
  const taxDropdownRef = useRef<HTMLDivElement>(null);
  const yearDropdownRef = useRef<HTMLDivElement>(null);

  const availableYears = useMemo(() => {
    const minYear = parseInt(meta.date_range.min.substring(0, 4));
    const maxYear = parseInt(meta.date_range.max.substring(0, 4));
    const years = [];
    for (let i = maxYear; i >= minYear; i--) years.push(i);
    return years;
  }, [meta.date_range]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (dropdownRef.current && !dropdownRef.current.contains(target)) {
        setProvDropdownOpen(false);
      }
      if (taxDropdownRef.current && !taxDropdownRef.current.contains(target)) {
        setTaxDropdownOpen(false);
      }
      if (yearDropdownRef.current && !yearDropdownRef.current.contains(target)) {
        setYearDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleProvinceToggle = (province: string) => {
    const current = filters.selectedProvinces;
    const updated = current.includes(province)
      ? current.filter((p) => p !== province)
      : [...current, province];
    onFilterChange({ selectedProvinces: updated.length > 0 ? updated : meta.provinces });
  };

  const isAllSelected = filters.selectedProvinces.length === meta.provinces.length;

  const handleSelectAll = () => {
    if (isAllSelected) {
      // Just select one default to prevent empty state error
      onFilterChange({ selectedProvinces: [meta.provinces[0]] }); 
    } else {
      onFilterChange({ selectedProvinces: meta.provinces });
    }
  };

  const filteredProvinces = useMemo(() => {
    return meta.provinces.filter(p => p.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [meta.provinces, searchQuery]);

  return (
    <aside 
      className={`sidebar ${isOpen ? "open" : ""}`}
      style={isOpen ? { 
        position: "fixed",
        top: 0, 
        left: 0, 
        width: "280px", 
        height: "100vh", 
        backgroundColor: "#ffffff",
        zIndex: 999999, 
        transform: "none", 
        visibility: "visible",
        display: "flex",
        flexDirection: "column",
        boxShadow: "2px 0 20px rgba(0,0,0,0.5)"
      } : {}}
    >
      {/* Logo */}
      <div className="sidebar-logo" style={{ justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div className="sidebar-logo-icon">
            <Activity size={20} strokeWidth={3} />
          </div>
          <div className="sidebar-logo-text">
            <h2>RevDadas</h2>
            <span>REVENUE DAERAH CERDAS</span>
          </div>
        </div>
        {onClose && (
          <button className="mobile-only btn-icon" onClick={onClose} style={{ border: "none", background: "transparent", color: "#64748b" }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
        )}
      </div>

      {/* Fiscal Year Filter */}
      <div>
        <p className="sidebar-section-title">TAHUN ANGGARAN</p>
        <div className="dropdown-container" ref={yearDropdownRef}>
          <div 
            className="dropdown-trigger" 
            onClick={() => setYearDropdownOpen(!yearDropdownOpen)}
          >
            <span>TA {filters.selectedYear}</span>
            <ChevronDown size={14} color="#64748b" />
          </div>
          
          {yearDropdownOpen && (
            <div className="dropdown-menu">
              {availableYears.map((y) => {
                const isSelected = filters.selectedYear === y;
                return (
                  <div 
                    key={y} 
                    className={`dropdown-item ${isSelected ? "selected" : ""}`}
                    onClick={() => {
                      onFilterChange({ selectedYear: y });
                      setYearDropdownOpen(false);
                    }}
                  >
                    <span>TA {y}</span>
                    {isSelected && <Check size={14} strokeWidth={3} />}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Tax Type Filter */}
      <div>
        <p className="sidebar-section-title">JENIS PENDAPATAN</p>
        <div className="dropdown-container" ref={taxDropdownRef}>
          <div 
            className="dropdown-trigger" 
            onClick={() => setTaxDropdownOpen(!taxDropdownOpen)}
          >
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {filters.selectedTaxType}
            </span>
            <ChevronDown size={14} color="#64748b" />
          </div>
          
          {taxDropdownOpen && (
            <div className="dropdown-menu">
              {allTaxTypes.map((t) => {
                const isSelected = filters.selectedTaxType === t;
                return (
                  <div 
                    key={t} 
                    className={`dropdown-item ${isSelected ? "selected" : ""}`}
                    onClick={() => {
                      onFilterChange({ selectedTaxType: t });
                      setTaxDropdownOpen(false);
                    }}
                  >
                    <span>{t}</span>
                    {isSelected && <Check size={14} strokeWidth={3} />}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Province Filter */}
      <div>
        <p className="sidebar-section-title">PROVINSI TARGET</p>
        <div className="dropdown-container" ref={dropdownRef}>
          <div 
            className="dropdown-trigger" 
            onClick={() => setProvDropdownOpen(!provDropdownOpen)}
          >
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {filters.selectedProvinces.length === meta.provinces.length 
                ? "Semua Provinsi" 
                : `${filters.selectedProvinces.length} Provinsi Terpilih`}
            </span>
            <ChevronDown size={14} color="#64748b" />
          </div>
          
          {provDropdownOpen && (
            <div className="dropdown-menu">
              <div style={{ padding: "8px", borderBottom: "1px solid #e2e8f0" }}>
                <div style={{ display: "flex", alignItems: "center", background: "#f1f5f9", borderRadius: 6, padding: "4px 8px" }}>
                  <Search size={14} color="#64748b" />
                  <input 
                    type="text" 
                    placeholder="Cari provinsi..." 
                    style={{ border: "none", background: "transparent", outline: "none", padding: "4px 8px", width: "100%", fontSize: 13 }}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>
              </div>
              
              <div 
                className="dropdown-item" 
                style={{ borderBottom: "1px solid #e2e8f0", fontWeight: 700, color: "#1e3a5f" }}
                onClick={handleSelectAll}
              >
                <span>{isAllSelected ? "Hapus Semua" : "Pilih Semua Provinsi"}</span>
                {isAllSelected && <Check size={14} strokeWidth={3} />}
              </div>

              <div style={{ maxHeight: 200, overflowY: "auto" }}>
                {filteredProvinces.length === 0 ? (
                  <div style={{ padding: 12, textAlign: "center", color: "#64748b", fontSize: 13 }}>Tidak ada hasil</div>
                ) : (
                  filteredProvinces.map((prov) => {
                    const isSelected = filters.selectedProvinces.includes(prov);
                    return (
                      <div 
                        key={prov} 
                        className={`dropdown-item ${isSelected ? "selected" : ""}`}
                        onClick={() => handleProvinceToggle(prov)}
                      >
                        <span>{prov}</span>
                        {isSelected && <Check size={14} strokeWidth={3} />}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Forecast Period Slider */}
      <div className="slider-container" style={{ marginTop: 8 }}>
        <div className="slider-header">
          <span className="sidebar-section-title" style={{ margin: 0 }}>
            PERIODE PREDIKSI
          </span>
          <span className="slider-value">{filters.forecastMonths} BULAN</span>
        </div>
        <input
          type="range"
          className="slider-input"
          min={6}
          max={24}
          value={filters.forecastMonths}
          onChange={(e) =>
            onFilterChange({ forecastMonths: Number(e.target.value) })
          }
        />
        <div className="slider-range-labels">
          <span>6 Bln</span>
          <span>24 Bln</span>
        </div>
      </div>

      {/* Audit Effectiveness Slider */}
      <div>
        <div className="fraud-box">
          <div className="fraud-header">
            <span>ASUMSI EFEKTIVITAS AUDIT</span>
            <span>{filters.fraudPreventionPct}%</span>
          </div>
          <input
            type="range"
            className="slider-input"
            min={0}
            max={100}
            value={filters.fraudPreventionPct}
            onChange={(e) =>
              onFilterChange({ fraudPreventionPct: Number(e.target.value) })
            }
          />
        </div>
        <p className="fraud-note">*Estimasi efektivitas audit AI</p>
      </div>

      {/* Status */}
      <div className="sidebar-status">
        <p>
          Status Sistem: <span className="status-dot">●</span> Terkoneksi (Satu Data)
        </p>
        <p>Data: {meta.total_rows.toLocaleString()} records</p>
      </div>
    </aside>
  );
}
