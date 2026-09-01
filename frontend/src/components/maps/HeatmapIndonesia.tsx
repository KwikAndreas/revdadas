import { useState, useEffect, useRef } from "react";
import L from "leaflet";
import { X, MapPin, AlertTriangle } from "lucide-react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import type {
  HistoricalRecord,
  ForecastRecord,
  AnomalyRecord,
} from "@/lib/types";
import {
  PROVINCE_COORDS,
  INDONESIA_BOUNDS,
  formatCurrency,
  getRiskColor,
} from "@/lib/utils";

// Leaflet default icon fix for Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

interface HeatmapProps {
  historical: HistoricalRecord[];
  forecast: ForecastRecord[];
  anomalies: AnomalyRecord[];
  selectedProvinces: string[];
}

// Component to handle map view updates when selected provinces change
function MapUpdater({ selectedProvinces }: { selectedProvinces: string[] }) {
  const map = useMap();
  useEffect(() => {
    if (selectedProvinces.length === 0) return;
    
    // Create a bounding box covering all selected provinces
    const bounds = L.latLngBounds([]);
    selectedProvinces.forEach((prov) => {
      if (PROVINCE_COORDS[prov]) {
        bounds.extend(PROVINCE_COORDS[prov]);
      }
    });

    if (bounds.isValid()) {
      // Pad bounds slightly
      map.fitBounds(bounds.pad(0.1), { maxZoom: 8 });
    } else {
      map.fitBounds(INDONESIA_BOUNDS);
    }
  }, [map, selectedProvinces]);
  
  return null;
}

export default function HeatmapIndonesia({
  historical,
  forecast,
  anomalies,
  selectedProvinces,
}: HeatmapProps) {
  const [modalProv, setModalProv] = useState<string | null>(null);
  // Aggregate data per province
  const provinceData = selectedProvinces.map((prov) => {
    // Total historical revenue (exclude expenditures)
    const provHistorical = historical.filter((r) => r.Provinsi === prov && !r.Jenis_Pendapatan.toLowerCase().includes("belanja"));
    const totalRev = provHistorical.reduce((sum, r) => sum + r.Realisasi, 0);

    // Total forecast (exclude expenditures)
    const provForecast = forecast.filter((r) => r.Provinsi === prov && !r.Jenis_Pendapatan.toLowerCase().includes("belanja"));
    const totalForecast = provForecast.reduce((sum, r) => sum + r.Prediksi, 0);

    // Risk / Anomalies — use Deviasi (deviation from expected) not total Realisasi
    const provAnomalies = anomalies.filter(
      (r) => r.Provinsi === prov && r.Anomaly
    );
    const totalAnomalyValue = provAnomalies.reduce(
      (sum, r) => sum + Math.abs(r.Deviasi ?? 0),
      0
    );
    const riskPct = totalRev > 0 ? (totalAnomalyValue / totalRev) * 100 : 0;

    return {
      provinsi: prov,
      coords: PROVINCE_COORDS[prov] || [-2.5, 118.0],
      totalRev,
      totalForecast,
      riskPct,
      color: getRiskColor(riskPct),
    };
  });

  const activeData = modalProv ? provinceData.find((p) => p.provinsi === modalProv) : null;

  const leafletPopupStyle = `
    .leaflet-popup-content-wrapper {
      border-radius: 12px !important;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.15) !important;
      padding: 0 !important;
      overflow: hidden;
    }
    .leaflet-popup-content {
      margin: 0 !important;
      width: 240px !important;
    }
    .leaflet-container a.leaflet-popup-close-button {
      color: #64748b !important;
      right: 8px !important;
      top: 10px !important;
      font-size: 18px !important;
      padding: 0 !important;
      width: 24px !important;
      height: 24px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      border-radius: 50% !important;
      background: transparent !important;
      z-index: 99 !important;
      text-decoration: none !important;
    }
    .leaflet-container a.leaflet-popup-close-button:hover {
      background: #f1f5f9 !important;
      color: #0f172a !important;
    }
  `;

  return (
    <div style={{ position: "relative" }}>
      <style>{leafletPopupStyle}</style>
      <MapContainer
      center={[-2.5, 118.0]}
      zoom={5}
      minZoom={4}
      maxBounds={INDONESIA_BOUNDS}
      maxBoundsViscosity={1.0}
      style={{ height: 400, width: "100%", background: "#f8fafc" }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <MapUpdater selectedProvinces={selectedProvinces} />

      {provinceData.map((data) => (
        <CircleMarker
          key={data.provinsi}
          center={data.coords as [number, number]}
          radius={14}
          pathOptions={{
            color: "white",
            weight: 2,
            fillColor: data.color,
            fillOpacity: 0.9,
          }}
        >
          <Popup>
            <div style={{ padding: "16px", fontFamily: "var(--font-sans)" }}>
              <div style={{ fontSize: "15px", fontWeight: 700, color: "#0f172a", paddingBottom: "12px", marginBottom: "12px", borderBottom: "1px solid #e2e8f0", paddingRight: "24px" }}>
                {data.provinsi}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "8px", color: "#475569" }}>
                <span>Rev Aktual:</span>
                <span style={{ fontWeight: 600, color: "#0f172a" }}>{formatCurrency(data.totalRev)}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "8px", color: "#475569" }}>
                <span>Forecast:</span>
                <span style={{ fontWeight: 600, color: "#3b82f6" }}>{formatCurrency(data.totalForecast)}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "16px", color: "#475569" }}>
                <span>Risiko Anomali:</span>
                <span style={{ fontWeight: 700, color: data.riskPct > 1 ? "#dc2626" : "#16a34a" }}>{data.riskPct.toFixed(1)}%</span>
              </div>
              <button 
                onClick={() => setModalProv(data.provinsi)}
                style={{ 
                  width: "100%", 
                  padding: "10px", 
                  background: "#1e3a5f", 
                  color: "white", 
                  border: "none", 
                  borderRadius: "6px", 
                  fontWeight: 600, 
                  fontSize: "12px",
                  cursor: "pointer",
                  transition: "background 0.2s"
                }}
                onMouseOver={(e) => e.currentTarget.style.background = "#2a4f7f"}
                onMouseOut={(e) => e.currentTarget.style.background = "#1e3a5f"}
              >
                Tinjau Detail Wilayah
              </button>
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>

    {/* Modal Detail Wilayah */}
    {modalProv && activeData && (
      <div className="modal-overlay" onClick={() => setModalProv(null)} style={{ position: "fixed", zIndex: 9999 }}>
        <div className="modal-content" onClick={e => e.stopPropagation()}>
          <div className="modal-header">
            <div className="modal-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <MapPin size={20} color="#1e3a5f" />
              {activeData.provinsi}
            </div>
            <button className="modal-close-btn" onClick={() => setModalProv(null)}>
              <X size={20} />
            </button>
          </div>
          <div className="modal-body">
            <div className="metrics-row" style={{ marginBottom: 16 }}>
              <div className="metric-card" style={{ padding: 12 }}>
                <div className="metric-label">Total Realisasi</div>
                <div className="metric-value" style={{ fontSize: 18 }}>{formatCurrency(activeData.totalRev)}</div>
              </div>
              <div className="metric-card" style={{ padding: 12 }}>
                <div className="metric-label">Total Proyeksi</div>
                <div className="metric-value" style={{ fontSize: 18, color: "#3b82f6" }}>{formatCurrency(activeData.totalForecast)}</div>
              </div>
            </div>
            <div className="warning-box" style={{ background: activeData.riskPct > 1 ? "#fef2f2" : "#f0fdf4", borderLeftColor: activeData.riskPct > 1 ? "#dc2626" : "#22c55e", marginBottom: 0 }}>
              <div className="warning-icon" style={{ color: activeData.riskPct > 1 ? "#dc2626" : "#22c55e" }}>
                {activeData.riskPct > 1 ? <AlertTriangle size={20} /> : <MapPin size={20} />}
              </div>
              <div>
                <div className="warning-label" style={{ color: activeData.riskPct > 1 ? "#991b1b" : "#166534" }}>
                  {activeData.riskPct > 1 ? `RISIKO KEBOCORAN: ${activeData.riskPct.toFixed(1)}%` : "WILAYAH OPTIMAL"}
                </div>
                <div className="warning-text">
                  {activeData.riskPct > 1 
                    ? "Terdapat potensi anomali penerimaan di wilayah ini berdasarkan rekam jejak historis dan deviasi musiman."
                    : "Pola penerimaan kas daerah berfluktuasi secara normal tanpa indikasi anomali tajam."}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )}
    </div>
  );
}
