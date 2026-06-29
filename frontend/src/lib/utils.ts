/**
 * Utility functions for RevDadas dashboard.
 */

/**
 * Format a number as Indonesian Rupiah.
 */
export function formatCurrency(value: number): string {
  if (Math.abs(value) >= 1e12) {
    return `Rp ${(value / 1e12).toFixed(1)} T`;
  }
  if (Math.abs(value) >= 1e9) {
    return `Rp ${(value / 1e9).toFixed(1)} M`;
  }
  if (Math.abs(value) >= 1e6) {
    return `Rp ${(value / 1e6).toFixed(1)} Jt`;
  }
  return `Rp ${value.toLocaleString("id-ID", { maximumFractionDigits: 0 })}`;
}

/**
 * Format a number as billions (Miliar Rp) for chart display.
 */
export function toBillions(value: number): number {
  return value / 1e9;
}

/**
 * Province coordinates for the Indonesia heatmap.
 */
export const PROVINCE_COORDS: Record<string, [number, number]> = {
  "Aceh": [4.6951, 96.7494],
  "Sumatera Utara": [2.1154, 99.5451],
  "Sumatera Barat": [-0.7399, 100.8000],
  "Riau": [0.2933, 101.7068],
  "Kepulauan Riau": [3.9457, 108.1429],
  "Jambi": [-1.4852, 102.4381],
  "Sumatera Selatan": [-3.3194, 103.9140],
  "Bangka Belitung": [-2.7411, 106.4406],
  "Bengkulu": [-3.5778, 102.3464],
  "Lampung": [-4.5586, 105.4068],
  "DKI Jakarta": [-6.2000, 106.8166],
  "Jawa Barat": [-6.9204, 107.6046],
  "Banten": [-6.4058, 106.0640],
  "Jawa Tengah": [-7.1500, 110.1403],
  "DI Yogyakarta": [-7.8754, 110.4262],
  "Jawa Timur": [-7.5361, 112.2384],
  "Bali": [-8.4095, 115.1889],
  "Nusa Tenggara Barat": [-8.6529, 117.3616],
  "Nusa Tenggara Timur": [-8.6574, 121.0794],
  "Kalimantan Barat": [-0.2788, 111.4753],
  "Kalimantan Tengah": [-1.6815, 113.3824],
  "Kalimantan Selatan": [-3.0926, 115.2838],
  "Kalimantan Timur": [0.5387, 116.4194],
  "Kalimantan Utara": [3.0731, 116.0414],
  "Sulawesi Utara": [0.6247, 123.9750],
  "Gorontalo": [0.6999, 122.4467],
  "Sulawesi Tengah": [-1.4300, 121.4456],
  "Sulawesi Barat": [-2.8441, 119.2321],
  "Sulawesi Selatan": [-3.6688, 119.9741],
  "Sulawesi Tenggara": [-4.1449, 122.1746],
  "Maluku": [-3.2385, 130.1453],
  "Maluku Utara": [1.5710, 127.8088],
  "Papua": [-4.2699, 138.0804],
  "Papua Barat": [-1.3361, 133.1747],
};

/**
 * Indonesia map bounds for constraining the view.
 */
export const INDONESIA_BOUNDS: [[number, number], [number, number]] = [
  [-11.5, 94.5],
  [6.5, 141.5],
];

/**
 * Default checked provinces for initial state.
 */
export const DEFAULT_PROVINCES = ["DKI Jakarta", "Jawa Barat", "Jawa Timur"];

/**
 * Get risk color based on percentage.
 */
export function getRiskColor(riskPct: number): string {
  if (riskPct > 5.0) return "#ef4444"; // Red
  if (riskPct > 2.0) return "#eab308"; // Yellow
  return "#22c55e"; // Green
}

/**
 * Get priority badge colors.
 */
export function getPriorityColors(priority: string): { bg: string; text: string } {
  switch (priority) {
    case "Tinggi":
      return { bg: "#fee2e2", text: "#dc2626" };
    case "Sedang":
      return { bg: "#fef3c7", text: "#d97706" };
    case "Rendah":
      return { bg: "#dcfce7", text: "#16a34a" };
    default:
      return { bg: "#f1f5f9", text: "#64748b" };
  }
}

/**
 * Business sector label colors.
 */
export function getLabelColor(label: string): string {
  switch (label) {
    case "Sangat Prospektif":
      return "#0f766e";
    case "Prospektif":
      return "#1e3a5f";
    case "Cukup / Hati-hati":
      return "#b45309";
    case "Kurang Mendukung":
      return "#991b1b";
    default:
      return "#475569";
  }
}

/**
 * Snap a value to the nearest available forecast period.
 */
export function snapToPeriod(value: number, periods: number[]): number {
  return periods.reduce((prev, curr) =>
    Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
  );
}
