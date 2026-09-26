import { ImageResponse } from "next/og";
import { EVENT_NAME, SITE_NAME, SITE_TAGLINE, TEAM } from "@/lib/site";

export const alt = `${SITE_NAME} — ${SITE_TAGLINE}: intelijen fiskal daerah berbasis AI`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "64px 72px",
          background: "linear-gradient(145deg, #1e3a5f 0%, #0f172a 75%)",
          color: "#ffffff",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div
            style={{
              width: 72,
              height: 72,
              borderRadius: 16,
              background: "#b91c1c",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12.5h3.2l2.3-5.5 3.6 10 2.4-6.2 1.6 1.7H21" />
            </svg>
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 44, fontWeight: 800 }}>{SITE_NAME}</div>
            <div style={{ fontSize: 20, color: "#94a3b8", letterSpacing: 3 }}>{SITE_TAGLINE.toUpperCase()}</div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ fontSize: 60, fontWeight: 800, lineHeight: 1.12, maxWidth: 980 }}>
            Intelijen Fiskal Daerah Berbasis AI untuk Optimalisasi Kas &amp; PAD
          </div>
          <div style={{ fontSize: 26, color: "#cbd5e1" }}>
            Proyeksi pendapatan · Deteksi anomali APBD · Rekomendasi kebijakan · 38 provinsi
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 22 }}>
          <div style={{ display: "flex", color: "#bbf7d0" }}>{EVENT_NAME}</div>
          <div style={{ display: "flex", color: "#94a3b8" }}>{TEAM.name}</div>
        </div>
      </div>
    ),
    size
  );
}
