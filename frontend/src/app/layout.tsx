import type { Metadata } from "next";
import "leaflet/dist/leaflet.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "RevDadas — Revenue Daerah Cerdas",
  description:
    "Sistem analitik berbasis AI untuk deteksi fraud dan peramalan pendapatan pemerintah daerah. Ditenagai oleh Prophet & Isolation Forest.",
  keywords: [
    "RevDadas",
    "Revenue Daerah",
    "Anomaly Detection",
    "AI Forecasting",
    "APBD",
    "Pajak Daerah",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}