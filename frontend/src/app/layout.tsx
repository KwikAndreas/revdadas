import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RevDadas — Revenue Daerah Cerdas",
  description:
    "Sistem analitik berbasis AI untuk deteksi fraud dan peramalan pendapatan pemerintah daerah. Ditenagai oleh Prophet & Isolation Forest.",
  keywords: [
    "RevDadas",
    "Revenue Daerah",
    "Fraud Detection",
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
        <link
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
