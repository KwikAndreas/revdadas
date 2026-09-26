import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Dashboard Intelijen Fiskal",
  description:
    "Dashboard RevDadas: realisasi & proyeksi pendapatan daerah, peta risiko anomali 38 provinsi, dan rekomendasi kebijakan berbasis data APBD DJPK.",
  alternates: { canonical: "/dashboard" },
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return children;
}
