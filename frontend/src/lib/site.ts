/**
 * Konfigurasi situs & identitas tim (dipakai landing, kartu tim, dan SEO).
 */
export const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL || "https://revdadas.vercel.app").replace(/\/$/, "");

export const SITE_NAME = "RevDadas";
export const SITE_TAGLINE = "Revenue Daerah Cerdas";
export const SITE_DESCRIPTION =
  "RevDadas adalah sistem intelijen fiskal daerah berbasis AI: peramalan pendapatan dengan Profil Serapan Berjangkar, " +
  "deteksi anomali APBD dengan Isolation Forest, dan rekomendasi kebijakan untuk Bapenda, APIP, serta Bank Indonesia di 38 provinsi.";

export const EVENT_NAME = "PIDI BI DIGDAYA x Hackathon 2026";
export const REPO_URL = "https://github.com/KwikAndreas/revdadas";

export const TEAM = {
  name: "Team BITGrow",
  university: "Universitas Bunda Mulia",
  members: [
    { name: "Kwik Andreas Jonathan", role: "QA Engineer" },
    { name: "Clay Micholaz Fu", role: "AI Engineer" },
    { name: "Moses Chisthoper Adisam", role: "Software Engineer" },
    { name: "Gwyneth Eunice Widjaja", role: "UI UX Design" },
  ],
} as const;
