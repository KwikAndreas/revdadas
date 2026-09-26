"use client";

import React, { createContext, useContext, useState, useEffect, useMemo } from "react";

export type Language = "id" | "en";

interface LanguageContextType {
  lang: Language;
  setLang: (lang: Language) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
}

const translations: Record<Language, Record<string, string>> = {
  id: {
    // Header
    "header.title": "Prediksi Pendapatan & Deteksi Anomali Berbasis AI",
    "header.refresh": "Muat Ulang",
    "header.export": "Ekspor",
    "header.export_title": "PILIH FORMAT EKSPOR",
    "header.export_pdf": "Laporan Eksekutif Dossier B2G Resmi",
    "header.export_pdf_desc": "Analisis fiskal & rekomendasi kebijakan terstruktur",
    "header.export_excel": "Workbook Data Mentah & Estimasi Model",
    "header.export_excel_desc": "Historical, proyeksi multi-horizon, & anomali",
    "header.export_word": "Ringkasan Eksekutif Terstruktur untuk Rapat",
    "header.export_word_desc": "Format dokumen siap telaah staf ahli",

    // Sidebar
    "sidebar.subtitle": "PENDAPATAN DAERAH CERDAS",
    "sidebar.year": "TAHUN ANGGARAN",
    "sidebar.revenue_type": "JENIS PENDAPATAN",
    "sidebar.all_revenues": "Semua Pendapatan",
    "sidebar.target_prov": "PROVINSI TARGET",
    "sidebar.all_provinces": "Semua Provinsi",
    "sidebar.selected_count": "{count} Provinsi Terpilih",
    "sidebar.select_all": "Pilih Semua",
    "sidebar.reset": "Reset",
    "sidebar.search_prov": "Cari provinsi...",
    "sidebar.forecast_period": "PERIODE PREDIKSI",
    "sidebar.months": "BULAN",
    "sidebar.recovery_rate": "ASUMSI RECOVERY RATE ANOMALI",
    "sidebar.recovery_hint": "*Estimasi persentase pemulihan anomali",
    "sidebar.connected": "Terkoneksi (Satu Data)",
    "sidebar.records_count": "{count} records",

    // EWS FiscalStatusBanner
    "ews.critical_title": "Peringatan Kritis",
    "ews.critical_desc": "{count} pos deviasi terdeteksi (paparan {pct}% dari realisasi). Memerlukan audit kepatuhan terarah Inspektorat/APIP.",
    "ews.warning_title": "Status Waspada",
    "ews.warning_desc": "{count} pos deviasi terdeteksi (paparan {pct}% dari realisasi). Direkomendasikan evaluasi rekonsiliasi data Bapenda.",
    "ews.normal_title": "Status Normal",
    "ews.normal_desc": "Arus kas & kepatuhan PAD dalam lintasan stabil sesuai profil APBD. Tidak diperlukan intervensi darurat.",
    "ews.guide_btn": "Panduan Persona ({count} Wilayah)",
    "ews.guide_title": "Matriks Panduan & Tindakan Berdasarkan Peran Instansi",
    "ews.persona_bapenda": "Bapenda (Badan Pendapatan Daerah)",
    "ews.persona_bapenda_task": "Rekonsiliasi harian pajak & percepatan kanal pembayaran digital QRIS/VA.",
    "ews.persona_bpkad": "BPKAD (Pengelola Keuangan & Aset)",
    "ews.persona_bpkad_task": "Penyesuaian estimasi arus kas triwulanan & mitigasi defisit berjalan.",
    "ews.persona_apip": "Inspektorat Daerah / APIP",
    "ews.persona_apip_task": "Audit uji petik berbasis deviasi material pada pos berisiko tinggi.",
    "ews.persona_bi": "Kantor Perwakilan Bank Indonesia (KPw BI)",
    "ews.persona_bi_task": "Monitoring implementasi ETPD, Indeks IETPD, dan elektronifikasi pemda.",

    // KPI Cards
    "kpi.realisasi_pad": "Realisasi PAD Kumulatif",
    "kpi.realisasi_total": "Realisasi Pendapatan Daerah TA {year}",
    "kpi.realisasi_type": "Realisasi {type} TA {year}",
    "kpi.target_annual": "{pct}% dari Target Tahunan",
    "kpi.actual_realization": "Realisasi Aktual",
    "kpi.proyeksi": "Proyeksi ({months} Bln ke Depan)",
    "kpi.risiko_deviasi": "Risiko Deviasi Kas",
    "kpi.deviasi_view": "{count} deviasi (Lihat Tabel)",
    "kpi.normal_status": "Status Normal (Z < 2.0σ)",
    "kpi.transaksi_tinjauan": "Transaksi Perlu Tinjauan",
    "kpi.manual_apip": "Verifikasi Manual APIP",
    "kpi.kas_aman": "Kas Aman Terkendali",
    "kpi.kemandirian_fiskal": "Kemandirian Fiskal",
    "kpi.simulasi_hkpd": "Simulasi UU HKPD",

    // Map
    "map.title": "Heatmap Sebaran Revenue & Risiko Daerah",
    "map.optimal": "Optimal (≤2%)",
    "map.moderate": "Moderat (2–5%)",
    "map.critical": "Kritis (>5%)",

    // FiscalIntelligencePanel
    "intel.title": "Intelijen Fiskal & Mitigasi Risiko",
    "intel.subtitle": "Berbasis UU HKPD No. 1/2022 & PP 35/2023",
    "intel.controlled": "Koridor Fiskal Terkendali",
    "intel.audit_attention": "Perhatian Khusus Audit",
    "intel.est_recovery": "Estimasi Pemulihan Kas (Skenario {pct}%):",
    "intel.from_risk": "dari risiko",
    "intel.review_recs": "Tinjau Rekomendasi",
    "intel.diagnosis": "Diagnosis:",
    "intel.guidance": "Arahan BPKAD & Inspektorat Daerah:",
    "intel.footnote": "Rekomendasi dipetakan secara deterministik sesuai katalog regulasi fiskal resmi.",

    // RegionalContext
    "reg.title": "Profil Spasial & Konteks Fiskal:",
    "reg.sub_nat": "Sintesis makro-fiskal, disparitas regional, dan kalibrasi elektronifikasi daerah se-Indonesia",
    "reg.sub_prov": "Kalibrasi intelijen berdasarkan struktur ekonomi dan karakteristik daerah",
    "reg.nat_btn": "Agregat Nasional (38 Provinsi)",
    "reg.pad_source": "Sumber Utama PAD",
    "reg.absorption_pattern": "Pola Penyerapan Kas",
    "reg.policy_strategy": "Strategi TP2DD & Fiskal",
    "reg.heatmap_status": "Status di Heatmap:",
    "reg.anomalies_detected": "{count} Pos Anomali Terdeteksi ({provinces} Wilayah)",
    "reg.normal_flow": "Semua pos dalam batas toleransi normal",
    "reg.tp2dd_mandate": "Mandat TP2DD: Memaksimalkan kanal nontunai daerah",

    // Policy Recommendations
    "policy.title": "Rekomendasi Kebijakan Berbasis Data",
    "policy.sub": "Diurutkan berdasarkan tingkat urgensi • Menampilkan {count} dari {total} rekomendasi",
    "policy.close": "Tutup Rekomendasi",
    "policy.priority": "Prioritas",
    "policy.baseline": "BASELINE (EXISTING)",
    "policy.ai_rec": "REKOMENDASI AI",
    "policy.pros": "Kelebihan",
    "policy.cons": "Risiko / Kekurangan",
    "policy.justification": "Justifikasi:",
    "policy.indicator": "Indikator:",
    "policy.impact": "Dampak:",
    "policy.prev": "Sebelumnya",
    "policy.next": "Selanjutnya",
    "policy.page": "Halaman",
    "policy.page_of": "dari",
    "policy.per_page": "rekomendasi per halaman",
    "policy.disclaimer": "Rekomendasi bersifat indikatif sebagai bahan diskusi kebijakan, bukan keputusan final.",

    // Charts
    "chart.revenue_forecast": "Realisasi Historis vs Proyeksi Pendapatan",
    "chart.model_caption": "Model proyeksi: {model}",
    "chart.revenue_proportion": "Proporsi Sumber Pendapatan",
    "chart.legend_actual": "Pendapatan Historis",
    "chart.legend_forecast": "Proyeksi AI",
    "chart.legend_ci": "Rentang Keyakinan",

    // DataTabs
    "tabs.title": "Log Detail Data",
    "tabs.forecast": "Data Proyeksi",
    "tabs.anomalies": "Deteksi Anomali",
    "tabs.exploratory": "Eksplorasi Data",
    "tabs.accuracy": "Akurasi Model",
    "tabs.whatif": "Simulasi What-If",
    "tabs.dictionary": "Kamus Data",
    "tabs.methodology": "Metodologi",
    "tabs.download_csv": "Unduh CSV",
    "tabs.showing_rows": "Menampilkan {count} baris data proyeksi.",
    "tabs.showing_anomalies": "Menampilkan {count} deteksi anomali operasional.",
    "tabs.sort_by": "Urutkan:",
    "tabs.sort_severity": "Tingkat Keparahan",
    "tabs.sort_realization": "Nominal Realisasi",
    "tabs.sort_date": "Tanggal",
    "tabs.col_date": "Tanggal",
    "tabs.col_prov": "Provinsi",
    "tabs.col_rev_type": "Jenis Pendapatan",
    "tabs.col_pred": "Prediksi",
    "tabs.col_lower": "Batas Bawah",
    "tabs.col_upper": "Batas Atas",
    "tabs.col_method": "Metode",
    "tabs.col_context": "Konteks Anomali",
    "tabs.col_value": "Nilai Transaksi",
    "tabs.col_reason": "Alasan Anomali",
    "tabs.limited_accuracy": "Estimasi Terbatas (WAPE > 50%)",
    "tabs.context_hkpd": "Transisi UU HKPD",
    "tabs.context_seasonal": "Siklus Musiman",
    "tabs.context_operational": "Deviasi Operasional",
  },
  en: {
    // Header
    "header.title": "AI-Driven Revenue Forecasting & Anomaly Detection",
    "header.refresh": "Refresh",
    "header.export": "Export",
    "header.export_title": "CHOOSE EXPORT FORMAT",
    "header.export_pdf": "Official Executive B2G Dossier",
    "header.export_pdf_desc": "Fiscal analysis & structured policy guidance",
    "header.export_excel": "Raw Data & Model Forecast Workbook",
    "header.export_excel_desc": "Historical, multi-horizon forecasts, & anomalies",
    "header.export_word": "Structured Executive Brief for Meetings",
    "header.export_word_desc": "Document format ready for expert staff review",

    // Sidebar
    "sidebar.subtitle": "SMART REGIONAL REVENUE",
    "sidebar.year": "FISCAL YEAR",
    "sidebar.revenue_type": "REVENUE TYPE",
    "sidebar.all_revenues": "All Revenues",
    "sidebar.target_prov": "TARGET PROVINCES",
    "sidebar.all_provinces": "All Provinces",
    "sidebar.selected_count": "{count} Provinces Selected",
    "sidebar.select_all": "Select All",
    "sidebar.reset": "Reset",
    "sidebar.search_prov": "Search province...",
    "sidebar.forecast_period": "FORECAST PERIOD",
    "sidebar.months": "MONTHS",
    "sidebar.recovery_rate": "ANOMALY RECOVERY RATE ASSUMPTION",
    "sidebar.recovery_hint": "*Estimated anomaly recovery percentage",
    "sidebar.connected": "Connected (Satu Data)",
    "sidebar.records_count": "{count} records",

    // EWS FiscalStatusBanner
    "ews.critical_title": "Critical Warning",
    "ews.critical_desc": "{count} deviating posts detected ({pct}% exposure of realization). Requires targeted compliance audit by Inspectorate/APIP.",
    "ews.warning_title": "Advisory Status",
    "ews.warning_desc": "{count} deviating posts detected ({pct}% exposure of realization). Bapenda reconciliation review recommended.",
    "ews.normal_title": "Normal Status",
    "ews.normal_desc": "Cash flow & local tax compliance on stable trajectory according to APBD profile. No emergency intervention needed.",
    "ews.guide_btn": "Persona Guide ({count} Regions)",
    "ews.guide_title": "Guidance Matrix & Actions by Institutional Role",
    "ews.persona_bapenda": "Bapenda (Regional Revenue Agency)",
    "ews.persona_bapenda_task": "Daily tax reconciliation & acceleration of digital payment channels (QRIS/VA).",
    "ews.persona_bpkad": "BPKAD (Financial & Asset Management)",
    "ews.persona_bpkad_task": "Quarterly cash flow projection adjustment & ongoing deficit mitigation.",
    "ews.persona_apip": "Regional Inspectorate / APIP",
    "ews.persona_apip_task": "Risk-based spot audit on high-risk material deviation posts.",
    "ews.persona_bi": "Bank Indonesia Regional Office (KPw BI)",
    "ews.persona_bi_task": "Monitoring regional electronification (ETPD) implementation & IETPD Index.",

    // KPI Cards
    "kpi.realisasi_pad": "Cumulative Regional Revenue Realization",
    "kpi.realisasi_total": "Regional Revenue Realization FY {year}",
    "kpi.realisasi_type": "{type} Realization FY {year}",
    "kpi.target_annual": "{pct}% of Annual Target",
    "kpi.actual_realization": "Actual Realization",
    "kpi.proyeksi": "Forecast ({months} Months Ahead)",
    "kpi.risiko_deviasi": "Cash Deviation Risk",
    "kpi.deviasi_view": "{count} deviations (View Table)",
    "kpi.normal_status": "Normal Status (Z < 2.0σ)",
    "kpi.transaksi_tinjauan": "Transactions for Review",
    "kpi.manual_apip": "APIP Manual Verification",
    "kpi.kas_aman": "Cash Secure & Controlled",
    "kpi.kemandirian_fiskal": "Fiscal Independence",
    "kpi.simulasi_hkpd": "Law UU HKPD Simulation",

    // Map
    "map.title": "Regional Revenue & Risk Spatial Heatmap",
    "map.optimal": "Optimal (≤2%)",
    "map.moderate": "Moderate (2–5%)",
    "map.critical": "Critical (>5%)",

    // FiscalIntelligencePanel
    "intel.title": "Fiscal Intelligence & Risk Mitigation",
    "intel.subtitle": "Based on Law UU HKPD No. 1/2022 & PP 35/2023",
    "intel.controlled": "Fiscal Corridor Controlled",
    "intel.audit_attention": "Special Audit Attention",
    "intel.est_recovery": "Cash Recovery Estimate (Scenario {pct}%):",
    "intel.from_risk": "from identified risk",
    "intel.review_recs": "Review Recommendations",
    "intel.diagnosis": "Diagnosis:",
    "intel.guidance": "BPKAD & Regional Inspectorate Guidance:",
    "intel.footnote": "Recommendations mapped deterministically according to official fiscal regulatory catalog.",

    // RegionalContext
    "reg.title": "Spatial Profile & Fiscal Context:",
    "reg.sub_nat": "Macro-fiscal synthesis, regional disparity, and regional electronification calibration across Indonesia",
    "reg.sub_prov": "Intelligence calibration based on regional economic structure and characteristics",
    "reg.nat_btn": "National Aggregate (38 Provinces)",
    "reg.pad_source": "Primary Regional Revenue Sources",
    "reg.absorption_pattern": "Cash Absorption Pattern",
    "reg.policy_strategy": "TP2DD & Fiscal Strategy",
    "reg.heatmap_status": "Heatmap Status:",
    "reg.anomalies_detected": "{count} Anomaly Posts Detected ({provinces} Regions)",
    "reg.normal_flow": "All posts within normal tolerance bounds",
    "reg.tp2dd_mandate": "TP2DD Mandate: Maximize regional cashless payment channels",

    // Policy Recommendations
    "policy.title": "Data-Driven Policy Recommendations",
    "policy.sub": "Sorted by urgency level • Displaying {count} of {total} recommendations",
    "policy.close": "Close Recommendations",
    "policy.priority": "Priority",
    "policy.baseline": "BASELINE (EXISTING)",
    "policy.ai_rec": "AI RECOMMENDATION",
    "policy.pros": "Advantages",
    "policy.cons": "Risks / Limitations",
    "policy.justification": "Justification:",
    "policy.indicator": "Indicator:",
    "policy.impact": "Impact:",
    "policy.prev": "Previous",
    "policy.next": "Next",
    "policy.page": "Page",
    "policy.page_of": "of",
    "policy.per_page": "recommendations per page",
    "policy.disclaimer": "Recommendations are indicative for policy discussions, not final executive decisions.",

    // Charts
    "chart.revenue_forecast": "Historical Revenue vs Forecast",
    "chart.model_caption": "Forecast model: {model}",
    "chart.revenue_proportion": "Revenue Source Proportion",
    "chart.legend_actual": "Historical Revenue",
    "chart.legend_forecast": "AI Forecast",
    "chart.legend_ci": "Confidence Interval",

    // DataTabs
    "tabs.title": "Detailed Data Logs",
    "tabs.forecast": "Forecast Data",
    "tabs.anomalies": "Anomaly Detection",
    "tabs.exploratory": "Exploratory Data",
    "tabs.accuracy": "Model Accuracy",
    "tabs.whatif": "What-If Simulation",
    "tabs.dictionary": "Data Dictionary",
    "tabs.methodology": "Methodology",
    "tabs.download_csv": "Download CSV",
    "tabs.showing_rows": "Displaying {count} forecast data rows.",
    "tabs.showing_anomalies": "Displaying {count} operational anomaly detections.",
    "tabs.sort_by": "Sort by:",
    "tabs.sort_severity": "Severity Level",
    "tabs.sort_realization": "Realization Amount",
    "tabs.sort_date": "Date",
    "tabs.col_date": "Date",
    "tabs.col_prov": "Province",
    "tabs.col_rev_type": "Revenue Type",
    "tabs.col_pred": "Prediction",
    "tabs.col_lower": "Lower Bound",
    "tabs.col_upper": "Upper Bound",
    "tabs.col_method": "Method",
    "tabs.col_context": "Anomaly Context",
    "tabs.col_value": "Transaction Value",
    "tabs.col_reason": "Anomaly Reason",
    "tabs.limited_accuracy": "Limited Signal (WAPE > 50%)",
    "tabs.context_hkpd": "UU HKPD Transition",
    "tabs.context_seasonal": "Seasonal Cycle",
    "tabs.context_operational": "Operational Deviation",
  },
};

const LanguageContext = createContext<LanguageContextType>({
  lang: "id",
  setLang: () => {},
  t: (key: string) => key,
});

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Language>("id");

  useEffect(() => {
    try {
      localStorage.setItem("revdadas_lang", "id");
    } catch (_) {}
  }, []);

  const setLang = (newLang: Language) => {
    setLangState(newLang);
    try {
      localStorage.setItem("revdadas_lang", newLang);
    } catch (_) {}
  };

  const t = useMemo(() => {
    return (key: string, params?: Record<string, string | number>): string => {
      let str = translations[lang]?.[key] ?? translations["id"]?.[key] ?? key;
      if (params) {
        Object.entries(params).forEach(([k, v]) => {
          str = str.replace(new RegExp(`\\{${k}\\}`, "g"), String(v));
        });
      }
      return str;
    };
  }, [lang]);

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
