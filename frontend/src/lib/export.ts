import * as XLSX from "xlsx";
import {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  AlignmentType,
  HeadingLevel,
  BorderStyle,
  ShadingType
} from "docx";
import { generatePDF } from "./pdf";
import type { DashboardFilters, PolicyRecommendation, AnomalyRecord, BusinessData, Meta, ForecastRecord } from "./types";
import { formatCurrency } from "./utils";

export interface ExportDataPayload {
  kpis: {
    totalRevenue: number;
    targetPercentage?: number;
    forecastTotal: number;
    anomalyCount: number;
    potentialLoss: number;
    anomalyPct: number;
    kemandirianFiskal: number;
    savedRevenue: number;
    anomalies: AnomalyRecord[];
  };
  policyRecs: PolicyRecommendation[];
  filters: DashboardFilters;
  bizData?: BusinessData;
  insightData?: {
    isOptimal?: boolean;
    kondisi?: string;
    rekomendasi?: string;
    metricLabel?: string;
    metricValue?: string;
    recoveryValue?: string;
  };
  meta?: Meta;
  forecasts?: ForecastRecord[];
}

// ─────────────────────────────────────────────────────────────
// 1. EXPORT TO PDF
// ─────────────────────────────────────────────────────────────
export function exportToPDF(payload: ExportDataPayload) {
  generatePDF(
    payload.kpis,
    payload.policyRecs,
    payload.filters,
    payload.bizData,
    payload.insightData,
    payload.meta
  );
}

// ─────────────────────────────────────────────────────────────
// 2. EXPORT TO EXCEL (XLSX)
// ─────────────────────────────────────────────────────────────
export function exportToXLSX(payload: ExportDataPayload) {
  const { kpis, policyRecs, filters, bizData, insightData, meta, forecasts } = payload;
  const d = new Date();
  const dateStr = d.toLocaleDateString("id-ID", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
  const activeModelName = meta?.active_model_name || (meta?.active_model === "prophet" ? "Prophet (Secondary)" : "Theta Method (Primary)");

  const wb = XLSX.utils.book_new();

  // Sheet 1: Ringkasan Eksekutif & IKU
  const summaryAoa: any[][] = [
    ["REVDADAS FISCAL INTELLIGENCE DOSSIER — B2G EXECUTIVE SUMMARY"],
    ["Klasifikasi:", "Dokumen Resmi Telaah Anggaran & Pengawasan Kas Daerah"],
    ["Waktu Terbit:", dateStr],
    ["Model Peramalan:", activeModelName],
    ["Horizon Waktu:", `${filters.forecastMonths} Bulan`],
    ["Wilayah Analisis:", filters.selectedProvinces.join(", ") || "Semua Provinsi"],
    ["Pos Rekening:", filters.selectedTaxType],
    ["Efisiensi Fraud Mitigation:", `${filters.fraudPreventionPct}%`],
    [],
    ["1. SINTESIS STRATEGIS"],
    ["Kondisi Fiskal:", insightData?.kondisi || "-"],
    ["Arahan Kebijakan:", insightData?.rekomendasi || "-"],
    [],
    ["2. INDIKATOR KINERJA UTAMA (IKU FISKAL)"],
    ["Indikator Fiskal Strategis", "Nilai / Rasio", "Interpretasi & Implikasi B2G"],
    ["Total Realisasi Pendapatan", formatCurrency(kpis.totalRevenue), "Akumulasi penerimaan kas daerah tahun berjalan"],
    ["Capaian Target Anggaran", kpis.targetPercentage ? `${kpis.targetPercentage.toFixed(1)}%` : "Basis Estimasi", kpis.targetPercentage && kpis.targetPercentage < 85 ? "Perlu akselerasi penagihan" : "Stabil sesuai kalender fiskal"],
    [`Proyeksi AI (${filters.forecastMonths} Bulan)`, formatCurrency(kpis.forecastTotal), `Estimasi penerimaan menggunakan ${activeModelName}`],
    ["Derajat Kemandirian Fiskal (PAD)", `${kpis.kemandirianFiskal.toFixed(1)}%`, kpis.kemandirianFiskal >= 40 ? "Kemandirian Tinggi" : "Ketergantungan Transfer Pusat Dominan"],
    ["Paparan Anomali Kas", `${kpis.anomalyPct.toFixed(2)}% (${kpis.anomalyCount} Kasus)`, "Transaksi menyimpang >2.5 sigma untuk audit kepatuhan"],
    ["Potensi Penyelamatan Kas", formatCurrency(kpis.savedRevenue), `Target intervensi mitigasi fraud ${filters.fraudPreventionPct}%`]
  ];

  const wsSummary = XLSX.utils.aoa_to_sheet(summaryAoa);
  wsSummary["!cols"] = [{ wch: 32 }, { wch: 28 }, { wch: 55 }];
  XLSX.utils.book_append_sheet(wb, wsSummary, "Ringkasan IKU");

  // Sheet 2: Proyeksi Pendapatan (Forecast)
  if (forecasts && forecasts.length > 0) {
    const fcAoa: any[][] = [
      ["Tanggal", "Provinsi", "Jenis Pendapatan", "Nilai Proyeksi (Rp)", "Batas Bawah (Rp)", "Batas Atas (Rp)", "Metode Pemodelan"]
    ];
    forecasts.forEach((r) => {
      fcAoa.push([
        r.Tanggal.split("T")[0],
        r.Provinsi,
        r.Jenis_Pendapatan,
        r.Prediksi,
        r.Batas_Bawah,
        r.Batas_Atas,
        r.Metode || activeModelName
      ]);
    });
    const wsForecast = XLSX.utils.aoa_to_sheet(fcAoa);
    wsForecast["!cols"] = [{ wch: 12 }, { wch: 20 }, { wch: 32 }, { wch: 22 }, { wch: 22 }, { wch: 22 }, { wch: 24 }];
    XLSX.utils.book_append_sheet(wb, wsForecast, "Data Proyeksi");
  }

  // Sheet 3: Deteksi Anomali
  if (kpis.anomalies && kpis.anomalies.length > 0) {
    const anomAoa: any[][] = [
      ["Tanggal", "Provinsi", "Pos Rekening", "Realisasi (Rp)", "Deviasi (Rp)", "Tingkat Risiko", "Jenis Indikasi", "Analisis Algoritma"]
    ];
    kpis.anomalies.forEach((a) => {
      anomAoa.push([
        a.Tanggal.split("T")[0],
        a.Provinsi,
        a.Jenis_Pendapatan,
        a.Realisasi,
        a.Deviasi ?? 0,
        a.Severity || "Menengah",
        a.Jenis_Fraud || "Deviasi Anomali",
        a.Alasan || "Penyimpangan pola musiman terdeteksi"
      ]);
    });
    const wsAnom = XLSX.utils.aoa_to_sheet(anomAoa);
    wsAnom["!cols"] = [{ wch: 12 }, { wch: 18 }, { wch: 32 }, { wch: 22 }, { wch: 22 }, { wch: 16 }, { wch: 20 }, { wch: 45 }];
    XLSX.utils.book_append_sheet(wb, wsAnom, "Deteksi Anomali");
  }

  // Sheet 4: Rekomendasi Kebijakan
  if (policyRecs && policyRecs.length > 0) {
    const polAoa: any[][] = [
      ["Pilar Kebijakan", "Tingkat Prioritas", "Arahan Rencana Aksi", "Indikator Dampak Terukur", "Kaitan Sektor Bisnis", "Justifikasi Fiskal"]
    ];
    policyRecs.forEach((p) => {
      polAoa.push([
        p.judul,
        p.prioritas,
        p.detail,
        p.indikator_dampak || "-",
        p.kaitan_bisnis || "-",
        p.justifikasi || "-"
      ]);
    });
    const wsPol = XLSX.utils.aoa_to_sheet(polAoa);
    wsPol["!cols"] = [{ wch: 35 }, { wch: 16 }, { wch: 50 }, { wch: 35 }, { wch: 35 }, { wch: 45 }];
    XLSX.utils.book_append_sheet(wb, wsPol, "Matriks Kebijakan");
  }

  // Sheet 5: Sektor Unggulan Daerah
  if (bizData && bizData.scored) {
    const bizAoa: any[][] = [
      ["Provinsi", "Sektor Bisnis", "Skor AI (0-100)", "Label Kelayakan", "Katalis Fiskal & Alasan"]
    ];
    Object.keys(bizData.scored).forEach((prov) => {
      if (filters.selectedProvinces.includes(prov)) {
        bizData.scored[prov].forEach((s) => {
          bizAoa.push([
            prov,
            s.sektor,
            s.skor,
            s.label,
            s.alasan || s.narasi
          ]);
        });
      }
    });
    const wsBiz = XLSX.utils.aoa_to_sheet(bizAoa);
    wsBiz["!cols"] = [{ wch: 20 }, { wch: 28 }, { wch: 16 }, { wch: 18 }, { wch: 60 }];
    XLSX.utils.book_append_sheet(wb, wsBiz, "Potensi Sektor");
  }

  // Download
  const fileDate = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  const scope = filters.selectedProvinces.length === 1 ? filters.selectedProvinces[0].replace(/\s+/g, "_") : "Nasional";
  XLSX.writeFile(wb, `RevDadas_Dataset_Fiskal_${scope}_${fileDate}.xlsx`);
}

// ─────────────────────────────────────────────────────────────
// 3. EXPORT TO WORD (DOCX)
// ─────────────────────────────────────────────────────────────
export async function exportToDOCX(payload: ExportDataPayload) {
  const { kpis, policyRecs, filters, bizData, insightData, meta } = payload;
  const d = new Date();
  const dateStr = d.toLocaleDateString("id-ID", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
  const activeModelName = meta?.active_model_name || (meta?.active_model === "prophet" ? "Prophet (Secondary)" : "Theta Method (Primary)");

  // Color constants (Hex without #)
  const cNavyHex = "0F172A";
  const cDeepBlueHex = "1E3A8A";
  const cSlateHex = "475569";
  const cBorderHex = "CBD5E1";

  // Table Cell Helper
  const createCell = (text: string, isHeader = false, isBold = false, align: (typeof AlignmentType)[keyof typeof AlignmentType] = AlignmentType.LEFT, bgColor?: string) => {
    return new TableCell({
      shading: {
        fill: bgColor || (isHeader ? cNavyHex : "FFFFFF"),
        type: ShadingType.CLEAR
      },
      margins: { top: 120, bottom: 120, left: 140, right: 140 },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 4, color: cBorderHex },
        bottom: { style: BorderStyle.SINGLE, size: 4, color: cBorderHex },
        left: { style: BorderStyle.SINGLE, size: 4, color: cBorderHex },
        right: { style: BorderStyle.SINGLE, size: 4, color: cBorderHex },
      },
      children: [
        new Paragraph({
          alignment: align,
          children: [
            new TextRun({
              text,
              bold: isBold || isHeader,
              color: isHeader ? "FFFFFF" : cNavyHex,
              size: isHeader ? 19 : 18,
              font: "Arial"
            })
          ]
        })
      ]
    });
  };

  // 1. KPI Table Rows
  const kpiRows: TableRow[] = [
    new TableRow({
      children: [
        createCell("Indikator Fiskal Strategis", true, true, AlignmentType.LEFT),
        createCell("Nilai / Rasio", true, true, AlignmentType.RIGHT),
        createCell("Interpretasi & Implikasi B2G", true, true, AlignmentType.LEFT)
      ]
    }),
    new TableRow({
      children: [
        createCell("Total Realisasi Pendapatan", false, true),
        createCell(formatCurrency(kpis.totalRevenue), false, false, AlignmentType.RIGHT),
        createCell("Akumulasi penerimaan kas daerah dari seluruh sumber pos pendapatan tahun berjalan.")
      ]
    }),
    new TableRow({
      children: [
        createCell("Rasio Capaian Target Anggaran", false, true),
        createCell(kpis.targetPercentage ? `${kpis.targetPercentage.toFixed(1)}%` : "Basis Estimasi", false, false, AlignmentType.RIGHT),
        createCell(kpis.targetPercentage && kpis.targetPercentage < 85 ? "Realisasi di bawah 85%; direkomendasikan akselerasi penagihan aktif." : "Realisasi stabil sesuai lintasan kalender fiskal daerah.")
      ]
    }),
    new TableRow({
      children: [
        createCell(`Proyeksi AI (${filters.forecastMonths} Bulan)`, false, true),
        createCell(formatCurrency(kpis.forecastTotal), false, false, AlignmentType.RIGHT),
        createCell(`Estimasi arus penerimaan menggunakan ${activeModelName} dengan dekomposisi musiman 12 bulan.`)
      ]
    }),
    new TableRow({
      children: [
        createCell("Derajat Kemandirian Fiskal (PAD)", false, true),
        createCell(`${kpis.kemandirianFiskal.toFixed(1)}%`, false, false, AlignmentType.RIGHT),
        createCell(kpis.kemandirianFiskal >= 40 ? "Kemandirian Tinggi: Daerah memiliki fleksibilitas pendanaan belanja modal yang solid." : "Kemandirian Rentan: Ketergantungan terhadap transfer pemerintah pusat masih dominan.")
      ]
    }),
    new TableRow({
      children: [
        createCell("Tingkat Paparan Anomali Kas", false, true),
        createCell(`${kpis.anomalyPct.toFixed(2)}% (${kpis.anomalyCount} Kasus)`, false, false, AlignmentType.RIGHT),
        createCell("Proporsi transaksi menyimpang (>2.5 sigma) yang memerlukan klarifikasi audit kepatuhan.")
      ]
    }),
    new TableRow({
      children: [
        createCell("Potensi Penyelamatan Kas Fiskal", false, true),
        createCell(formatCurrency(kpis.savedRevenue), false, false, AlignmentType.RIGHT),
        createCell(`Nilai kas terpulihkan dengan target intervensi pencegahan kebocoran sebesar ${filters.fraudPreventionPct}%.`)
      ]
    })
  ];

  // 2. Anomaly Table Rows
  const anomalyRows: TableRow[] = [
    new TableRow({
      children: [
        createCell("Wilayah & Tanggal", true, true),
        createCell("Pos Rekening", true, true),
        createCell("Nilai Realisasi", true, true, AlignmentType.RIGHT),
        createCell("Tingkat Risiko", true, true, AlignmentType.CENTER),
        createCell("Catatan Investigasi Algoritma", true, true)
      ]
    })
  ];

  if (kpis.anomalies && kpis.anomalies.length > 0) {
    const top5 = [...kpis.anomalies]
      .sort((a, b) => Math.abs(b.Deviasi ?? 0) - Math.abs(a.Deviasi ?? 0))
      .slice(0, 5);

    top5.forEach((a) => {
      anomalyRows.push(
        new TableRow({
          children: [
            createCell(`${a.Provinsi}\n(${a.Tanggal.split("T")[0]})`, false, true),
            createCell(a.Jenis_Pendapatan),
            createCell(formatCurrency(a.Realisasi), false, false, AlignmentType.RIGHT),
            createCell(a.Severity || "Menengah", false, true, AlignmentType.CENTER, a.Severity === "Tinggi" ? "FEE2E2" : "FEF3C7"),
            createCell(a.Alasan || "Deviasi pola musiman signifikan terdeteksi oleh algoritma.")
          ]
        })
      );
    });
  } else {
    anomalyRows.push(
      new TableRow({
        children: [
          createCell("Semua Wilayah", false, true),
          createCell("Seluruh Akun"),
          createCell("-", false, false, AlignmentType.RIGHT),
          createCell("Aman", false, true, AlignmentType.CENTER, "DCFCE7"),
          createCell("Tidak ditemukan deviasi ekstrem (>2.5 sigma). Rekonsiliasi kas daerah terkonfirmasi stabil.")
        ]
      })
    );
  }

  // 3. Policy Recommendation Table Rows
  const policyTableRows: TableRow[] = [
    new TableRow({
      children: [
        createCell("Pilar Kebijakan", true, true),
        createCell("Urgensi", true, true, AlignmentType.CENTER),
        createCell("Arahan Rencana Aksi & Indikator Kunci", true, true)
      ]
    })
  ];

  policyRecs.slice(0, 5).forEach((p) => {
    policyTableRows.push(
      new TableRow({
        children: [
          createCell(p.judul, false, true),
          createCell(p.prioritas.toUpperCase(), false, true, AlignmentType.CENTER, p.prioritas === "Tinggi" ? "FEE2E2" : "FEF3C7"),
          createCell(`${p.detail}\n\nIndikator Dampak: ${p.indikator_dampak || "Peningkatan efisiensi kepatuhan pajak daerah."}`)
        ]
      })
    );
  });

  // Construct Document
  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }
          }
        },
        children: [
          // Header / Title
          new Paragraph({
            text: "REVDADAS FISCAL INTELLIGENCE DOSSIER",
            heading: HeadingLevel.TITLE,
            alignment: AlignmentType.LEFT,
            children: [
              new TextRun({
                text: "REVDADAS FISCAL INTELLIGENCE DOSSIER",
                bold: true,
                size: 36,
                color: cNavyHex,
                font: "Arial"
              })
            ]
          }),
          new Paragraph({
            children: [
              new TextRun({
                text: "Dokumen Telaah Strategis Perencanaan Anggaran & Pengawasan Kas Daerah (B2G Policy Brief)",
                color: cSlateHex,
                size: 20,
                font: "Arial"
              })
            ]
          }),
          new Paragraph({
            children: [
              new TextRun({
                text: `Waktu Terbit: ${dateStr}  |  Model: ${activeModelName}  |  Horizon: ${filters.forecastMonths} Bulan`,
                bold: true,
                color: cDeepBlueHex,
                size: 18,
                font: "Arial"
              })
            ],
            spacing: { after: 300 }
          }),

          // Section 1: Sintesis
          new Paragraph({
            text: "1. SINTESIS STRATEGIS & KONDISI FISKAL",
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
            children: [
              new TextRun({
                text: "1. SINTESIS STRATEGIS & KONDISI FISKAL",
                bold: true,
                color: cNavyHex,
                size: 24,
                font: "Arial"
              })
            ]
          }),
          new Paragraph({
            children: [
              new TextRun({ text: "Diagnosis Fiskal: ", bold: true, color: cNavyHex }),
              new TextRun({ text: insightData?.kondisi || `Realisasi pendapatan tercatat sebesar ${formatCurrency(kpis.totalRevenue)} dengan proyeksi kumulatif mencapai ${formatCurrency(kpis.forecastTotal)}.` })
            ],
            spacing: { after: 140 }
          }),
          new Paragraph({
            children: [
              new TextRun({ text: "Arahan Kebijakan: ", bold: true, color: cDeepBlueHex }),
              new TextRun({ text: insightData?.rekomendasi || "Perkuat disiplin monitoring kas daerah dan percepat penyelesaian audit uji petik berbasis risiko." })
            ],
            spacing: { after: 280 }
          }),

          // Section 2: IKU Fiskal
          new Paragraph({
            text: "2. INDIKATOR KINERJA UTAMA (IKU FISKAL)",
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 200, after: 140 },
            children: [
              new TextRun({
                text: "2. INDIKATOR KINERJA UTAMA (IKU FISKAL)",
                bold: true,
                color: cNavyHex,
                size: 24,
                font: "Arial"
              })
            ]
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: kpiRows
          }),

          // Section 3: Anomaly Detection
          new Paragraph({
            text: "3. MATRIKS DETEKSI ANOMALI & RISIKO PENERIMAAN",
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 300, after: 140 },
            children: [
              new TextRun({
                text: "3. MATRIKS DETEKSI ANOMALI & RISIKO PENERIMAAN",
                bold: true,
                color: cNavyHex,
                size: 24,
                font: "Arial"
              })
            ]
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: anomalyRows
          }),

          // Section 4: Policy Recommendations
          new Paragraph({
            text: "4. REKOMENDASI KEBIJAKAN BERBASIS BUKTI (POLICY ACTION MATRIX)",
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 300, after: 140 },
            children: [
              new TextRun({
                text: "4. REKOMENDASI KEBIJAKAN BERBASIS BUKTI (POLICY ACTION MATRIX)",
                bold: true,
                color: cNavyHex,
                size: 24,
                font: "Arial"
              })
            ]
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: policyTableRows
          }),

          // Closing Note
          new Paragraph({
            spacing: { before: 300 },
            children: [
              new TextRun({
                text: "RevDadas Fiscal Intelligence Platform — Dokumen bahan pertimbangan pengambilan keputusan fiskal daerah (B2G Decision Support). Di-generate secara otomatis oleh AI System.",
                italics: true,
                size: 16,
                color: cSlateHex
              })
            ]
          })
        ]
      }
    ]
  });

  // Pack and download
  const blob = await Packer.toBlob(doc);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const fileDate = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  const scope = filters.selectedProvinces.length === 1 ? filters.selectedProvinces[0].replace(/\s+/g, "_") : "Nasional";
  a.href = url;
  a.download = `RevDadas_Dossier_Fiskal_${scope}_${fileDate}.docx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
