import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { DashboardFilters, PolicyRecommendation, AnomalyRecord, BusinessData, Meta } from "./types";
import { formatCurrency } from "./utils";

// Expose jspdf autoTable type
declare module "jspdf" {
  interface jsPDF {
    autoTable: (options: any) => jsPDF;
    lastAutoTable: { finalY: number };
  }
}

interface KPIProps {
  totalRevenue: number;
  targetPercentage?: number;
  forecastTotal: number;
  anomalyCount: number;
  potentialLoss: number;
  anomalyPct: number;
  kemandirianFiskal: number;
  savedRevenue: number;
  anomalies: AnomalyRecord[];
}

interface InsightDataProps {
  isOptimal?: boolean;
  kondisi?: string;
  rekomendasi?: string;
  metricLabel?: string;
  metricValue?: string;
  recoveryValue?: string;
}

export function generatePDF(
  kpis: KPIProps,
  recs: PolicyRecommendation[],
  filters: DashboardFilters,
  bizData?: BusinessData,
  insightData?: InsightDataProps,
  meta?: Meta
) {
  const doc = new jsPDF("p", "mm", "a4");

  // Modern Institutional B2G Palette (Bank Indonesia & Ministry of Finance style)
  const cNavy: [number, number, number] = [15, 23, 42];       // #0F172A Slate 900
  const cDeepBlue: [number, number, number] = [30, 58, 138];  // #1E3A8A Blue 900
  const cGold: [number, number, number] = [180, 83, 9];       // #B45309 Amber 700
  const cSlate: [number, number, number] = [71, 85, 105];     // #475569 Slate 600
  const cLightSlate: [number, number, number] = [241, 245, 249]; // #F1F5F9 Slate 100
  const cDanger: [number, number, number] = [185, 28, 28];    // #B91C1C Red 700
  const cSuccess: [number, number, number] = [4, 120, 87];    // #047857 Emerald 700
  const cBorder: [number, number, number] = [203, 213, 225];  // #CBD5E1 Slate 300

  const activeModelName = meta?.active_model_name || (meta?.active_model === "prophet" ? "Prophet (Secondary)" : "Theta Method (Primary)");

  // ─────────────────────────────────────────────────────────────
  // 1. INSTITUTIONAL HEADER & BANNER
  // ─────────────────────────────────────────────────────────────
  // Top Header Background
  doc.setFillColor(...cNavy);
  doc.rect(0, 0, 210, 42, "F");

  // Dual Accent Bands (Deep Blue & Gold)
  doc.setFillColor(...cDeepBlue);
  doc.rect(0, 42, 210, 2, "F");
  doc.setFillColor(...cGold);
  doc.rect(0, 44, 210, 1.2, "F");

  // Badge Category
  doc.setFontSize(8);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(217, 119, 6); // Amber 600
  doc.text("EXECUTIVE FISCAL INTELLIGENCE DOSSIER | B2G POLICY BRIEF", 15, 12);

  // Main Title
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(20);
  doc.setFont("helvetica", "bold");
  doc.text("RevDadas Fiscal Intelligence Report", 15, 22);

  // Subtitle
  doc.setFontSize(9.5);
  doc.setTextColor(203, 213, 225); // Slate 300
  doc.setFont("helvetica", "normal");
  doc.text("Evaluasi Kinerja Realisasi Pendapatan Daerah, Proyeksi Multivariat, dan Rekomendasi Strategis", 15, 30);
  doc.setFontSize(8);
  doc.setTextColor(148, 163, 184); // Slate 400
  doc.text("Klasifikasi: Dokumen Telaah Perencanaan Anggaran & Pengawasan Kas Daerah", 15, 36);

  // ─────────────────────────────────────────────────────────────
  // 2. PARAMETER METADATA BOX
  // ─────────────────────────────────────────────────────────────
  const d = new Date();
  const dateOptions: Intl.DateTimeFormatOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
  const formattedDate = d.toLocaleDateString("id-ID", dateOptions);

  doc.setFillColor(...cLightSlate);
  doc.roundedRect(15, 49, 180, 22, 2, 2, "F");
  doc.setDrawColor(...cBorder);
  doc.setLineWidth(0.3);
  doc.roundedRect(15, 49, 180, 22, 2, 2, "S");

  doc.setFontSize(8.5);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...cNavy);
  doc.text("Waktu Terbit:", 20, 55);
  doc.text("Lingkup Wilayah:", 20, 61);
  doc.text("Pos Rekening:", 20, 67);

  doc.setFont("helvetica", "normal");
  doc.setTextColor(...cSlate);
  doc.text(formattedDate, 52, 55);
  const provSummary = filters.selectedProvinces.length === 0 || filters.selectedProvinces.length > 5
    ? `${filters.selectedProvinces.length} Provinsi Terpilih`
    : filters.selectedProvinces.join(", ");
  doc.text(provSummary, 52, 61);
  doc.text(filters.selectedTaxType, 52, 67);

  // Right column of metadata
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...cNavy);
  doc.text("Model Proyeksi:", 112, 55);
  doc.text("Horizon Waktu:", 112, 61);
  doc.text("Mitigasi Fraud:", 112, 67);

  doc.setFont("helvetica", "normal");
  doc.setTextColor(...cSlate);
  doc.text(activeModelName, 142, 55);
  doc.text(`${filters.forecastMonths} Bulan ke Depan`, 142, 61);
  doc.text(`Target Efisiensi ${filters.fraudPreventionPct}%`, 142, 67);

  // ─────────────────────────────────────────────────────────────
  // 3. EXECUTIVE SYNTHESIS & STRATEGIC BRIEFING BOX
  // ─────────────────────────────────────────────────────────────
  let curY = 76;
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...cNavy);
  doc.text("1. Sintesis Strategis & Implikasi Fiskal", 15, curY);

  curY += 4;
  const synthBgColor: [number, number, number] = [248, 250, 252];
  doc.setFillColor(...synthBgColor);
  doc.setDrawColor(...cBorder);
  doc.setLineWidth(0.4);
  
  // Prepare narrative text
  const kondisiText = insightData?.kondisi || 
    `Realisasi pendapatan daerah tercatat sebesar ${formatCurrency(kpis.totalRevenue)} dengan tingkat kemandirian fiskal ${kpis.kemandirianFiskal.toFixed(1)}%. Model proyeksi memproyeksikan potensi kumulatif mencapai ${formatCurrency(kpis.forecastTotal)} dalam ${filters.forecastMonths} bulan mendatang.`;
  const rekomendasiText = insightData?.rekomendasi || 
    "Perkuat disiplin monitoring rekening kas daerah, integrasikan data wajib pajak antardaerah, dan optimalkan kanal pemungutan digital untuk mempercepat penyerapan target pendapatan.";

  const splitKondisi = doc.splitTextToSize(`Kondisi Fiskal: ${kondisiText}`, 170);
  const splitRekomendasi = doc.splitTextToSize(`Arahan Kebijakan: ${rekomendasiText}`, 170);
  
  const boxHeight = (splitKondisi.length * 4) + (splitRekomendasi.length * 4) + 14;
  doc.roundedRect(15, curY, 180, boxHeight, 2, 2, "FD");

  // Left accent bar
  doc.setFillColor(...cDeepBlue);
  doc.roundedRect(15, curY, 2.5, boxHeight, 1, 1, "F");

  doc.setFontSize(8.5);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...cNavy);
  let textY = curY + 6;

  doc.text(splitKondisi, 21, textY);
  textY += splitKondisi.length * 4 + 3;

  doc.setTextColor(...cDeepBlue);
  doc.setFont("helvetica", "bold");
  doc.text(splitRekomendasi, 21, textY);

  curY += boxHeight + 8;

  // ─────────────────────────────────────────────────────────────
  // 4. SCORECARD INDIKATOR KINERJA UTAMA (IKU FISKAL)
  // ─────────────────────────────────────────────────────────────
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...cNavy);
  doc.text("2. Matriks Indikator Kinerja Utama (IKU Fiskal)", 15, curY);

  curY += 2;
  const targetLabel = kpis.targetPercentage 
    ? `${kpis.targetPercentage.toFixed(1)}% (${kpis.targetPercentage >= 85 ? "On Track" : "Perlu Percepatan"})`
    : "Basis Estimasi Pro-Rata";

  const kpiTableData = [
    [
      "Total Realisasi Pendapatan",
      formatCurrency(kpis.totalRevenue),
      "Akumulasi penerimaan kas daerah dari seluruh sumber pos pendapatan tahun berjalan."
    ],
    [
      "Rasio Capaian Target Anggaran",
      targetLabel,
      kpis.targetPercentage && kpis.targetPercentage < 85
        ? "Realisasi di bawah 85%; direkomendasikan akselerasi penagihan aktif dan peninjauan objek pajak."
        : "Realisasi stabil sesuai lintasan kalender fiskal daerah."
    ],
    [
      `Proyeksi AI (${filters.forecastMonths} Bulan)`,
      formatCurrency(kpis.forecastTotal),
      `Estimasi arus penerimaan menggunakan ${activeModelName} dengan dekomposisi musiman 12 bulan.`
    ],
    [
      "Derajat Kemandirian Fiskal (PAD)",
      `${kpis.kemandirianFiskal.toFixed(1)}%`,
      kpis.kemandirianFiskal >= 40 
        ? "Kemandirian Tinggi: Daerah memiliki fleksibilitas pendanaan belanja modal yang solid."
        : "Kemandirian Rentan: Ketergantungan terhadap dana transfer pemerintah pusat masih dominan."
    ],
    [
      "Tingkat Paparan Anomali Kas",
      `${kpis.anomalyPct.toFixed(2)}% (${kpis.anomalyCount} Kasus)`,
      "Proporsi transaksi menyimpang (>2.5 sigma) yang memerlukan klarifikasi audit kepatuhan."
    ],
    [
      "Potensi Penyelamatan Kas Fiskal",
      formatCurrency(kpis.savedRevenue),
      `Nilai kas terpulihkan dengan target intervensi pencegahan kebocoran sebesar ${filters.fraudPreventionPct}%.`
    ]
  ];

  autoTable(doc, {
    startY: curY + 2,
    head: [["Indikator Fiskal Strategis", "Nilai / Rasio", "Interpretasi & Implikasi B2G"]],
    body: kpiTableData,
    headStyles: { 
      fillColor: cNavy, 
      textColor: [255, 255, 255], 
      fontStyle: "bold",
      fontSize: 8.5,
      cellPadding: 3.5
    },
    bodyStyles: { 
      textColor: cNavy, 
      fontSize: 8,
      cellPadding: 3.2
    },
    alternateRowStyles: { fillColor: [248, 250, 252] },
    columnStyles: {
      0: { cellWidth: 55, fontStyle: "bold" },
      1: { cellWidth: 42, fontStyle: "bold", halign: "right" },
      2: { cellWidth: "auto", fontSize: 7.5, textColor: cSlate }
    },
    theme: "grid",
  });

  // ─────────────────────────────────────────────────────────────
  // 5. TOP ANOMALIES & AUDIT RISK MATRIX
  // ─────────────────────────────────────────────────────────────
  let nextY = doc.lastAutoTable.finalY + 9;

  if (nextY > 220) {
    doc.addPage();
    nextY = 20;
  }

  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...cNavy);
  doc.text("3. Matriks Deteksi Anomali & Mitigasi Risiko Penerimaan", 15, nextY);

  if (kpis.anomalies && kpis.anomalies.length > 0) {
    const topAnomalies = [...kpis.anomalies]
      .sort((a, b) => Math.abs(b.Deviasi ?? 0) - Math.abs(a.Deviasi ?? 0))
      .slice(0, 5);

    const anomalyRows = topAnomalies.map((a) => [
      `${a.Provinsi}\n${a.Tanggal.split("T")[0]}`,
      a.Jenis_Pendapatan,
      formatCurrency(a.Realisasi),
      a.Severity || "Menengah",
      a.Alasan || "Deviasi pola musiman signifikan terdeteksi oleh algoritma."
    ]);

    autoTable(doc, {
      startY: nextY + 3,
      head: [["Wilayah / Tanggal", "Pos Rekening", "Nilai Realisasi", "Tingkat Risiko", "Catatan Investigasi Algoritma"]],
      body: anomalyRows,
      headStyles: { 
        fillColor: cDanger, 
        textColor: [255, 255, 255], 
        fontStyle: "bold",
        fontSize: 8.5,
        cellPadding: 3.5
      },
      bodyStyles: { 
        textColor: cNavy, 
        fontSize: 8, 
        cellPadding: 3 
      },
      columnStyles: {
        0: { cellWidth: 32, fontStyle: "bold" },
        1: { cellWidth: 40, fontStyle: "bold" },
        2: { cellWidth: 32, halign: "right" },
        3: { cellWidth: 24, halign: "center", fontStyle: "bold" },
        4: { cellWidth: "auto", fontSize: 7.5 }
      },
      didParseCell: function (data) {
        if (data.section === "body" && data.column.index === 3) {
          if (data.cell.raw === "Tinggi") {
            data.cell.styles.textColor = cDanger;
          } else {
            data.cell.styles.textColor = cGold;
          }
        }
      },
      alternateRowStyles: { fillColor: [254, 242, 242] }, // Soft red tint
      theme: "grid"
    });
  } else {
    // Elegant Zero-Anomalies Institutional Notification
    doc.setFillColor(240, 253, 244); // Light emerald
    doc.roundedRect(15, nextY + 3, 180, 14, 1.5, 1.5, "F");
    doc.setDrawColor(187, 247, 208);
    doc.roundedRect(15, nextY + 3, 180, 14, 1.5, 1.5, "S");
    doc.setFontSize(8.5);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(...cSuccess);
    doc.text("Status Rekening Terverifikasi Stabil:", 20, nextY + 9);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...cNavy);
    doc.text("Tidak ditemukan deviasi ekstrem (>2.5 sigma) pada data filter yang dipilih. Tetap pertahankan rekonsiliasi kas bulanan.", 20, nextY + 14);
    (doc as any).lastAutoTable = { finalY: nextY + 18 };
  }

  // ─────────────────────────────────────────────────────────────
  // 6. ACTIONABLE POLICY RECOMMENDATION MATRIX
  // ─────────────────────────────────────────────────────────────
  let policyY = doc.lastAutoTable.finalY + 9;
  if (policyY > 215) {
    doc.addPage();
    policyY = 20;
  }

  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...cNavy);
  doc.text("4. Rekomendasi Kebijakan Berbasis Bukti (Policy Action Matrix)", 15, policyY);

  const policyRows = recs.slice(0, 4).map((r) => [
    r.judul,
    r.prioritas.toUpperCase(),
    `${r.detail}\n\nIndikator Dampak: ${r.indikator_dampak || "Peningkatan efisiensi kepatuhan pajak daerah."}`
  ]);

  autoTable(doc, {
    startY: policyY + 3,
    head: [["Pilar Kebijakan", "Prioritas", "Arahan Rencana Aksi & Indikator Kunci"]],
    body: policyRows,
    headStyles: { 
      fillColor: cDeepBlue, 
      textColor: [255, 255, 255], 
      fontStyle: "bold",
      fontSize: 8.5,
      cellPadding: 3.5 
    },
    bodyStyles: { 
      textColor: cNavy, 
      fontSize: 8, 
      cellPadding: 3 
    },
    columnStyles: {
      0: { cellWidth: 46, fontStyle: "bold" },
      1: { cellWidth: 24, fontStyle: "bold", halign: "center" },
      2: { cellWidth: "auto", fontSize: 7.5 }
    },
    didParseCell: function (data) {
      if (data.section === "body" && data.column.index === 1) {
        if (data.cell.raw === "TINGGI") data.cell.styles.textColor = cDanger;
        else if (data.cell.raw === "SEDANG" || data.cell.raw === "MENENGAH") data.cell.styles.textColor = cGold;
        else data.cell.styles.textColor = cSuccess;
      }
    },
    alternateRowStyles: { fillColor: [248, 250, 252] },
    theme: "grid"
  });

  // ─────────────────────────────────────────────────────────────
  // 7. STRATEGIC SECTOR SYNERGIES (B2G & PRIVATE SECTOR)
  // ─────────────────────────────────────────────────────────────
  if (bizData && bizData.scored) {
    const provs = Object.keys(bizData.scored).filter((p) => filters.selectedProvinces.includes(p));
    if (provs.length > 0) {
      let bizY = doc.lastAutoTable.finalY + 9;
      if (bizY > 215) {
        doc.addPage();
        bizY = 20;
      }

      doc.setFontSize(11);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(...cNavy);
      doc.text("5. Pemetaan Potensi Sektor Ekonomi Unggulan Daerah", 15, bizY);

      const bizRows: any[] = [];
      provs.slice(0, 6).forEach((prov) => {
        const sectors = bizData.scored[prov];
        if (sectors && sectors.length > 0) {
          const top = sectors[0];
          bizRows.push([
            prov,
            top.sektor,
            `${top.skor}/100`,
            top.alasan || top.narasi || "Sektor pengungkit pertumbuhan ekonomi regional."
          ]);
        }
      });

      autoTable(doc, {
        startY: bizY + 3,
        head: [["Provinsi", "Sektor Unggulan Daerah", "Skor AI", "Katalis Fiskal & Potensi PAD"]],
        body: bizRows,
        headStyles: { 
          fillColor: [5, 150, 105], // Emerald 600
          textColor: [255, 255, 255], 
          fontStyle: "bold",
          fontSize: 8.5,
          cellPadding: 3.5 
        },
        bodyStyles: { 
          textColor: cNavy, 
          fontSize: 8, 
          cellPadding: 3 
        },
        columnStyles: {
          0: { cellWidth: 35, fontStyle: "bold" },
          1: { cellWidth: 42, fontStyle: "bold" },
          2: { cellWidth: 20, halign: "center", fontStyle: "bold" },
          3: { cellWidth: "auto", fontSize: 7.5 }
        },
        alternateRowStyles: { fillColor: [240, 253, 244] },
        theme: "grid"
      });
    }
  }

  // ─────────────────────────────────────────────────────────────
  // 8. RUNNING INSTITUTIONAL FOOTER (All Pages)
  // ─────────────────────────────────────────────────────────────
  const pageCount = (doc as any).internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(7.5);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...cSlate);

    // Top footer line
    doc.setDrawColor(203, 213, 225); // Slate 300
    doc.setLineWidth(0.4);
    doc.line(15, 284, 195, 284);

    doc.text(
      "RevDadas Fiscal Intelligence Platform — Bahan Telaah Kebijakan Fiskal (B2G Decision Support)",
      15,
      289
    );
    doc.text(
      `Halaman ${i} dari ${pageCount}`,
      195,
      289,
      { align: "right" }
    );
  }

  // Save the PDF
  const dateStr = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  const safeScope = filters.selectedProvinces.length === 1 ? filters.selectedProvinces[0].replace(/\s+/g, "_") : "Nasional";
  const filename = `RevDadas_Dossier_Fiskal_${safeScope}_${dateStr}.pdf`;
  doc.save(filename);
}

