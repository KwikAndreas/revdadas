import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { DashboardFilters, PolicyRecommendation, AnomalyRecord } from "./types";
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

export function generatePDF(
  kpis: KPIProps,
  recs: PolicyRecommendation[],
  filters: DashboardFilters,
  bizData?: any
) {
  const doc = new jsPDF("p", "mm", "a4");

  // Colors - Premium Palette
  const primaryColor: [number, number, number] = [15, 23, 42]; // Slate 900
  const secondaryColor: [number, number, number] = [71, 85, 105]; // Slate 600
  const accentColor: [number, number, number] = [37, 99, 235]; // Blue 600
  const dangerColor: [number, number, number] = [220, 38, 38]; // Red 600
  const successColor: [number, number, number] = [22, 163, 74]; // Green 600
  const lightBg: [number, number, number] = [248, 250, 252]; // Slate 50

  // --- Cover / Header ---
  doc.setFillColor(...primaryColor);
  doc.rect(0, 0, 210, 45, "F");
  
  // Decorative line
  doc.setFillColor(...accentColor);
  doc.rect(0, 45, 210, 2, "F");

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(26);
  doc.setFont("helvetica", "bold");
  doc.text("RevDadas Executive Report", 15, 22);
  
  doc.setFontSize(11);
  doc.setTextColor(203, 213, 225); // Slate 300
  doc.setFont("helvetica", "normal");
  doc.text("Analisis Pendapatan Daerah, Proyeksi AI, dan Mitigasi Risiko Fiskal", 15, 32);

  // --- Document Meta / Filter Info ---
  doc.setTextColor(...primaryColor);
  doc.setFontSize(14);
  doc.setFont("helvetica", "bold");
  doc.text("Parameter Analisis", 15, 60);

  doc.setFontSize(10);
  doc.setTextColor(...secondaryColor);
  doc.setFont("helvetica", "normal");
  
  const d = new Date();
  const dateOptions: Intl.DateTimeFormatOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
  const formattedDate = d.toLocaleDateString("id-ID", dateOptions);
  
  doc.text(`Tanggal Laporan: ${formattedDate}`, 15, 68);
  
  const provs = filters.selectedProvinces.join(", ");
  const wrappedProvs = doc.splitTextToSize(`Target Wilayah: ${provs}`, 90);
  doc.text(wrappedProvs, 15, 74);
  
  doc.text(`Fokus Analisis: ${filters.selectedTaxType}`, 110, 68);
  doc.text(`Horizon Proyeksi: ${filters.forecastMonths} Bulan`, 110, 74);
  doc.text(`Tingkat Pencegahan Fraud: ${filters.fraudPreventionPct}%`, 110, 80);

  // --- Executive Summary (KPIs) ---
  doc.setTextColor(...primaryColor);
  doc.setFontSize(14);
  doc.setFont("helvetica", "bold");
  doc.text("Ringkasan Eksekutif (Tahun Berjalan)", 15, 96);

  const kpiData = [
    ["Total Realisasi Pendapatan", formatCurrency(kpis.totalRevenue)],
    ["Target Pencapaian", kpis.targetPercentage ? `${kpis.targetPercentage.toFixed(1)}%` : "Tidak ada target tersedia"],
    ["Total Proyeksi (AI Forecast)", formatCurrency(kpis.forecastTotal)],
    ["Kemandirian Fiskal (Rasio PAD)", `${kpis.kemandirianFiskal.toFixed(1)}%`],
    ["Total Risiko Anomali", `${kpis.anomalyPct.toFixed(2)}% (${kpis.anomalyCount} Kasus)`],
    ["Nilai Transaksi untuk Ditinjau", formatCurrency(kpis.potentialLoss)],
  ];

  autoTable(doc, {
    startY: 102,
    head: [["Indikator Kinerja Utama", "Nilai Terkini"]],
    body: kpiData,
    headStyles: { fillColor: accentColor, textColor: [255, 255, 255], fontStyle: "bold" },
    bodyStyles: { textColor: [15, 23, 42], fontSize: 10 },
    alternateRowStyles: { fillColor: lightBg },
    theme: "grid",
    styles: { cellPadding: 6 },
    columnStyles: {
      0: { fontStyle: "bold" },
      1: { halign: "right" }
    }
  });

  // --- Top Anomalies ---
  let finalY = doc.lastAutoTable.finalY + 20;
  
  if (kpis.anomalies && kpis.anomalies.length > 0) {
    if (finalY > 230) { doc.addPage(); finalY = 20; }

    // Sort anomalies by highest deviation (most anomalous first)
    const topAnomalies = [...kpis.anomalies].sort((a, b) => Math.abs(b.Deviasi ?? 0) - Math.abs(a.Deviasi ?? 0)).slice(0, 5);
    
    doc.setTextColor(...primaryColor);
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("Top 5 Deteksi Anomali & Risiko Tertinggi", 15, finalY);

    const anomalyBody = topAnomalies.map((a) => [
      `${a.Provinsi}\n${a.Tanggal.split("T")[0]}`, 
      a.Jenis_Pendapatan, 
      formatCurrency(a.Realisasi),
      a.Severity || "-",
      a.Alasan || "-"
    ]);

    autoTable(doc, {
      startY: finalY + 8,
      head: [["Wilayah/Tanggal", "Sektor/Akun", "Nilai Realisasi", "Severity", "Analisis Algoritma"]],
      body: anomalyBody,
      headStyles: { fillColor: dangerColor, textColor: [255, 255, 255], fontStyle: "bold" },
      columnStyles: {
        0: { cellWidth: 35 },
        1: { cellWidth: 40, fontStyle: "bold" },
        2: { cellWidth: 35, halign: "right" },
        3: { cellWidth: 25, halign: "center", fontStyle: "bold" },
        4: { cellWidth: "auto", fontSize: 9 }
      },
      didParseCell: function(data) {
        if (data.section === 'body' && data.column.index === 3) {
          if (data.cell.raw === 'Tinggi') data.cell.styles.textColor = dangerColor;
          else data.cell.styles.textColor = [180, 83, 9]; // Amber
        }
      },
      styles: { overflow: "linebreak", cellPadding: 5, fontSize: 10 },
      alternateRowStyles: { fillColor: [254, 252, 252] }, // Light red tint
      theme: "grid",
    });
    finalY = doc.lastAutoTable.finalY + 20;
  }

  // --- Policy Recommendations ---
  if (finalY > 230) { doc.addPage(); finalY = 20; }
  
  doc.setTextColor(...primaryColor);
  doc.setFontSize(14);
  doc.setFont("helvetica", "bold");
  doc.text("Rekomendasi Kebijakan (Berbasis AI)", 15, finalY);

  const recBody = recs.map((r) => [
    r.judul, 
    r.prioritas.toUpperCase(), 
    `${r.detail}\n\nDampak: ${r.indikator_dampak || "-"}`
  ]);

  autoTable(doc, {
    startY: finalY + 8,
    head: [["Area Kebijakan", "Urgensi", "Detail & Dampak"]],
    body: recBody,
    headStyles: { fillColor: primaryColor, textColor: [255, 255, 255], fontStyle: "bold" },
    columnStyles: {
      0: { cellWidth: 50, fontStyle: "bold" },
      1: { cellWidth: 25, fontStyle: "bold", halign: "center" },
      2: { cellWidth: "auto" },
    },
    didParseCell: function(data) {
      if (data.section === 'body' && data.column.index === 1) {
        if (data.cell.raw === 'TINGGI') data.cell.styles.textColor = dangerColor;
        else if (data.cell.raw === 'MENENGAH') data.cell.styles.textColor = [180, 83, 9];
        else data.cell.styles.textColor = successColor;
      }
    },
    styles: { overflow: "linebreak", cellPadding: 6, fontSize: 10 },
    alternateRowStyles: { fillColor: lightBg },
    theme: "grid",
  });
  
  finalY = doc.lastAutoTable.finalY + 20;

  // --- Business Sector Recommendations ---
  if (bizData && bizData.scored) {
    const provs = Object.keys(bizData.scored).filter(p => filters.selectedProvinces.includes(p));
    if (provs.length > 0) {
      if (finalY > 230) { doc.addPage(); finalY = 20; }
      
      doc.setTextColor(...primaryColor);
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Potensi Sektor Ekonomi Daerah", 15, finalY);

      const bizBody: any[] = [];
      provs.forEach(prov => {
        const sectors = bizData.scored[prov];
        if (sectors && sectors.length > 0) {
          const topSector = sectors[0]; // get the highest scored sector
          bizBody.push([
            prov,
            topSector.sektor,
            `${topSector.skor}/100`,
            topSector.alasan
          ]);
        }
      });

      autoTable(doc, {
        startY: finalY + 8,
        head: [["Provinsi", "Sektor Unggulan", "Skor AI", "Katalis / Alasan"]],
        body: bizBody,
        headStyles: { fillColor: [16, 185, 129], textColor: [255, 255, 255], fontStyle: "bold" }, // Emerald 500
        columnStyles: {
          0: { cellWidth: 35, fontStyle: "bold" },
          1: { cellWidth: 45, fontStyle: "bold" },
          2: { cellWidth: 25, halign: "center" },
          3: { cellWidth: "auto", fontSize: 9 },
        },
        styles: { overflow: "linebreak", cellPadding: 5, fontSize: 10 },
        alternateRowStyles: { fillColor: [240, 253, 244] }, // Light green tint
        theme: "grid",
      });
    }
  }

  // --- Footer (Applied to all pages) ---
  const pageCount = (doc as any).internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(...secondaryColor);
    // Draw top footer line
    doc.setDrawColor(226, 232, 240); // Slate 200
    doc.setLineWidth(0.5);
    doc.line(15, 285, 195, 285);
    
    doc.text(
      `RevDadas Analytics — Di-generate secara otomatis oleh AI System`,
      15,
      290
    );
    doc.text(
      `Halaman ${i} dari ${pageCount}`,
      195,
      290,
      { align: "right" }
    );
  }

  // Save the PDF
  const filename = `RevDadas_Executive_Report_${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}.pdf`;
  doc.save(filename);
}
