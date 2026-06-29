import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { DashboardFilters, PolicyRecommendation } from "./types";
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
  forecastTotal: number;
  anomalyCount: number;
  potentialLoss: number;
  anomalyPct: number;
}

export function generatePDF(
  kpis: KPIProps,
  recs: PolicyRecommendation[],
  filters: DashboardFilters
) {
  const doc = new jsPDF("p", "mm", "a4");

  // Colors
  const primaryColor: [number, number, number] = [30, 58, 95]; // #1e3a5f
  const secondaryColor: [number, number, number] = [100, 116, 139]; // #64748b
  const accentColor: [number, number, number] = [185, 28, 28]; // #b91c1c
  const greenColor: [number, number, number] = [22, 101, 52]; // #166534

  // --- Header ---
  doc.setFillColor(248, 250, 252);
  doc.rect(0, 0, 210, 36, "F");
  
  doc.setDrawColor(...primaryColor);
  doc.setLineWidth(1);
  doc.line(0, 36, 210, 36);

  doc.setTextColor(...primaryColor);
  doc.setFontSize(22);
  doc.setFont("helvetica", "bold");
  doc.text("RevDadas Analytics Report", 15, 20);
  
  doc.setFontSize(11);
  doc.setTextColor(...secondaryColor);
  doc.setFont("helvetica", "normal");
  doc.text("Sistem Pendukung Keputusan Eksekutif & Deteksi Risiko Fiskal", 15, 28);

  // --- Summary ---
  doc.setTextColor(...primaryColor);
  doc.setFontSize(14);
  doc.setFont("helvetica", "bold");
  doc.text("Ringkasan Laporan", 15, 50);

  doc.setFontSize(10);
  doc.setTextColor(...secondaryColor);
  doc.setFont("helvetica", "normal");
  
  const d = new Date();
  const dateOptions: Intl.DateTimeFormatOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
  const formattedDate = d.toLocaleDateString("id-ID", dateOptions);
  
  doc.text(`Tanggal Cetak: ${formattedDate}`, 15, 58);
  doc.text(`Cakupan Wilayah: ${filters.selectedProvinces.join(", ")}`, 15, 64);
  doc.text(`Fokus Pendapatan: ${filters.selectedTaxType}`, 15, 70);
  doc.text(`Horizon Proyeksi: ${filters.forecastMonths} Bulan ke Depan`, 15, 76);

  // --- KPI Table ---
  autoTable(doc, {
    startY: 86,
    head: [["Metrik Indikator", "Nilai Terkini"]],
    body: [
      ["Total Penerimaan Historis (Realisasi)", formatCurrency(kpis.totalRevenue)],
      ["Proyeksi Pendapatan Mendatang", formatCurrency(kpis.forecastTotal)],
      ["Risiko Kebocoran (Fraud/Anomali)", `${kpis.anomalyPct.toFixed(2)}% (${kpis.anomalyCount} titik terdeteksi)`],
      ["Potensi Recovery Arus Kas", formatCurrency(kpis.potentialLoss)],
    ],
    headStyles: { fillColor: primaryColor, textColor: [255, 255, 255], fontStyle: "bold" },
    bodyStyles: { textColor: [15, 23, 42], fontSize: 10 },
    alternateRowStyles: { fillColor: [248, 250, 252] },
    theme: "grid",
    styles: { cellPadding: 6 },
  });

  // --- Policy Recommendations ---
  const finalY = doc.lastAutoTable.finalY + 20;
  doc.setTextColor(...primaryColor);
  doc.setFontSize(14);
  doc.setFont("helvetica", "bold");
  doc.text("Rekomendasi Kebijakan Strategis", 15, finalY);

  const recBody = recs.map((r) => [r.judul, r.prioritas.toUpperCase(), r.detail]);

  autoTable(doc, {
    startY: finalY + 8,
    head: [["Area Kebijakan", "Urgensi", "Detail Rekomendasi"]],
    body: recBody,
    headStyles: { fillColor: accentColor, textColor: [255, 255, 255], fontStyle: "bold" },
    columnStyles: {
      0: { cellWidth: 50, fontStyle: "bold" },
      1: { cellWidth: 30, fontStyle: "bold", halign: "center" },
      2: { cellWidth: "auto" },
    },
    didParseCell: function(data) {
      if (data.section === 'body' && data.column.index === 1) {
        if (data.cell.raw === 'TINGGI') data.cell.styles.textColor = accentColor;
        else if (data.cell.raw === 'MENENGAH') data.cell.styles.textColor = [180, 83, 9]; // amber-700
        else data.cell.styles.textColor = greenColor;
      }
    },
    styles: { overflow: "linebreak", cellPadding: 6, fontSize: 10 },
    alternateRowStyles: { fillColor: [254, 252, 252] }, // very light red tint
    theme: "grid",
  });

  // --- Footer ---
  const pageCount = (doc as any).internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(...secondaryColor);
    doc.text(
      `Dicetak oleh RevDadas AI — Halaman ${i} dari ${pageCount}`,
      105,
      290,
      { align: "center" }
    );
  }

  // Save the PDF
  const filename = `RevDadas_Report_${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}.pdf`;
  doc.save(filename);
}
