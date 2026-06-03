"""
Pembuat laporan PDF ringkas untuk RevDadas.

Menggunakan reportlab (murni Python, tanpa dependensi sistem) agar mudah
dijalankan di lingkungan Streamlit Cloud. Mengembalikan PDF sebagai bytes
sehingga bisa langsung dipakai st.download_button.
"""

import io
from datetime import datetime


def _fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1e12:
        return f"Rp {v/1e12:.1f} T"
    if abs(v) >= 1e9:
        return f"Rp {v/1e9:.1f} M"
    return f"Rp {v:,.0f}"


def build_pdf(summary: dict, recommendations: list) -> bytes:
    """
    summary: dict berisi KPI & konteks, mis.:
        {
          'provinsi': [...], 'jenis': [...], 'periode_bulan': 9,
          'total_revenue': float, 'forecast_total': float,
          'anomaly_pct': float, 'anomaly_count': int, 'potential_loss': float,
        }
    recommendations: list dict {judul, prioritas, detail}
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    styles = getSampleStyleSheet()
    NAVY = colors.HexColor("#1e3a5f")
    GRAY = colors.HexColor("#64748b")

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=NAVY, fontSize=20, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=GRAY, fontSize=9, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=14)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, textColor=GRAY, leading=11)

    el = []
    el.append(Paragraph("RevDadas — Laporan Revenue Daerah Cerdas", h1))
    el.append(Paragraph(
        f"Dibuat: {datetime.now().strftime('%d %B %Y, %H:%M')} &nbsp;|&nbsp; "
        f"Sumber data: DJPK Kemenkeu (SIKD)", sub))

    prov = ", ".join(summary.get("provinsi", [])) or "-"
    jenis = ", ".join(summary.get("jenis", [])) or "-"
    el.append(Paragraph("Konteks Analisis", h2))
    el.append(Paragraph(f"<b>Provinsi:</b> {prov}", body))
    el.append(Paragraph(f"<b>Jenis pendapatan:</b> {jenis}", body))
    el.append(Paragraph(f"<b>Periode prediksi:</b> {summary.get('periode_bulan','-')} bulan", body))
    el.append(Spacer(1, 6))

    # KPI table
    el.append(Paragraph("Ringkasan Indikator (KPI)", h2))
    kpi_rows = [
        ["Indikator", "Nilai"],
        ["Total Revenue (Aktual)", _fmt(summary.get("total_revenue", 0))],
        [f"Forecast {summary.get('periode_bulan','-')} Bulan", _fmt(summary.get("forecast_total", 0))],
        ["Akurasi Model (backtest)", f"{summary.get('akurasi', 0):.0f}%  "
            f"({summary.get('n_reliable', 0)}/{summary.get('n_series', 0)} seri andal)"],
        ["Risiko Fraud/Anomali", f"{summary.get('anomaly_pct', 0):.1f}%  ({summary.get('anomaly_count', 0)} catatan)"],
        ["Revenue Loss Terdeteksi", _fmt(summary.get("potential_loss", 0))],
    ]
    t = Table(kpi_rows, colWidths=[90*mm, 80*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    el.append(t)

    # Recommendations
    el.append(Paragraph("Rekomendasi Kebijakan", h2))
    pcolor = {"Tinggi": colors.HexColor("#dc2626"),
              "Sedang": colors.HexColor("#d97706"),
              "Rendah": colors.HexColor("#16a34a")}
    for i, r in enumerate(recommendations, 1):
        badge = r.get("prioritas", "Rendah")
        el.append(Paragraph(
            f'<b>{i}. {r["judul"]}</b> '
            f'<font color="{pcolor.get(badge, GRAY).hexval()}">[{badge}]</font>', body))
        el.append(Paragraph(r["detail"], body))
        el.append(Spacer(1, 5))

    el.append(Spacer(1, 10))
    el.append(Paragraph(
        "Catatan: data bulanan mencakup 2023–2025 (2021–2022 hanya tersedia tahunan "
        "di SIKD sehingga dikecualikan); 2025 kemungkinan masih preliminer. "
        "Rekomendasi bersifat indikatif sebagai bahan diskusi, bukan keputusan final.",
        small))

    doc.build(el)
    buf.seek(0)
    return buf.getvalue()
