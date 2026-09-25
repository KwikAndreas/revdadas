"use client";

import React from "react";
import Link from "next/link";
import { ArrowRight, Activity, CheckCircle2 } from "lucide-react";
import "./landing.css";

export default function LandingPage() {
  return (
    <div className="landing-page">
      {/* ── Navigation Bar ───────────────────────────────────────── */}
      <header className="landing-nav">
        <div className="landing-container landing-nav-inner">
          <Link href="/" className="landing-logo">
            <div className="landing-logo-icon">
              <Activity className="w-5 h-5 text-white" strokeWidth={2.5} />
            </div>
            <div className="landing-logo-text">
              <h1>RevDadas</h1>
              <span>Revenue Daerah Cerdas</span>
            </div>
          </Link>
        </div>
      </header>

      {/* ── Hero Section ─────────────────────────────────────────── */}
      <section className="landing-hero">
        <div className="landing-container">
          <div className="hero-badge">
            <span className="hero-badge-dot"></span>
            <span>PIDI BI DIGDAYA x HACKATHON 2026 • BANK INDONESIA</span>
          </div>

          <h1 className="hero-headline">
            Intelijensi Fiskal Daerah Berbasis AI untuk{" "}
            <span className="hero-headline-accent">Optimalisasi Kas &amp; PAD</span>
          </h1>

          <p className="hero-subheadline">
            RevDadas mentransformasi data APBD DJPK Kemenkeu 38 provinsi di
            Indonesia melalui dekumulasi runtun waktu diskret, peramalan profil
            serapan berjangkar pagu, dan deteksi anomali multi-variat untuk
            mitigasi <em>idle cash</em> dan kebocoran pendapatan.
          </p>

          {/* Call to Action Button in the Middle */}
          <div className="hero-cta-wrapper">
            <Link href="/dashboard" className="hero-cta-primary" id="cta-hero-main">
              <span>Akses Dashboard RevDadas Sekarang</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>

          {/* Hero Metrics Ribbon */}
          <div className="hero-metrics-grid">
            <div className="hero-metric-item">
              <div className="hero-metric-value">38 Provinsi</div>
              <div className="hero-metric-label">Cakupan Spasial 100% RI</div>
            </div>
            <div className="hero-metric-item">
              <div className="hero-metric-value">17.000+</div>
              <div className="hero-metric-label">Titik Observasi DJPK</div>
            </div>
            <div className="hero-metric-item">
              <div className="hero-metric-value">70.2%</div>
              <div className="hero-metric-label">Akurasi Model (WAPE 29.8%)</div>
            </div>
            <div className="hero-metric-item">
              <div className="hero-metric-value">Rp 4.5+ T</div>
              <div className="hero-metric-label">Potensi Anomali Terdeteksi</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer className="landing-footer">
        <div className="landing-container landing-footer-inner">
          <div className="footer-copy">
            <strong>RevDadas</strong> &bull; Revenue Daerah Cerdas &bull; Karya{" "}
            <strong>Team BITGrow</strong> untuk{" "}
            <strong>PIDI BI DIGDAYA x HACKATHON 2026</strong> (Bank Indonesia).
          </div>
        </div>
      </footer>
    </div>
  );
}
