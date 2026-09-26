import Link from "next/link";
import "./landing.css";
import { LogoMark, ArrowRightIcon } from "@/components/landing/Icons";
import { getLandingStats, idNumber, idRupiahShort } from "@/lib/landingStats";
import { EVENT_NAME, REPO_URL, SITE_DESCRIPTION, SITE_NAME, SITE_URL, TEAM } from "@/lib/site";

export default function LandingPage() {
  const stats = getLandingStats();

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "SoftwareApplication",
        name: SITE_NAME,
        alternateName: "Revenue Daerah Cerdas",
        url: SITE_URL,
        description: SITE_DESCRIPTION,
        applicationCategory: "BusinessApplication",
        operatingSystem: "Web",
        inLanguage: "id-ID",
        creator: { "@id": `${SITE_URL}/#team` },
      },
      {
        "@type": "Organization",
        "@id": `${SITE_URL}/#team`,
        name: TEAM.name,
        url: SITE_URL,
        sameAs: [REPO_URL],
        parentOrganization: { "@type": "CollegeOrUniversity", name: TEAM.university },
        member: TEAM.members.map((m) => ({ "@type": "Person", name: m.name, jobTitle: m.role })),
      },
    ],
  };

  return (
    <div className="landing-page lp">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />

      <header className="lp-nav">
        <div className="lp-shell lp-nav-inner">
          <Link href="/" className="lp-logo" aria-label="RevDadas, beranda">
            <span className="lp-logo-mark">
              <LogoMark size={20} />
            </span>
            <span className="lp-logo-text">
              <strong>RevDadas</strong>
              <span>Pendapatan Daerah Cerdas</span>
            </span>
          </Link>
          <span className="lp-event">{EVENT_NAME}</span>
        </div>
      </header>

      <main className="lp-shell lp-main">
        <section className="lp-hero" aria-labelledby="lp-title">
          <div className="lp-badge">
            <span className="lp-badge-dot" />
            <span>Sistem Deteksi Dini Fiskal Daerah</span>
          </div>

          <h1 id="lp-title" className="lp-title">
            Deteksi <span className="lp-highlight">Pendapatan Daerah yang Meleset</span> Sebelum Tutup Tahun
          </h1>
          <p className="lp-lead">
            Proyeksi penerimaan dan deteksi anomali <strong>APBD {stats.provinces} provinsi</strong> berbasis data DJPK Kemenkeu — menyatukan sinergi pengawasan <strong>Bapenda, APIP, dan Bank Indonesia</strong>.
          </p>

          <Link href="/dashboard" className="lp-btn lp-btn-primary" id="cta-hero-main">
            <span>Buka Dashboard</span>
            <ArrowRightIcon size={16} />
          </Link>

          <dl className="lp-facts">
            <div>
              <dt>Observasi Bulanan</dt>
              <dd className="lp-fact-value">{idNumber(stats.observations)}</dd>
              <dd className="lp-fact-note">APBD {stats.provinces} Provinsi &bull; DJPK</dd>
            </div>
            <div>
              <dt>Akurasi Proyeksi</dt>
              <dd className="lp-fact-value">{idNumber(stats.accuracyPct, 1)}%</dd>
              <dd className="lp-fact-note">Median {idNumber(stats.reliableSeries)} Seri Teruji</dd>
            </div>
            <div>
              <dt>Deviasi Anomali</dt>
              <dd className="lp-fact-value">{idRupiahShort(stats.anomalyDeviation)}</dd>
              <dd className="lp-fact-note">TA {stats.latestYear} &bull; {stats.anomalyCount} Pos Terdeteksi</dd>
            </div>
          </dl>
        </section>
      </main>

      <section className="lp-credit" aria-label="Pembuat">
        <div className="lp-shell lp-credit-inner">
          <p className="lp-credit-title">
            Dibuat oleh <strong>{TEAM.name}</strong>, {TEAM.university}
          </p>
          <ul className="lp-credit-names">
            {TEAM.members.map((m) => (
              <li key={m.name}>
                {m.name}
                <span>{m.role}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <footer className="lp-foot">
        <div className="lp-shell lp-foot-inner">
          <p>Sumber data: realisasi APBD, DJPK Kementerian Keuangan RI.</p>
          <p>&copy; 2026 {SITE_NAME}</p>
        </div>
      </footer>
    </div>
  );
}
