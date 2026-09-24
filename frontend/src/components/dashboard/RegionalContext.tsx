import { useState } from "react";
import { MapPin, Briefcase, Target, ShieldCheck, AlertTriangle, QrCode, Layers, CheckCircle2, Globe } from "lucide-react";
import type { AnomalyRecord, HistoricalRecord } from "@/lib/types";
import { useLanguage } from "@/lib/LanguageContext";

interface RegionalContextProps {
  selectedProvinces: string[];
  anomalies?: AnomalyRecord[];
  historical?: HistoricalRecord[];
}

// Konteks DNA Makro Fiskal Konsolidasi Nasional (ketika semua provinsi dipilih)
const NATIONAL_DNA = {
  andalan: "Pajak Daerah (PKB, BBNKB, PBB-P2) & Retribusi Komersial (mendominasi 72,4% PAD konsolidasi nasional, terkonsentrasi di Pulau Jawa & Sumatera)",
  fokus_belanja: "Infrastruktur konektivitas logistik nasional, pemenuhan alokasi mandatory spending pendidikan/kesehatan, dan transformasi digital Pemda",
  target_historis: "Disparitas kemandirian tajam: Koridor Jawa (60–80%) vs Luar Jawa (15–35%, sangat bergantung transfer TKDD Pusat). Siklus serapan menumpuk di Q4",
  kebijakan: "Standardisasi integrasi kanal QRIS Pemda & KKPD via Satgas TP2DD, percepatan interoperabilitas SIPD-RI, dan mitigasi penumpukan idle cash di BPD",
  etpd_level: "Tahap Digital Nasional (Indeks 91.2% - 542 Pemda)"
};

// Kamus "DNA" fiskal per wilayah strategis
const REGIONAL_DNA: Record<string, { andalan: string; fokus_belanja: string; target_historis: string; kebijakan: string; etpd_level: string }> = {
  "DI Yogyakarta": {
    andalan: "Pajak Hotel & Restoran, Pajak Reklame, & Retribusi Wisata Budaya/Pendidikan",
    fokus_belanja: "Pengembangan cagar budaya, pariwisata heritage, dan transportasi perkotaan ramah lingkungan",
    target_historis: "Tinggi pada musim libur semester & akhir tahun (Q3–Q4)",
    kebijakan: "Perluasan QRIS pada retribusi destinasi wisata dan sistem monitoring tapping box pada hotel/restoran",
    etpd_level: "Tahap Digital (Indeks 94.8%)"
  },
  "DKI Jakarta": {
    andalan: "Pajak Kendaraan Bermotor (PKB), BPHTB, & Pajak Reklame/Komersial",
    fokus_belanja: "Transportasi publik terintegrasi (MRT/LRT), penanggulangan banjir, dan subsidi sosial",
    target_historis: "Stabil dan berbobot terbesar nasional, namun rentan pada transisi IKN",
    kebijakan: "Intensifikasi digital terpusat, pengawasan faktur BPHTB, dan penegakan pajak progresif kendaraan",
    etpd_level: "Tahap Digital (Indeks 98.2%)"
  },
  "Jawa Barat": {
    andalan: "Pajak Kendaraan Bermotor, BBNKB, & PBBKB (Pusat Industri & Manufaktur)",
    fokus_belanja: "Pendidikan, kesehatan (populasi terbesar), dan konektivitas kawasan industri Rebana",
    target_historis: "Stabil seiring pertumbuhan populasi dan ekspansi koridor logistik",
    kebijakan: "Program pemutihan denda pajak berkala, kemudahan integrasi pembayaran via perbankan digital",
    etpd_level: "Tahap Digital (Indeks 93.1%)"
  },
  "Jawa Timur": {
    andalan: "Pajak Kendaraan Bermotor (PKB), BBNKB, & PBBKB (Kawasan Industri & Agribisnis)",
    fokus_belanja: "Konektivitas pelabuhan, jalan lingkar industri, dan lumbung pangan daerah",
    target_historis: "Stabil dan tangguh terhadap gejolak ekonomi makro",
    kebijakan: "Intensifikasi e-Samsat pedesaan dan integrasi sistem perpajakan digital berbasis NIK",
    etpd_level: "Tahap Digital (Indeks 95.0%)"
  },
  "Bali": {
    andalan: "Pajak Hotel, Restoran, & Hiburan (Hospitality & Pariwisata Mancanegara)",
    fokus_belanja: "Infrastruktur penunjang pariwisata berkelanjutan dan pelestarian seni budaya",
    target_historis: "Sangat fluktuatif mengikuti siklus pariwisata global (peak season Juli–Agustus)",
    kebijakan: "Digitalisasi retribusi wisatawan asing dan sistem proteksi likuiditas kas darurat daerah",
    etpd_level: "Tahap Digital (Indeks 96.4%)"
  },
  "Kalimantan Timur": {
    andalan: "PBBKB, Dana Bagi Hasil (DBH) Pertambangan/Migas, & Pajak Air Permukaan",
    fokus_belanja: "Pembangunan penyangga IKN, konektivitas pelabuhan kargo, dan hilirisasi energi",
    target_historis: "Volatil karena sensitif terhadap pergerakan harga komoditas batubara/migas global",
    kebijakan: "Diversifikasi sumber PAD non-tambang dan pembentukan dana abadi daerah (sovereign wealth)",
    etpd_level: "Tahap Digital (Indeks 89.6%)"
  },
  "Sumatera Utara": {
    andalan: "PBBKB, Pajak Air Permukaan Industri, & Sektor Perkebunan/Agroindustri",
    fokus_belanja: "Konektivitas Pelabuhan Belawan, jalan lintas logistik, dan hilirisasi sawit",
    target_historis: "Siklikal mengikuti siklus panen dan ekspor komoditas perkebunan",
    kebijakan: "Audit tera meteran air permukaan korporasi dan digitalisasi pos retribusi perhubungan",
    etpd_level: "Tahap Digital (Indeks 88.5%)"
  },
  "Sulawesi Selatan": {
    andalan: "Pajak Kendaraan Bermotor & Retribusi Simpul Logistik Kawasan Timur",
    fokus_belanja: "Hub logistik pelabuhan Makassar New Port, modernisasi pertanian, dan rumah sakit rujukan",
    target_historis: "Tumbuh konsisten di atas rata-rata pertumbuhan fiskal Kawasan Timur Indonesia",
    kebijakan: "Pengembangan kanal QRIS pada seluruh pasar induk dan retribusi terminal kargo",
    etpd_level: "Tahap Digital (Indeks 91.2%)"
  },
  "Papua": {
    andalan: "Dana Bagi Hasil (DBH) Minerba, Otsus, & Pajak Air Permukaan Tambang",
    fokus_belanja: "Aksesibilitas wilayah pedalaman, fasilitas kesehatan dasar, dan vokasi putra daerah",
    target_historis: "Sangat bertumpu pada transfer pemerintah pusat dan royalti tambang",
    kebijakan: "Peningkatan kapasitas fiskal mandiri dan pengawasan realisasi penyerapan belanja modal",
    etpd_level: "Tahap Maju (Indeks 79.4%)"
  }
};

export default function RegionalContext({ selectedProvinces, anomalies = [], historical = [] }: RegionalContextProps) {
  const { lang, t } = useLanguage();
  if (!selectedProvinces || selectedProvinces.length === 0) return null;

  const isAllProvinces = selectedProvinces.length >= 30;

  // Active key: "NASIONAL" jika semua provinsi, atau nama provinsi spesifik
  const [activeKey, setActiveKey] = useState<string>(isAllProvinces ? "NASIONAL" : (selectedProvinces[0] || ""));

  const isNationalActive = isAllProvinces && (activeKey === "NASIONAL" || !selectedProvinces.includes(activeKey));
  const currentProv = isNationalActive 
    ? (lang === "en" ? "National Consolidation (38 Provinces)" : "Konsolidasi Nasional (38 Provinsi)") 
    : (selectedProvinces.includes(activeKey) ? activeKey : selectedProvinces[0] || "");

  const dna = isNationalActive ? NATIONAL_DNA : (REGIONAL_DNA[currentProv] || {
    andalan: "Pendapatan Asli Daerah (Pajak Kendaraan, Restoran, & Retribusi Umum)",
    fokus_belanja: "Penyelenggaraan pelayanan dasar, sarana pendidikan, dan pemeliharaan jalan",
    target_historis: "Bertumbuh moderat mengikuti pertumbuhan ekonomi regional",
    kebijakan: "Optimalisasi kanal pembayaran nontunai QRIS dan rekonsiliasi data wajib pajak",
    etpd_level: "Tahap Digital (Rata-rata Nasional)"
  });

  // Anomali terdeteksi untuk konteks aktif
  const provAnomalies = isNationalActive 
    ? anomalies.filter(a => a.Anomaly) 
    : anomalies.filter(a => a.Anomaly && a.Provinsi === currentProv);

  const impactedProvincesCount = isNationalActive 
    ? new Set(provAnomalies.map(a => a.Provinsi)).size 
    : 1;

  return (
    <div 
      className="animate-fade-in-up" 
      style={{ 
        margin: 0,
        padding: "18px 22px", 
        background: "#ffffff", 
        border: "1px solid #e2e8f0",
        borderRadius: 8,
        boxShadow: "0 1px 3px rgba(15, 23, 42, 0.03)"
      }}
    >
      {/* Header Bar */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "flex-start", 
        marginBottom: 14, 
        flexWrap: "wrap", 
        gap: 10 
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10, flex: "1 1 280px", minWidth: 0 }}>
          <div style={{ 
            background: "#eff6ff", 
            padding: 7, 
            borderRadius: 6, 
            color: "#0284c7", 
            display: "flex",
            flexShrink: 0,
            marginTop: 1
          }}>
            {isNationalActive ? (
              <Globe size={16} strokeWidth={2.2} />
            ) : (
              <MapPin size={16} strokeWidth={2.2} />
            )}
          </div>
          <div style={{ minWidth: 0 }}>
            <h4 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "#0f172a", lineHeight: 1.3 }}>
              {t("reg.title")} <span style={{ color: "#0284c7" }}>{currentProv}</span>
            </h4>
            <span style={{ fontSize: 11.5, color: "#64748b", display: "block", marginTop: 2, lineHeight: 1.4 }}>
              {isNationalActive 
                ? t("reg.sub_nat") 
                : t("reg.sub_prov")}
            </span>
          </div>
        </div>

      </div>

      {/* Provinsi Selector Chips */}
      {isAllProvinces ? (
        <div style={{ 
          display: "flex", 
          alignItems: "center",
          gap: 8, 
          marginBottom: 14, 
          flexWrap: "wrap"
        }}>
          {/* Chip Konsolidasi Nasional */}
          <button
            onClick={() => setActiveKey("NASIONAL")}
            style={{
              minHeight: 36,
              fontSize: 11.5,
              padding: "6px 14px",
              borderRadius: 6,
              border: isNationalActive ? "1.5px solid #0284c7" : "1px solid #cbd5e1",
              background: isNationalActive ? "#f0f9ff" : "#ffffff",
              color: isNationalActive ? "#0369a1" : "#475569",
              fontWeight: isNationalActive ? 700 : 500,
              cursor: "pointer",
              transition: "all 0.15s ease",
              display: "inline-flex",
              alignItems: "center",
              gap: 6
            }}
            aria-pressed={isNationalActive}
          >
            <Globe size={13} strokeWidth={2.2} color={isNationalActive ? "#0284c7" : "#64748b"} />
            <span>{t("reg.nat_btn")}</span>
          </button>

          {/* Quick-filter koridor ekonomi utama */}
          {["DKI Jakarta", "Jawa Barat", "Jawa Timur", "Bali", "Kalimantan Timur", "Sumatera Utara", "Sulawesi Selatan", "Papua"].map((prov) => {
            const isSelected = !isNationalActive && activeKey === prov;
            return (
              <button
                key={prov}
                onClick={() => setActiveKey(prov)}
                style={{
                  minHeight: 36,
                  fontSize: 11.5,
                  padding: "6px 12px",
                  borderRadius: 6,
                  border: isSelected ? "1.5px solid #0284c7" : "1px solid #cbd5e1",
                  background: isSelected ? "#f0f9ff" : "#ffffff",
                  color: isSelected ? "#0369a1" : "#475569",
                  fontWeight: isSelected ? 700 : 500,
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  transition: "all 0.15s ease",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 5
                }}
                aria-pressed={isSelected}
              >
                <MapPin size={12} strokeWidth={2.2} color={isSelected ? "#0284c7" : "#94a3b8"} />
                <span>{prov}</span>
              </button>
            );
          })}

          {/* Dropdown pemilih provinsi lainnya */}
          <select
            value={["DKI Jakarta", "Jawa Barat", "Jawa Timur", "Bali", "Kalimantan Timur", "Sumatera Utara", "Sulawesi Selatan", "Papua"].includes(activeKey) || isNationalActive ? "" : activeKey}
            onChange={(e) => {
              if (e.target.value) setActiveKey(e.target.value);
            }}
            style={{
              minHeight: 36,
              fontSize: 11.5,
              padding: "6px 10px",
              borderRadius: 6,
              border: !["DKI Jakarta", "Jawa Barat", "Jawa Timur", "Bali", "Kalimantan Timur", "Sumatera Utara", "Sulawesi Selatan", "Papua"].includes(activeKey) && !isNationalActive ? "1.5px solid #0284c7" : "1px solid #cbd5e1",
              background: !["DKI Jakarta", "Jawa Barat", "Jawa Timur", "Bali", "Kalimantan Timur", "Sumatera Utara", "Sulawesi Selatan", "Papua"].includes(activeKey) && !isNationalActive ? "#f0f9ff" : "#ffffff",
              color: !["DKI Jakarta", "Jawa Barat", "Jawa Timur", "Bali", "Kalimantan Timur", "Sumatera Utara", "Sulawesi Selatan", "Papua"].includes(activeKey) && !isNationalActive ? "#0369a1" : "#475569",
              fontWeight: 500,
              cursor: "pointer",
              outline: "none"
            }}
          >
            <option value="">{lang === "en" ? "Other Provinces (38 Regions)..." : "Provinsi Lainnya (38 Daerah)..."}</option>
            {selectedProvinces.map((prov) => (
              <option key={prov} value={prov}>{prov}</option>
            ))}
          </select>
        </div>
      ) : selectedProvinces.length > 1 ? (
        <div style={{ 
          display: "flex", 
          gap: 8, 
          marginBottom: 14, 
          overflowX: "auto", 
          paddingBottom: 4,
          WebkitOverflowScrolling: "touch"
        }}>
          {selectedProvinces.map((prov) => {
            const isSelected = currentProv === prov;
            return (
              <button
                key={prov}
                onClick={() => setActiveKey(prov)}
                style={{
                  minHeight: 36,
                  fontSize: 11.5,
                  padding: "6px 12px",
                  borderRadius: 6,
                  border: isSelected ? "1.5px solid #0284c7" : "1px solid #cbd5e1",
                  background: isSelected ? "#f0f9ff" : "#ffffff",
                  color: isSelected ? "#0369a1" : "#475569",
                  fontWeight: isSelected ? 700 : 500,
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  transition: "all 0.15s ease",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 5
                }}
                aria-pressed={isSelected}
              >
                <MapPin size={12} strokeWidth={2.2} color={isSelected ? "#0284c7" : "#94a3b8"} />
                <span>{prov}</span>
              </button>
            );
          })}
        </div>
      ) : null}

      {/* Grid 3 Kolom DNA (Clean & Non-cluttered) */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 280px), 1fr))", 
        gap: 24,
        paddingTop: 4
      }}>
        {/* Kolom 1: Sumber Andalan */}
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 11.5, fontWeight: 700, color: "#334155", marginBottom: 4, display: "flex", alignItems: "center", gap: 5 }}>
            <Briefcase size={13} color="#0284c7" /> {t("reg.pad_source")}
          </div>
          <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.5 }}>
            {dna.andalan}
          </div>
        </div>

        {/* Kolom 2: Karakteristik Target */}
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 11.5, fontWeight: 700, color: "#334155", marginBottom: 4, display: "flex", alignItems: "center", gap: 5 }}>
            <Target size={13} color="#d97706" /> {t("reg.absorption_pattern")}
          </div>
          <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.5 }}>
            {dna.target_historis}
          </div>
        </div>

        {/* Kolom 3: Fokus Kebijakan */}
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 11.5, fontWeight: 700, color: "#334155", marginBottom: 4, display: "flex", alignItems: "center", gap: 5 }}>
            <ShieldCheck size={13} color="#16a34a" /> {t("reg.policy_strategy")}
          </div>
          <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.5 }}>
            {dna.kebijakan}
          </div>
        </div>
      </div>

      {/* Footer Mini: Relevansi dengan Peta */}
      <div style={{ 
        marginTop: 12, 
        paddingTop: 8, 
        borderTop: "1px solid #f1f5f9", 
        fontSize: 11, 
        color: "#64748b", 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: 6
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <Layers size={12} color="#64748b" />
          <span>
            {t("reg.heatmap_status")} <b>
              {isNationalActive 
                ? (lang === "en" 
                    ? `${provAnomalies.length} Anomaly Posts Detected (${impactedProvincesCount} Regions)` 
                    : `${provAnomalies.length} Pos Anomali Terdeteksi (${impactedProvincesCount} Wilayah)`)
                : provAnomalies.length > 0 
                  ? (lang === "en" ? `${provAnomalies.length} Posts to Monitor` : `${provAnomalies.length} Pos Perlu Pantau`) 
                  : (lang === "en" ? "Optimal Realization" : "Realisasi Optimal")}
            </b>
          </span>
        </span>
        <span style={{ color: "#0369a1", fontWeight: 600 }}>
          {t("reg.tp2dd_mandate")}
        </span>
      </div>
    </div>
  );
}
