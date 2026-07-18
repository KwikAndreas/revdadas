import { MapPin, Briefcase, Target, ShieldCheck } from "lucide-react";

interface RegionalContextProps {
  selectedProvinces: string[];
}

// Kamus "DNA" daerah (bisa di-extend)
const REGIONAL_DNA: Record<string, { andalan: string; fokus_belanja: string; target_historis: string; kebijakan: string }> = {
  "Bali": {
    andalan: "Pajak Hotel dan Restoran (Sektor Pariwisata)",
    fokus_belanja: "Infrastruktur penunjang pariwisata dan pelestarian budaya",
    target_historis: "Kerap fluktuatif (sangat sensitif terhadap krisis/low-season)",
    kebijakan: "Proteksi ekosistem wisata, asuransi pariwisata, digitalisasi retribusi daerah wisata"
  },
  "DKI Jakarta": {
    andalan: "Pajak Kendaraan Bermotor, BPHTB, & Pajak Reklame/Iklan",
    fokus_belanja: "Transportasi massal (MRT/LRT), subsidi publik, dan pengendalian banjir",
    target_historis: "Stabil dan masif, namun rentan pada kemacetan dan perpindahan IKN",
    kebijakan: "Intensifikasi digital, pajak progresif kendaraan, ERP (Electronic Road Pricing)"
  },
  "Jawa Barat": {
    andalan: "Pajak Kendaraan Bermotor & Bea Balik Nama (Pusat Manufaktur/Industri)",
    fokus_belanja: "Pendidikan, kesehatan (populasi terbesar), dan infrastruktur industri",
    target_historis: "Stabil seiring pertumbuhan populasi dan daya beli kelas menengah",
    kebijakan: "Program diskon pajak kendaraan (pemutihan), kemudahan investasi kawasan industri"
  },
  "Kalimantan Timur": {
    andalan: "Pajak Bahan Bakar Kendaraan Bermotor (PBBKB) & Dana Bagi Hasil (Sektor Pertambangan)",
    fokus_belanja: "Pembangunan IKN, infrastruktur konektivitas antar-tambang/kota",
    target_historis: "Volatile (sangat dipengaruhi harga komoditas global seperti batu bara)",
    kebijakan: "Ekstensifikasi di luar tambang, pengelolaan royalti/DBH yang prudent"
  }
};

export default function RegionalContext({ selectedProvinces }: RegionalContextProps) {
  if (!selectedProvinces || selectedProvinces.length === 0) return null;

  // Untuk kesederhanaan, jika ada > 1, kita ambil provinsi pertama, 
  // atau bisa tampilkan pesan khusus. 
  // Idealnya dashboard memiliki "Primary Province"
  const primaryProv = selectedProvinces[0];
  const dna = REGIONAL_DNA[primaryProv] || {
    andalan: "Pendapatan Asli Daerah (Umum)",
    fokus_belanja: "Pelayanan Dasar (Kesehatan, Pendidikan, Infrastruktur)",
    target_historis: "Moderat",
    kebijakan: "Optimalisasi PAD konvensional dan efisiensi belanja rutin"
  };

  return (
    <div className="insight-card animate-fade-in-up" style={{ padding: "16px 20px", marginBottom: 24, background: "#f8fafc", borderLeft: "4px solid #3b82f6" }}>
      <h3 style={{ margin: "0 0 12px 0", fontSize: 16, color: "#1e293b", display: "flex", alignItems: "center", gap: 8 }}>
        <MapPin size={18} color="#3b82f6" /> 
        Konteks Wilayah: {primaryProv}
      </h3>
      <p style={{ margin: "0 0 16px 0", fontSize: 13, color: "#475569" }}>
        Insight, target, dan analisis anomali pada dashboard ini dikalibrasi berdasarkan "DNA" spesifik dari wilayah yang dipilih.
      </p>
      
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 4, display: "flex", alignItems: "center", gap: 4 }}>
            <Briefcase size={12} /> SUMBER ANDALAN
          </div>
          <div style={{ fontSize: 13, color: "#0f172a", fontWeight: 500 }}>{dna.andalan}</div>
        </div>
        
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 4, display: "flex", alignItems: "center", gap: 4 }}>
            <Target size={12} /> KARAKTERISTIK TARGET
          </div>
          <div style={{ fontSize: 13, color: "#0f172a", fontWeight: 500 }}>{dna.target_historis}</div>
        </div>

        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 4, display: "flex", alignItems: "center", gap: 4 }}>
            <ShieldCheck size={12} /> FOKUS BELANJA & KEBIJAKAN
          </div>
          <div style={{ fontSize: 13, color: "#0f172a", fontWeight: 500 }}>{dna.fokus_belanja} — {dna.kebijakan}</div>
        </div>
      </div>
    </div>
  );
}
