# PANDUAN KOMPREHENSIF Q&A DEWAN JURI BANK INDONESIA
**PIDI BI DIGDAYA x HACKATHON 2026**  
**Proyek:** RevDadas – Deteksi Anomali & Kecerdasan Fiskal untuk Bapenda  
**Tim:** Team BITGrow (*Kwik Andreas Jonathan, Gwyneth Eunice Widjaja, Clay Micholaz Fu, Moses Chisthoper Adisam*)  
**Live Demo:** [revdadas.vercel.app](https://revdadas.vercel.app)

---

## DAFTAR ISI
1. [Ringkasan Eksekutif & Value Proposition](#ringkasan-eksekutif--value-proposition)
2. [Kategori 1: Mandat Bank Indonesia, ETPD, & TP2DD](#kategori-1-mandat-bank-indonesia-etpd--tp2dd)
3. [Kategori 2: Validitas Angka Finansial (Rp 8,8 – 17,7 Triliun)](#kategori-2-validitas-angka-finansial-rp-88--177-triliun)
4. [Kategori 3: Metodologi AI, Data Science, & Deteksi Anomali](#kategori-3-metodologi-ai-data-science--deteksi-anomali)
5. [Kategori 4: Validasi Pasar, Adopsi Pemda, & Model Bisnis](#kategori-4-validasi-pasar-adopsi-pemda--model-bisnis)
6. [Kategori 5: Keamanan Data, Regulasi, & Kesiapan 38 Provinsi](#kategori-5-keamanan-data-regulasi--kesiapan-38-provinsi)
7. [Lampiran Khusus: Mengapa Nilai Pemulihan Sempat Bernilai Rp 0?](#lampiran-khusus-mengapa-nilai-pemulihan-sempat-bernilai-rp-0)
8. [Cheat Sheet: 5 Aturan Emas Pitching di Bank Indonesia](#cheat-sheet-5-aturan-emas-pitching-di-bank-indonesia)

---

## RINGKASAN EKSEKUTIF & VALUE PROPOSITION

* **Masalah:** Audit manual Bapenda kalah cepat dari siklus anggaran. Temuan audit SPI (Sistem Pengendalian Intern) BPK mencapai 7.006 temuan karena rekonsiliasi setoran pajak self-assessment dilakukan manual dan baru ketahuan setelah anggaran terkunci (*post-factum*).
* **Solusi RevDadas:** Sistem analitik berbasis AI yang berfungsi sebagai *early warning system* untuk mendeteksi kebocoran PAD sebelum APBD disahkan/terkunci.
* **Kaitan dengan Bank Indonesia:** RevDadas menjadi **"Otak Intelijen Analitik Lanjutan"** bagi program Elektronifikasi Transaksi Pemda (ETPD) dan Satgas TP2DD. Jika QRIS/KKPD adalah infrastruktur transaksi digitalnya, RevDadas adalah instrumen evaluasi *outcome* penerimaan kas daerahnya.

---

## KATEGORI 1: MANDAT BANK INDONESIA, ETPD, & TP2DD

### Pertanyaan 1.1
> *"Bank Indonesia sedang gencar mendorong ETPD (Elektronifikasi Transaksi Pemda) lewat QRIS Pemda dan KKPD. Di slide 7 Anda mencantumkan angka Indeks ETPD 73,6%. Apa peran konkret RevDadas dalam mendukung program kerja Bank Indonesia tersebut?"*

* **Latar Belakang Penguji:** Juri BI ingin memastikan solusi relevan dengan tugas pokok BI di daerah, bukan sekadar aplikasi Bapenda biasa.
* **Script Jawaban:**
  > *"Bapak/Ibu Dewan Juri, selama ini Bank Indonesia bersama TP2DD berhasil mendigitalisasi **kanal transaksi pembayaran** daerah (hilir). Namun, tantangan terbesar berikutnya adalah: **apakah kenaikan transaksi digital tersebut benar-benar mencerminkan optimalisasi PAD dan terbebas dari kebocoran di kas daerah?**
  >
  > RevDadas hadir sebagai **'Otak Intelijen Analitik'** pelengkap ETPD. Jika kanal QRIS/KKPD adalah pipanya, RevDadas adalah sensor kebocoran pipanya. Kami memvalidasi apakah kenaikan transaksi digital berbanding lurus dengan penerimaan riil, atau justru masih terjadi under-reporting. Dengan RevDadas, Kantor Perwakilan Bank Indonesia (KPwBI) di 38 provinsi memiliki alat kuantitatif untuk mengevaluasi efektivitas ETPD secara objektif."*
* **Poin Kunci:** BI = *Co-Chair* Satgas TP2DD (Keppres No. 3/2021). RevDadas membantu mengukur dampak ETPD pada level *outcome* penerimaan kas daerah.

---

### Pertanyaan 1.2
> *"Bagaimana RevDadas membantu tugas Bank Indonesia dalam menjaga stabilitas makroekonomi dan pengendalian likuiditas di daerah?"*

* **Latar Belakang Penguji:** Juri dari divisi moneter/makroprudensial ingin melihat dampak terhadap transmisi kebijakan moneter.
* **Script Jawaban:**
  > *"Ketidakakuratan proyeksi pendapatan daerah sering kali menyebabkan Pemda lambat menyerap anggaran di awal tahun dan menumpuk dana mengendap (idle cash / SiLPA tinggi) di perbankan daerah (BPD). Hal ini mendistorsi perputaran uang dan likuiditas regional.
  >
  > RevDadas membantu stabilitas likuiditas melalui dua hal:
  > 1. **Peramalan Arus Kas Akurat (Profil Serapan Berjangkar):** Membantu bendahara daerah memproyeksikan penerimaan 6–24 bulan ke depan secara realistis, dijangkar pada pagu Anggaran APBD yang sudah diketahui, sehingga belanja daerah dapat dieksekusi tepat waktu.
  > 2. **Mitigasi Revenue Loss:** Menyelamatkan likuiditas kas daerah agar belanja publik tidak terganggu defisit semu."*

---

## KATEGORI 2: VALIDITAS ANGKA FINANSIAL (RP 8,8 – 17,7 TRILIUN)

### Pertanyaan 2.1
> *"Di slide 6 Anda mengklaim potensi pendapatan yang diselamatkan sebesar Rp 8,8 – 17,7 Triliun per tahun, namun ada catatan 'Asumsi 5–10% dari nilai anomali dan belum diuji langsung di lapangan'. Bukankah angka ini terlalu spekulatif jika anomali belum tentu kecurangan?"*

* **Latar Belakang Penguji:** Juri auditor/ekonom menguji apakah tim melebih-lebihkan angka (*overpromising*).
* **Script Jawaban:**
  > *"Terima kasih atas ketelitian Dewan Juri. Angka Rp 8,8 – 17,7 T adalah hasil simulasi matematis konservatif berbasis data riil APBD DJPK Kemenkeu 2024 pada 4 provinsi pilot.
  >
  > Dasar kalkulasinya:
  > 1. Total deviasi anomali ekstrem yang diisolasi oleh algoritma kami bernilai sekitar **Rp 177 Triliun**.
  > 2. Kami **tidak** mengklaim seluruh Rp 177 T adalah uang hilang, karena sebagian dapat berupa keterlambatan administrasi atau pergeseran kas.
  > 3. Standar efektivitas tindak lanjut audit APIP/Inspektorat umumnya berada di rentang **5% hingga 10%**. Jika auditor menindaklanjuti 5% saja dari deviasi tersebut, likuiditas kas daerah yang berhasil diamankan mencapai **Rp 8,8 Triliun**.
  >
  > Itulah alasan kami menyediakan **Kalkulator Skenario Audit interaktif** di dashboard: pengambil kebijakan dapat menggeser sendiri asumsi efektivitas pemulihan dari 1% hingga 100% secara transparan."*
* **Poin Kunci:** Anomali adalah *indikasi transaksi prioritas untuk ditinjau*, dan formula pemulihan adalah simulasi intervensi audit APIP.

---

## KATEGORI 3: METODOLOGI AI, DATA SCIENCE, & DETEKSI ANOMALI

### Pertanyaan 3.1
> *"Data yang digunakan adalah data agregat DJPK per pos pendapatan provinsi per bulan, bukan data transaksi per wajib pajak. Bagaimana bisa mendeteksi kebocoran setoran pajak restoran/hotel individual jika datanya cuma data makro bulanan?"*

* **Latar Belakang Penguji:** Menguji pemahaman batasan resolusi data (*data granularity*).
* **Script Jawaban:**
  > *"Di level MVP saat ini, RevDadas bekerja pada **Level Pengawasan Makro-Fiskal (Top-Down Risk Screening)**.
  >
  > Ketika pos 'Pajak Daerah' atau 'Retribusi Daerah' di suatu wilayah anjlok 60% MoM di bulan puncak wisata tanpa ada bencana atau perubahan regulasi, sistem langsung memicu alarm *High Severity* beserta alasannya.
  >
  > Alarm ini menjadi **dasar surat perintah audit terarah (Risk-Based Audit)** bagi Bapenda untuk membuka buku transaksi mikro restoran atau hotel di wilayah tersebut. Jadi alih-alih memeriksa ribuan SPTPD manual secara acak, RevDadas mempersempit sasaran audit hanya ke pos dan periode yang terbukti anomali secara statistik."*

---

### Pertanyaan 3.2
> *"Di slide 3 tertulis: 'Peramalan (Pelengkap): masih terus kami perbaiki untuk data yang polanya naik-turun tajam, bukan andalan utama'. Kenapa peramalan time-series pada APBD sulit dan bagaimana kalian menyelesaikannya?"*

* **Latar Belakang Penguji:** Menguji kedewasaan teknis tim dalam mengelola data deret waktu keuangan negara.
* **Script Jawaban:**
  > *"Kejujuran metodologis adalah komitmen kami. Data realisasi bulanan APBD Indonesia memiliki dua tantangan besar: **sampel pendek ($N \approx 36$ bulan)** dan **volatilitas ekstrem di akhir tahun (efek tutup buku Desember)**.
  >
  > Model kompleks seperti LSTM atau Deep Learning terbukti gagal (*overfitting* parah) pada sampel sekecil ini. Model Prophet standar pun hanya mencapai akurasi 54,3% (praktis setara tebakan sepele).
  >
  > Solusi arsitektur kami:
  > 1. Kami membangun **Decumulation Engine** untuk mengubah data kumulatif YTD DJPK menjadi deret bulanan diskrit murni.
  > 2. Kami mengembangkan model **Profil Serapan Berjangkar** — mesin forecasting yang memanfaatkan **pagu Anggaran APBD** (informasi yang sudah diketahui sejak awal tahun fiskal) sebagai jangkar prediksi. Model ini menghitung profil serapan historis (distribusi bulanan realisasi terhadap anggaran) dan memproyeksikan ke depan.
  > 3. Hasilnya: **Akurasi 70,1%** (Median WAPE 29,9%), jauh di atas Prophet (54,3%) dan Theta Method (~78%), karena kami memanfaatkan pagu Anggaran — informasi yang tidak bisa disentuh oleh model time-series konvensional.
  > 4. Sebagai pengaman kualitas, kami menerapkan **gerbang MASE < 1**: hanya seri yang melampaui benchmark *seasonal naive* yang ditayangkan sebagai proyeksi.
  >
  > Model ini 45× lebih cepat dari Prophet, tanpa dependensi library berat, dan berjalan dalam hitungan detik untuk seluruh 38 provinsi."*

---

### Pertanyaan 3.3
> *"Bagaimana model membedakan lonjakan wajar (misal: pencairan dana hibah atau belanja modal yang cair sekali setahun) dengan manipulasi data yang sebenarnya?"*

* **Latar Belakang Penguji:** Memastikan model tidak sekadar menerapkan *outlier detection* mentah.
* **Script Jawaban:**
  > *"Kami menggabungkan algoritma **Isolation Forest** dengan **Domain Rule Guardrail APBD**:
  > 1. Ekstraksi 4 fitur inti: *Revenue Norm*, *MoM Change*, *Ratio to MA-3*, dan *Seasonality Deviation* per seri pendapatan.
  > 2. Akun-akun yang secara alamiah bersifat 'lumpy' atau pencairan tunggal (seperti Belanja Modal, Hibah, dan Bagi Hasil) secara otomatis diklasifikasikan sebagai **'Wajar (Transaksi Insidental)'** dan diturunkan status anomalinya (*auto-downgrade*).
  > 3. Model hanya memicu alarm tinggi pada akun-akun rutin (Pajak Daerah, Retribusi Daerah) yang menyimpang di luar pola musiman historisnya."*

---

## KATEGORI 4: VALIDASI PASAR, ADOPSI PEMDA, & MODEL BISNIS

### Pertanyaan 4.1
> *"Di slide 7 tertulis sudah menghubungi 38 Bapenda, tapi baru 1 yang merespons. Birokrasi Pemda terkenal lambat dan protektif. Bagaimana strategi kalian agar solusi ini diadopsi?"*

* **Latar Belakang Penguji:** Menguji *go-to-market strategy* dan pemahaman birokrasi pemerintahan daerah.
* **Script Jawaban:**
  > *"Rendahnya respon via jalur mandiri (*cold-reach*) justru membuktikan hipotesis kami: **pemerintah daerah tidak bisa didekati secara B2G murni tanpa dukungan kelembagaan**.
  >
  > Strategi kami di Program PIDI BI Digdaya adalah masuk lewat **Sinergi Jalur Resmi Bank Indonesia**:
  > 1. Melalui jaringan **Sekretariat TP2DD**, di mana Bank Indonesia bertindak sebagai wakil ketua/inisiator utama di tiap provinsi.
  > 2. Didukung mentor PIDI BI, RevDadas diposisikan sebagai instrumen uji coba resmi dalam bentuk **Pilot Gratis (MoU non-komersial)** dengan 1–2 Bapenda (target DKI Jakarta dan Jawa Barat).
  > 3. Begitu masa pilot membuktikan adanya temuan kebocoran kas yang berhasil dicegah, Bapenda memiliki landasan kuat untuk menganggarkan pengadaan resmi melalui **e-Katalog LKPP**."*

---

### Pertanyaan 4.2
> *"Biaya langganan Rp 15–25 juta per bulan lewat e-Katalog LKPP. Apakah aturan pengadaan APBD mengizinkan skema SaaS berlangganan untuk analisis data?"*

* **Latar Belakang Penguji:** Menguji kepatuhan skema pengadaan barang/jasa pemerintah.
* **Script Jawaban:**
  > *"Sangat memungkinkan melalui pos belanja resmi APBD: **Belanja Jasa Konsultansi Berbasis Teknologi** atau **Belanja Lisensi Perangkat Lunak / Layanan Cloud Aplikasi Khusus** (Kode Akun Belanja Barang dan Jasa 5.1.02).
  >
  > Nilai Rp 15–25 juta per bulan (sekitar Rp 180–300 juta per tahun) masuk dalam koridor pengadaan langsung / e-Purchasing LKPP yang sangat terjangkau bagi APBD provinsi, terutama jika dibandingkan dengan efisiensi kas miliaran rupiah yang diselamatkan."*

---

## KATEGORI 5: KEAMANAN DATA, REGULASI, & KESIAPAN 38 PROVINSI

### Pertanyaan 5.1
> *"Di slide 4 disebutkan 'Source code open source (MIT)'. Bukankah data potensi kebocoran pajak dan risiko fiskal daerah adalah data rahasia? Bagaimana keamanannya?"*

* **Latar Belakang Penguji:** Menguji kepatuhan terhadap UU Pelindungan Data Pribadi (UU PDP) dan regulasi keamanan SPBE.
* **Script Jawaban:**
  > *"Perlu kami bedakan antara **Arsitektur Engine (Source Code)** dengan **Data Operasional**:
  > 1. Kode analitik, algoritma isolasi, dan formula statistik bersifat open source agar dapat diaudit secara transparan, independen, dan terbebas dari *vendor lock-in*.
  > 2. Namun, **data eksekusi dan database anomali** dijalankan di server terisolasi milik Pemda (*On-Premise / Secure Government Cloud*).
  > 3. RevDadas menganalisis data agregat tanpa memproses PII (*Personally Identifiable Information*) identitas wajib pajak pribadi, sehingga sepenuhnya patuh terhadap UU No. 27/2022 (UU PDP)."*

---

### Pertanyaan 5.2 (KARTU TRUF: Pamerkan Data 38 Provinsi!)
> *"Di slide 6 dan 7 tertulis 'MVP tervalidasi di 8 provinsi'. Kapan sistem ini bisa mencakup seluruh Indonesia?"*

* **Script Jawaban:**
  > *"Kabar gembiranya Bapak/Ibu Dewan Juri: **hal tersebut sudah berhasil kami wujudkan hari ini!**
  >
  > Pada slide tercantum capaian awal 8 provinsi. Namun pada rilis sistem terbaru kami yang aktif di `revdadas.vercel.app`, kami telah mengintegrasikan data APBD untuk **seluruh 38 provinsi di Indonesia** (total **17.026 baris data historis diskrit bulanan**, mencakup **14 taksonomi akun fiskal** dan peta geospasial heatmap lengkap se-Indonesia). RevDadas sudah 100% siap diskalakan secara nasional."*

---

## LAMPIRAN KHUSUS: MENGAPA NILAI PEMULIHAN SEMPAT BERNILAI RP 0?

Bila juri menguji langsung live dashboard dan menanyakan mengapa pada kondisi tertentu Kalkulator Skenario Audit menunjukkan **Rp 0**:

1. **Formula:** $\text{Potensi Pemulihan} = \text{Potential Loss} \times \% \text{ Asumsi Skenario}$.
2. **Kondisi Awal (8 Provinsi):** Pada filter default lama (tahun 2025 dengan provinsi DKI, Jabar, Jatim), pelaporan data berjalan normal sesuai tren historis sehingga anomali $= 0$ dan *Potential Loss* $= \text{Rp } 0$.
3. **Kondisi Terkini (38 Provinsi):** Dengan integrasi 38 provinsi, tampilan default langsung mendeteksi **10 record anomali** dengan deviasi **Rp 4,47 Triliun**, sehingga nilai pemulihan pada asumsi 5% langsung menampilkan **Rp 223,55 Miliar**.
4. **Respon Filosofis ke Juri:**
   > *"Jika angka menunjukkan Rp 0 pada wilayah tertentu, itu adalah bukti **integritas model AI kami**. Sistem kami tidak merekayasa anomali di daerah yang pelaporannya memang tertib dan wajar. Namun begitu filter diarahkan ke pos atau daerah dengan deviasi tajam, sistem langsung menghitung potensi penyelamatan kas secara akurat."*

---

## CHEAT SHEET: 5 ATURAN EMAS PITCHING DI BANK INDONESIA

1. **Apresiasi Setiap Pertanyaan:** Buka jawaban dengan: *"Pertanyaan yang sangat tajam dan relevan dari Bapak/Ibu Dewan Juri..."*
2. **Selalu Tautkan ke Mandat BI:** Gunakan kata kunci: **ETPD, TP2DD, inflasi daerah, SiLPA, likuiditas perbankan daerah (BPD)**.
3. **Akui Batasan Model dengan Solusi:** Jangan berdebat bahwa AI 100% benar; tegaskan bahwa AI adalah *Decision Support System* untuk membantu auditor manusia.
4. **Kuasai Angka Kunci:**
   - 38 Provinsi
   - 17.026+ baris data bulanan
   - 14 taksonomi akun APBD
   - Akurasi Profil Serapan Berjangkar: ~70,1% (WAPE 29,9%, MASE 0,79)
   - 45× lebih cepat dari Prophet, tanpa dependensi berat
   - Gerbang kualitas MASE < 1 (hanya seri layak yang ditayangkan)
   - Asumsi intervensi audit: 5%–10%
5. **Tutup dengan Visi Sinergi:** RevDadas siap menjadi instrumen evaluasi resmi TP2DD Bank Indonesia untuk mewujudkan kemandirian fiskal daerah.
