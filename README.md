# 🧠 Samre — Autonomous Cognitive Agent (v2.5.0)

**Samre** adalah agen kognitif otonom yang belajar, bernalar, dan berimajinasi secara mandiri.  

Ia dibangun di atas perpaduan **pengetahuan berbasis graf (RL)**, **penalaran simbolik**, **simulasi probabilistik**, dan **neuromodulasi**—memungkinkannya terus tumbuh tanpa aturan kaku.

## ✨ Fitur Utama

- 🧬 **Pengetahuan Hidup** – Samantic Garden v4.0 dengan node RL (REINFORCE + Multi‑Armed Bandit) yang terus belajar dari pengalaman.

- ⚖️ **Pengambilan Keputusan Adaptif** – `LinearActionValue` belajar online dari *reward*, tidak bergantung pada aturan hardcoded.

- 🌌 **Imajinasi & Simulasi** – Modul `Imagination` menjalankan ribuan skenario berlapis untuk mengeksplorasi kemungkinan.

- 🧘 **Penalaran Reflektif** – Aksi `CONTEMPLATE` membuktikan relasi logis antar konsep menggunakan mesin *backward chaining*.

- 📂 **Data Ingestion Otomatis** – File di folder `data/` (CSV, JSON, teks) otomatis diubah menjadi pengetahuan.

- 💤 **Konsolidasi Memori** – Tidur sibernetik: pruning, replay, abstraksi, dan evolusi kebijakan node.

- 🧠 **Neuromodulator** – Dopamin, serotonin, noradrenalin, dan asetilkolin memengaruhi pembelajaran dan eksplorasi.

- ⛰️ **Geologi Nyata** – Terhubung ke [Macrostrat API](https://macrostrat.org) untuk belajar dari data geologi dunia.

- 🛡️ **Executive Gatekeeper** – Lapisan keamanan yang mencegah aksi berbahaya sebelum dieksekusi.

---

## 🧱 Arsitektur Singkat

```
data/                  ← letakkan CSV/JSON/TXT di sini
core/
├── flock_of_thought.py   ← otak utama & siklus kognitif
├── cognitive_core.py     ← mesin vektor simbolik
├── sws_logic.py          ← penilaian aksi (RL linear)
├── samantic_garden.py    ← graf pengetahuan adaptif
├── imagination.py        ← mesin simulasi probabilistik
├── neuromodulator.py     ← sistem neuromodulasi
├── executive.py          ← gatekeeper keamanan
├── planner.py            ← perencana berbasis aturan
├── metacognition.py      ← refleksi & saran strategis
├── needs.py              ← kebutuhan internal (lapar, bosan, dll.)
├── chain_of_thought.py   ← mesin penalaran simbolik
├── neural_ecosystem.py   ← ANN otonom, ekosistem, kernel hybrid
├── reflection.py         ← evaluasi koherensi pemikiran
└── plan.py               ← struktur data rencana
tools/                   ← aktuator, file manager, API geologi
act/                     ← aktuator aksi (explore, learn, reason, evolve)
```

---

## 🚀 Memulai

### Prasyarat
- Python 3.8+
- Paket: `numpy`, `matplotlib`, `networkx`, `requests` (opsional untuk geologi)

```bash
pip install numpy matplotlib networkx requests
```

Menjalankan Samre

```bash
python main.py              # siklus 5 detik
python main.py --delay 2.0  # siklus lebih cepat
```

Saat pertama kali dijalankan, Samre akan bootstrap pengetahuan dari kode sumbernya sendiri, lalu langsung memindai folder data/.

---

## 📊 Menambahkan Data

1. Buat folder data/ (otomatis dibuat jika belum ada).
2. Letakkan file dataset di sana. Format yang didukung: .csv, .json, .txt, .md, .py, dll.
3. Samre akan otomatis membaca, mengonversi menjadi teks naratif (untuk CSV/JSON), dan memasukkannya ke dalam Samantic Garden.

Contoh:

```
data/
├── earthquakes.csv
├── reviews.json
└── my_notes.txt
```

---

## ⚙️ Konfigurasi Utama

Semua parameter penting bisa diubah langsung di dictionary config di SamanticGarden:

Parameter Default Keterangan
ingestion_reinforcement_threshold 0.5 Similaritas minimum untuk menggabungkan node (semakin kecil = semakin banyak node baru)
connection_similarity_threshold 0.5 Similaritas minimum untuk membentuk sinapsis
consolidation_pruning_threshold 0.02 Kekuatan sinapsis minimum agar tidak dipangkas
abstraction_cos_threshold 0.8 Kekuatan bidirectional minimum untuk menciptakan abstraksi

---

## 📝 Status Pengembangan

v2.5.0 adalah lompatan besar dari versi sebelumnya:

· ✅ Node pengetahuan berbasis RL menggantikan sub‑mind berbasis ANN
· ✅ Penilaian aksi belajar online (tidak lagi hardcoded)
· ✅ Aksi CONTEMPLATE untuk penalaran simbolik deduktif
· ✅ Data ingestion otomatis dari folder data/
· ✅ Perbaikan cognitive load death spiral
· ✅ Neuromodulator merespons kondisi fisik (efek domino)
· ✅ Threshold similarity diturunkan untuk memperkaya graf

Roadmap mendatang:

· Antarmuka percakapan (natural language)
· Planner adaptif (belajar dari pengalaman)
· CognitiveEngine belajar online (tidak statis)
· Visualisasi pengetahuan interaktif

---

🤝 Kontribusi

Proyek ini masih dalam pengembangan aktif. Jika kamu menemukan bug atau punya ide, silakan buka issue atau kirim pull request.

---

# Dikembangkan oleh [Alshihaf](https://alshihaf.github.io/asad-portfolio)