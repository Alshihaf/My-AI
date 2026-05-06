# Samre: Arsitektur Eksperimental untuk Agen AI Otonom

**Samre** adalah sebuah proyek penelitian yang bertujuan untuk membangun arsitektur agen kecerdasan buatan (AI) yang mampu beroperasi secara otonom. Berbeda dari model AI reaktif yang hanya merespons input, Samre dirancang untuk memiliki tujuan internal, kebutuhan, dan siklus pengambilan keputusan proaktif yang memungkinkannya bertindak secara mandiri untuk mencapai tujuannya.

## Visi Proyek

Tujuan utama proyek ini adalah untuk menjelajahi konsep-konsep inti dari agensi dan otonomi dalam sistem AI. Kami berupaya mensimulasikan "kesadaran" artifisial melalui interaksi dinamis antara beberapa komponen kognitif, termasuk:

- **Kebutuhan Internal:** Dorongan artifisial (seperti "rasa lapar" akan data atau "kebosanan") yang memotivasi agen untuk bertindak.
- **Pengambilan Keputusan Proaktif:** Kemampuan untuk mengevaluasi serangkaian tindakan yang mungkin dan memilih yang paling sesuai berdasarkan keadaan internal dan tujuan jangka panjang.
- **Pembelajaran Berbasis Imbalan:** Mekanisme untuk belajar dari keberhasilan atau kegagalan tindakan masa lalu untuk meningkatkan pengambilan keputusan di masa depan.
- **Evolusi Diri:** Kemampuan agen untuk menganalisis dan memodifikasi kode sumbernya sendiri untuk meningkatkan kinerjanya (fitur jangka panjang).

## Arsitektur Inti: Siklus Kognitif Otonom

Samre beroperasi dalam sebuah *loop* berkelanjutan yang disebut **Siklus Kognitif**. Setiap siklus merepresentasikan satu "denyut" kesadaran di mana agen mengevaluasi keadaan internalnya dan memutuskan tindakan apa yang akan diambil selanjutnya.

```graphviz
graph TD;
    A[Mulai Siklus] --> B{1. UPDATE_NEEDS<br>Perbarui Kebutuhan Internal};
    B --> C{2. SCORE_ACTIONS<br>Beri Skor Tindakan Potensial};
    C --> D{3. EVALUATE_ACTIONS<br>Persetujuan Eksekutif};
    D -- Disetujui --> E[4. EXECUTE_ACTION<br>Jalankan Tindakan Terpilih];
    D -- Ditolak --> F[Default ke REST];
    E --> G{5. RECORD_REWARD<br>Catat Hasil & Imbalan};
    F --> G;
    G --> H{6. UPDATE_LEARNING<br>Perbarui Sistem Pembelajaran};
    H --> I[Akhir Siklus];
    I --> A;
```

### Penjelasan Tahapan

1.  **UPDATE_NEEDS (Perbarui Kebutuhan):** Di awal setiap siklus, sistem meningkatkan tingkat kebutuhan internal seperti `hunger` (kebutuhan data baru), `boredom` (kebutuhan variasi tugas), `fatigue` (kebutuhan istirahat), dan `messiness` (kebutuhan untuk refactoring).
2.  **SCORE_ACTIONS (Beri Skor Tindakan):** Menggunakan modul `sws_logic.py`, sistem mensimulasikan dan memberi skor untuk setiap tindakan yang tersedia (`EXPLORE`, `EVOLVE`, `REST`, dll.). Skor ini adalah fungsi dari kebutuhan saat ini, level neuromodulator (dopamin, serotonin), dan data historis (keberhasilan tindakan di masa lalu).
3.  **EVALUATE_ACTIONS (Evaluasi Tindakan):** Tindakan dengan skor tertinggi diteruskan ke `executive.py`, yang bertindak sebagai "penjaga gerbang". Komponen ini dapat menolak tindakan jika dianggap terlalu berisiko atau tidak sesuai dengan prinsip sistem, seperti mencegah evolusi diri saat sistem sedang stres (kortisol tinggi).
4.  **EXECUTE_ACTION (Jalankan Tindakan):** Jika disetujui, tindakan dieksekusi. Ini bisa berupa menjelajahi sistem file, menganalisis kode, atau sekadar beristirahat untuk memulihkan "kelelahan".
5.  **RECORD_REWARD (Catat Imbalan):** Setelah tindakan selesai, hasilnya dievaluasi dan sebuah "imbalan" (reward) dihitung. Imbalan ini mengukur sejauh mana tindakan tersebut berhasil memenuhi kebutuhan yang mendasarinya.
6.  **UPDATE_LEARNING (Perbarui Sistem Pembelajaran):** Imbalan yang dihasilkan digunakan untuk memperbarui berbagai sistem pembelajaran. Tingkat keberhasilan tindakan di *Long-Term Memory* (LTM) diperbarui, dan level neuromodulator disesuaikan untuk memengaruhi keputusan di siklus berikutnya.

## Komponen Kunci

-   `Samre/main.py`: Titik masuk utama yang menjalankan siklus kognitif otonom.
-   `Samre/core/flock_of_thought.py`: Orkestrator utama. Mengelola seluruh siklus, dari pembaruan kebutuhan hingga eksekusi tindakan dan pembelajaran.
-   `Samre/core/needs.py`: Mensimulasikan dan mengelola kebutuhan internal agen.
-   `Samre/core/sws_logic.py`: "Jantung" pengambilan keputusan. Berisi logika untuk menilai dan memberi skor pada tindakan potensial.
-   `Samre/core/executive.py`: Bertindak sebagai fungsi eksekutif, memberikan persetujuan akhir untuk tindakan yang dipilih.
-   `Samre/core/neuromodulator.py`: Mensimulasikan pengaruh zat kimia saraf seperti dopamin (motivasi) dan kortisol (stres) pada pengambilan keputusan.

## Cara Menjalankan

1.  Pastikan Anda memiliki Python 3.6+ terinstal.
2.  *Clone* repositori ini.
3.  Jalankan agen dari direktori utama proyek:

    ```bash
    python Samre/main.py
    ```

Anda akan melihat *log output* di konsol yang menunjukkan setiap tahap dari siklus kognitif, termasuk kebutuhan yang diperbarui, skor tindakan, tindakan yang dipilih, dan imbalan yang dihasilkan. Untuk menghentikan agen, tekan `Ctrl+C`.

---
*Proyek ini bersifat eksperimental. Konsep-konsep seperti "kesadaran" dan "kebutuhan" adalah simulasi fungsional yang dirancang untuk menghasilkan perilaku otonom yang kompleks.*
