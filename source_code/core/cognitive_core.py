"""
Cognitive Core - Mesin Penalaran Vektor Simbolik Tingkat Tinggi
Dilengkapi dengan:
- Multi‑Head Self‑Attention untuk fokus selektif pada dimensi internal.
- Working Memory dengan mekanisme lupa terkendali.
- Plasticity Control (Elastic Weight Consolidation) untuk mencegah catastrophic forgetting.
- Reflective Monitoring: mengukur kebingungan (perplexity) dan memicu restrukturisasi.
"""

import math
import random
from collections import deque
from typing import List, Tuple, Optional

class CognitiveAttention:
    """Multi‑Head Attention untuk vektor keadaan kognitif."""
    def __init__(self, dim: int, heads: int = 4):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0, "Dimensi harus habis dibagi jumlah head"

        # Matriks proyeksi untuk query, key, value
        self.W_q = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
        self.W_k = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
        self.W_v = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
        self.W_o = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]

    def _matmul(self, A, B):
        """Perkalian matriks‑vektor sederhana."""
        if isinstance(B[0], (int, float)):
            return [sum(a * b for a, b in zip(row, B)) for row in A]
        else:
            # Perkalian matriks‑matriks (untuk keperluan internal)
            return [[sum(a * b for a, b in zip(row_A, col_B)) for col_B in zip(*B)] for row_A in A]

    def _softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def attend(self, state: List[float]) -> List[float]:
        """Menghasilkan representasi yang telah diberi atensi terhadap state."""
        # Proyeksi linear
        q = self._matmul(self.W_q, state)
        k = self._matmul(self.W_k, state)
        v = self._matmul(self.W_v, state)

        # Pecah menjadi head (untuk kesederhanaan, kita tidak split secara eksplisit,
        # tetapi gunakan pendekatan grouped attention dengan dimensi terpisah)
        head_outputs = []
        for h in range(self.heads):
            start = h * self.head_dim
            end = start + self.head_dim
            q_h = q[start:end]
            k_h = k[start:end]
            v_h = v[start:end]

            # Attention score: dot product antara query dan key
            score = self._dot(q_h, k_h) / math.sqrt(self.head_dim)
            # Dalam implementasi nyata, kita akan menggunakan softmax atas semua token,
            # tetapi di sini kita hanya punya satu vektor → self‑attention sederhana.
            # Untuk membuatnya lebih ekspresif, kita gunakan non‑linearitas.
            attention_weight = 1.0 / (1.0 + math.exp(-score))  # sigmoid

            # Output head = attention_weight * value
            head_out = [attention_weight * vi for vi in v_h]
            head_outputs.extend(head_out)

        # Proyeksi output
        attended = self._matmul(self.W_o, head_outputs)
        return attended

    def _dot(self, a, b):
        return sum(ai * bi for ai, bi in zip(a, b))


class WorkingMemory:
    """Penyimpanan jangka pendek dengan pembusukan eksponensial."""
    def __init__(self, capacity: int, decay: float = 0.9):
        self.capacity = capacity
        self.decay = decay
        self.buffer = deque(maxlen=capacity)

    def write(self, item: List[float]):
        self.buffer.append(item)

    def read(self) -> List[float]:
        """Membaca memori yang telah didekay, menghasilkan vektor gabungan."""
        if not self.buffer:
            return []
        # Gabungkan dengan bobot menurun
        combined = [0.0] * len(self.buffer[0])
        weight_sum = 0.0
        for i, mem in enumerate(reversed(self.buffer)):
            w = self.decay ** i
            weight_sum += w
            for j, val in enumerate(mem):
                combined[j] += w * val
        # Normalisasi
        if weight_sum > 0:
            combined = [c / weight_sum for c in combined]
        return combined


class CognitiveEngine:
    """
    Mesin Kognitif Vektor dengan:
    - Memori Semantik (matriks bobot) yang dapat belajar.
    - Attention untuk fokus internal.
    - Working memory untuk konteks jangka pendek.
    - Elastic Weight Consolidation (EWC) untuk mencegah lupa drastis.
    - Reflective monitoring (perplexity & entropy).
    """
    def __init__(self, dimensionality: int = 64, attention_heads: int = 4):
        self.dim = dimensionality
        self.attention = CognitiveAttention(self.dim, heads=attention_heads)
        self.working_memory = WorkingMemory(capacity=10, decay=0.85)

        # Memori semantik (bobot internal)
        self.memory = [[(i * j * 0.1) % 1.0 for j in range(self.dim)] for i in range(self.dim)]
        # State internal saat ini
        self.state = [((i * 0.5) % 1.0) for i in range(self.dim)]
        # Parameter kreativitas (rasio emas termodulasi)
        self.entropy = 0.618

        # Untuk EWC: menyimpan bobot penting dan Fisher information
        self.fisher_diag = [[1.0 for _ in range(self.dim)] for _ in range(self.dim)]
        self.optimal_memory = [row[:] for row in self.memory]
        self.ewc_lambda = 0.4

        # Statistik reflektif
        self.perplexity_history = deque(maxlen=20)
        self.last_surprise = 0.0

    def _dot(self, a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def _activate(self, x: float) -> float:
        # Sigmoid dengan temperatur adaptif
        temp = 1.0 + self.entropy
        return 1.0 / (1.0 + math.exp(-x / temp))

    def _transform(self, input_vector: List[float]) -> List[float]:
        """Proyeksi linear melalui memori semantik + aktivasi."""
        new_state = []
        for row in self.memory:
            proj = self._dot(row, input_vector)
            new_state.append(self._activate(proj))
        return new_state

    def _synthesize(self, v1: List[float], v2: List[float]) -> List[float]:
        """Rekombinasi stokastik dengan pengaruh entropi."""
        return [((a + b) / 2.0) * (1.0 + self.entropy * 0.1) for a, b in zip(v1, v2)]

    def _update_ewc(self, new_memory: List[List[float]]):
        """Memperbarui bobot dengan regularisasi EWC."""
        for i in range(self.dim):
            for j in range(self.dim):
                # Gradient dari loss terhadap bobot (di sini disederhanakan)
                delta = new_memory[i][j] - self.memory[i][j]
                # EWC penalty: lambda * F * (theta - theta_opt)^2
                penalty = self.ewc_lambda * self.fisher_diag[i][j] * (self.memory[i][j] - self.optimal_memory[i][j])
                self.memory[i][j] += delta - penalty

    def think(self, cycles: int = 10):
        """Satu siklus berpikir: abstraksi, atensi, penalaran, kreativitas, integrasi."""
        for _ in range(cycles):
            # 1. Recall working memory
            wm_context = self.working_memory.read()
            if wm_context:
                # Campurkan konteks memori kerja ke state (dengan bobot)
                blended = [0.7 * s + 0.3 * w for s, w in zip(self.state, wm_context)]
            else:
                blended = self.state

            # 2. Atensi internal: fokus pada dimensi penting
            attended_state = self.attention.attend(blended)

            # 3. Abstraksi: transformasi melalui memori semantik
            abstract = self._transform(attended_state)

            # 4. Penalaran: threshold logika (koherensi)
            logic_gate = [1.0 if x > 0.5 else 0.0 for x in abstract]

            # 5. Kreativitas: sintesis state asli dengan abstraksi
            imagination = self._synthesize(self.state, abstract)

            # 6. Integrasi: update state
            new_state = [
                0.3 * s + 0.4 * a + 0.3 * i
                for s, a, i in zip(self.state, logic_gate, imagination)
            ]
            self.state = new_state

            # 7. Update memori semantik (pembelajaran) dengan EWC
            new_memory = [row[:] for row in self.memory]
            for i in range(self.dim):
                for j in range(self.dim):
                    new_memory[i][j] += (self.state[i] * self.state[j]) * 0.005
            self._update_ewc(new_memory)

            # 8. Tulis state baru ke working memory
            self.working_memory.write(self.state)

            # 9. Update entropi (dinamika kreatif)
            self.entropy = (self.entropy * 1.01 + 0.001 * random.random()) % 1.0

            # 10. Hitung perplexity (surprise) untuk monitoring reflektif
            self._compute_perplexity()

    def _compute_perplexity(self):
        """Mengukur seberapa 'terkejut' sistem terhadap state saat ini."""
        # Sederhana: variansi state
        mean = sum(self.state) / self.dim
        var = sum((x - mean) ** 2 for x in self.state) / self.dim
        perplexity = math.exp(var)  # makin tinggi variansi, makin bingung
        self.perplexity_history.append(perplexity)
        self.last_surprise = perplexity

    def reflect(self) -> float:
        """Memantau kebingungan rata‑rata; jika tinggi, picu restrukturisasi."""
        if len(self.perplexity_history) < 5:
            return 0.0
        avg_perp = sum(self.perplexity_history) / len(self.perplexity_history)
        threshold = 1.5
        if avg_perp > threshold:
            self._restructure()
        return avg_perp

    def _restructure(self):
        """Restrukturisasi memori: tambahkan noise kreatif pada bobot penting."""
        for i in range(self.dim):
            for j in range(self.dim):
                if random.random() < 0.1:
                    self.memory[i][j] += random.gauss(0, 0.1)
        # Perbarui referensi optimal untuk EWC
        self.optimal_memory = [row[:] for row in self.memory]
        # Reset sebagian Fisher (opsional)
        for i in range(self.dim):
            for j in range(self.dim):
                self.fisher_diag[i][j] = max(0.5, self.fisher_diag[i][j] * 0.9)

    def query(self, stimulus: List[float]) -> List[float]:
        """Menerima stimulus eksternal, memproses, dan mengembalikan respons."""
        # Gabungkan stimulus dengan state internal
        combined = [0.6 * s + 0.4 * stim for s, stim in zip(self.state, stimulus)]
        self.state = combined
        self.think(cycles=5)
        return self.state