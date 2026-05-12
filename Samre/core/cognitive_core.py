# core/cognitive_core.py (v3.0 – NumPy-based, idempotent query)
import numpy as np
import math
import random
from collections import deque
from typing import List, Tuple
import copy

class CognitiveAttention:
    def __init__(self, dim: int, heads: int = 4):
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0
        self.head_dim = dim // heads
        # gunakan inisialisasi Glorot
        self.W_q = np.random.randn(dim, dim) * np.sqrt(2.0 / (dim + dim))
        self.W_k = np.random.randn(dim, dim) * np.sqrt(2.0 / (dim + dim))
        self.W_v = np.random.randn(dim, dim) * np.sqrt(2.0 / (dim + dim))
        self.W_o = np.random.randn(dim, dim) * np.sqrt(2.0 / (dim + dim))

    def attend(self, state: np.ndarray) -> np.ndarray:
        # state shape (dim,)
        q = np.dot(self.W_q, state)  # (dim,)
        k = np.dot(self.W_k, state)
        v = np.dot(self.W_v, state)
        # reshape ke (heads, head_dim)
        q = q.reshape(self.heads, self.head_dim)
        k = k.reshape(self.heads, self.head_dim)
        v = v.reshape(self.heads, self.head_dim)
        # hitung perhatian per head
        scores = np.sum(q * k, axis=1) / math.sqrt(self.head_dim)  # (heads,)
        attn = 1.0 / (1.0 + np.exp(-scores))  # sigmoid
        weighted = v * attn[:, np.newaxis]      # (heads, head_dim)
        combined = weighted.ravel()             # kembali ke (dim,)
        return np.dot(self.W_o, combined)

class WorkingMemory:
    def __init__(self, capacity: int, decay: float = 0.9):
        self.capacity = capacity
        self.decay = decay
        self.buffer = deque(maxlen=capacity)

    def write(self, item: np.ndarray):
        self.buffer.append(item.copy())

    def read(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros(64)  # default dim; akan diganti nanti
        weights = np.array([self.decay ** i for i in range(len(self.buffer))][::-1])
        stacked = np.array(self.buffer)
        combined = np.average(stacked, axis=0, weights=weights)
        return combined

class CognitiveEngine:
    def __init__(self, dimensionality: int = 64, attention_heads: int = 4):
        self.dim = dimensionality
        self.attention = CognitiveAttention(dimensionality, heads=attention_heads)
        self.working_memory = WorkingMemory(capacity=10, decay=0.85)
        # Memory matrix (dim x dim) diinisialisasi dengan Gaussian
        self.memory = np.random.randn(dimensionality, dimensionality) * 0.1
        self.state = np.random.randn(dimensionality) * 0.1
        self.entropy = 0.618
        self.fisher_diag = np.ones(dimensionality)
        self.optimal_memory = self.memory.copy()
        self.ewc_lambda = 0.4
        self.perplexity_history = deque(maxlen=20)
        self.last_surprise = 0.0

    def _activate(self, x: np.ndarray) -> np.ndarray:
        # fungsi aktivasi sigmoid
        return 1.0 / (1.0 + np.exp(-x / (1.0 + self.entropy)))

    def _transform(self, input_vector: np.ndarray) -> np.ndarray:
        return self._activate(np.dot(self.memory, input_vector))

    def _synthesize(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return (v1 + v2) * 0.5 * (1.0 + self.entropy * 0.1)

    def _update_ewc(self, new_memory: np.ndarray):
        delta = new_memory - self.memory
        penalty = self.ewc_lambda * self.fisher_diag * (self.memory - self.optimal_memory)
        self.memory += delta - penalty

    def think(self, state_to_process: np.ndarray, cycles: int = 10) -> np.ndarray:
        current = state_to_process.copy()
        for _ in range(cycles):
            wm = self.working_memory.read()
            if wm.shape[0] != self.dim:
                wm = np.zeros(self.dim)
            blended = 0.7 * current + 0.3 * wm
            attended = self.attention.attend(blended)
            abstract = self._transform(attended)
            logic_gate = (abstract > 0.5).astype(float)
            imagination = self._synthesize(current, abstract)
            current = 0.3 * current + 0.4 * logic_gate + 0.3 * imagination
        return current

    def internal_think(self, cycles: int = 10):
        new_state = self.think(self.state, cycles)
        # Hebbian update
        hebb = np.outer(new_state, self.state) * 0.005
        new_memory = self.memory + hebb
        self._update_ewc(new_memory)
        self.state = new_state
        self.working_memory.write(self.state)
        self.entropy = (self.entropy * 1.01 + 0.001 * random.random()) % 1.0
        self._compute_perplexity()

    def _compute_perplexity(self):
        var = np.var(self.state)
        self.perplexity_history.append(math.exp(var))

    def reflect(self):
        if len(self.perplexity_history) < 5:
            return
        if np.mean(self.perplexity_history) > 1.5:
            self._restructure()

    def _restructure(self):
        # tambah noise kecil
        self.memory += np.random.randn(self.dim, self.dim) * 0.1
        self.optimal_memory = self.memory.copy()
        self.fisher_diag = np.maximum(0.5, self.fisher_diag * 0.9)

    def query(self, stimulus: List[float]) -> List[float]:
        """Idempoten query – tidak mengubah state internal."""
        transient_state = copy.deepcopy(self.state)
        stim = np.array(stimulus, dtype=np.float64)
        combined = 0.6 * transient_state + 0.4 * stim
        response = self.think(combined, cycles=5)
        return response.tolist()