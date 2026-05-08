"""
Cognitive Core - Mesin Penalaran Vektor Simbolik Tingkat Tinggi (v2.1)

This version introduces idempotency to the query() method by operating on a
copy of the state, ensuring that querying does not alter the engine's internal state.
"""

import math
import random
from collections import deque
from typing import List, Tuple, Optional
import copy # INTEGRATION: To allow for idempotent queries

# ... (CognitiveAttention and WorkingMemory classes remain the same) ...
class CognitiveAttention:
    def __init__(self, dim: int, heads: int = 4):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0
        self.W_q = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
        self.W_k = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
        self.W_v = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
        self.W_o = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
    def _matmul(self, A, B):
        if isinstance(B[0], (int, float)):
            return [sum(a * b for a, b in zip(row, B)) for row in A]
        else:
            return [[sum(a * b for a, b in zip(row_A, col_B)) for col_B in zip(*B)] for row_A in A]
    def attend(self, state: List[float]) -> List[float]:
        q = self._matmul(self.W_q, state)
        k = self._matmul(self.W_k, state)
        v = self._matmul(self.W_v, state)
        head_outputs = []
        for h in range(self.heads):
            start, end = h * self.head_dim, (h + 1) * self.head_dim
            q_h, k_h, v_h = q[start:end], k[start:end], v[start:end]
            score = sum(qi * ki for qi, ki in zip(q_h, k_h)) / math.sqrt(self.head_dim)
            attention_weight = 1.0 / (1.0 + math.exp(-score))
            head_outputs.extend([attention_weight * vi for vi in v_h])
        return self._matmul(self.W_o, head_outputs)

class WorkingMemory:
    def __init__(self, capacity: int, decay: float = 0.9):
        self.capacity = capacity
        self.decay = decay
        self.buffer = deque(maxlen=capacity)
    def write(self, item: List[float]): self.buffer.append(item)
    def read(self) -> List[float]:
        if not self.buffer: return []
        combined = [0.0] * len(self.buffer[0])
        weight_sum = 0.0
        for i, mem in enumerate(reversed(self.buffer)):
            w = self.decay ** i
            weight_sum += w
            for j, val in enumerate(mem): combined[j] += w * val
        return [c / weight_sum for c in combined] if weight_sum > 0 else combined

class CognitiveEngine:
    """
    The core cognitive engine, now with idempotent queries.
    """
    def __init__(self, dimensionality: int = 64, attention_heads: int = 4):
        self.dim = dimensionality
        self.attention = CognitiveAttention(self.dim, heads=attention_heads)
        self.working_memory = WorkingMemory(capacity=10, decay=0.85)
        self.memory = [[(i * j * 0.1) % 1.0 for j in range(self.dim)] for i in range(self.dim)]
        self.state = [((i * 0.5) % 1.0) for i in range(self.dim)]
        self.entropy = 0.618
        self.fisher_diag = [[1.0 for _ in range(self.dim)] for _ in range(self.dim)]
        self.optimal_memory = [row[:] for row in self.memory]
        self.ewc_lambda = 0.4
        self.perplexity_history = deque(maxlen=20)
        self.last_surprise = 0.0

    def _dot(self, a, b): return sum(ai * bi for ai, bi in zip(a, b))
    def _activate(self, x: float) -> float: return 1.0 / (1.0 + math.exp(-x / (1.0 + self.entropy)))

    def _transform(self, input_vector: List[float]) -> List[float]:
        return [self._activate(self._dot(row, input_vector)) for row in self.memory]

    def _synthesize(self, v1: List[float], v2: List[float]) -> List[float]:
        return [((a + b) / 2.0) * (1.0 + self.entropy * 0.1) for a, b in zip(v1, v2)]

    def _update_ewc(self, new_memory: List[List[float]]):
        for i in range(self.dim):
            for j in range(self.dim):
                delta = new_memory[i][j] - self.memory[i][j]
                penalty = self.ewc_lambda * self.fisher_diag[i][j] * (self.memory[i][j] - self.optimal_memory[i][j])
                self.memory[i][j] += delta - penalty

    def think(self, state_to_process: List[float], cycles: int = 10) -> List[float]:
        """Performs a thinking cycle on a given state without modifying the instance's main state."""
        current_state = state_to_process
        for _ in range(cycles):
            wm_context = self.working_memory.read()
            blended = [0.7 * s + 0.3 * w for s, w in zip(current_state, wm_context)] if wm_context else current_state
            attended_state = self.attention.attend(blended)
            abstract = self._transform(attended_state)
            logic_gate = [1.0 if x > 0.5 else 0.0 for x in abstract]
            imagination = self._synthesize(current_state, abstract)
            current_state = [0.3 * s + 0.4 * a + 0.3 * i for s, a, i in zip(current_state, logic_gate, imagination)]
        return current_state

    def internal_think(self, cycles: int = 10):
        """The main thinking loop that DOES modify the internal state."""
        new_state = self.think(self.state, cycles)
        # Hebbian learning on the actual memory based on state change
        new_memory = [row[:] for row in self.memory]
        for i in range(self.dim):
            for j in range(self.dim):
                new_memory[i][j] += (new_state[i] * self.state[j]) * 0.005 # Associating new with old
        self._update_ewc(new_memory)
        self.state = new_state
        self.working_memory.write(self.state)
        self.entropy = (self.entropy * 1.01 + 0.001 * random.random()) % 1.0
        self._compute_perplexity()

    def _compute_perplexity(self):
        mean = sum(self.state) / self.dim
        var = sum((x - mean) ** 2 for x in self.state) / self.dim
        self.perplexity_history.append(math.exp(var))

    def reflect(self):
        if len(self.perplexity_history) < 5: return
        if (sum(self.perplexity_history) / len(self.perplexity_history)) > 1.5:
            self._restructure()

    def _restructure(self):
        for i in range(self.dim): 
            for j in range(self.dim): 
                if random.random() < 0.1: self.memory[i][j] += random.gauss(0,0.1)
        self.optimal_memory = [row[:] for row in self.memory]
        for i in range(self.dim): 
            for j in range(self.dim): self.fisher_diag[i][j] = max(0.5, self.fisher_diag[i][j] * 0.9)

    def query(self, stimulus: List[float]) -> List[float]:
        """Receives external stimulus, processes it without changing internal state, and returns a response."""
        # INTEGRATION: Use a deep copy to ensure idempotency
        transient_state = copy.deepcopy(self.state)
        # Combine stimulus with the copied state
        combined_state = [0.6 * s + 0.4 * stim for s, stim in zip(transient_state, stimulus)]
        # Process this temporary state
        return self.think(combined_state, cycles=5)
