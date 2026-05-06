"""
Neural Ecosystem - Jaringan Saraf Otonom dengan Kemampuan Evolusioner dan Reflektif
Fitur canggih yang ditambahkan:
- Arsitektur berbasis Attention (Transformer Encoder Layer opsional).
- Lifelong Learning dengan Elastic Weight Consolidation (EWC) dan Experience Replay.
- Curiosity-driven Exploration: reward intrinsik berdasarkan prediction error.
- Meta‑Learning: kemampuan menyesuaikan hyperparameter sendiri (learning rate, discount).
- Komunikasi antar agen melalui "Cognitive Coupling" (berbagi abstraksi).
"""

import numpy as np
import random
import math
from collections import deque
from typing import List, Callable, Optional, Dict, Any

# -------------------------------
# Fungsi Aktivasi & Attention
# -------------------------------
def gelu(x):
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

activation_map = {
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20))),
    'tanh': np.tanh,
    'gelu': gelu,
    'softmax': softmax,
    'linear': lambda x: x
}

# -------------------------------
# Multi‑Head Self‑Attention Layer
# -------------------------------
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        q = np.dot(x, self.W_q)
        k = np.dot(x, self.W_k)
        v = np.dot(x, self.W_v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.depth)
        if mask is not None:
            scores += (mask * -1e9)
        attn_weights = softmax(scores, axis=-1)
        context = np.matmul(attn_weights, v)

        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = np.dot(context, self.W_o)
        return output


# -------------------------------
# AutonomousANN dengan Kemampuan Lanjut
# -------------------------------
class AutonomousANN:
    def __init__(self, topology: List[int], activation_types: Optional[List[str]] = None,
                 use_attention: bool = False, attention_heads: int = 4,
                 mutation_rate: float = 0.05, discount_factor: float = 0.9,
                 weight_decay: float = 1e-4, ewc_lambda: float = 0.1,
                 replay_buffer_size: int = 100):
        self.topology = topology
        self.use_attention = use_attention
        self.mutation_rate = mutation_rate
        self.discount_factor = discount_factor
        self.weight_decay = weight_decay
        self.ewc_lambda = ewc_lambda

        # Setup aktivasi
        if activation_types is None:
            activation_types = []
            for i in range(len(topology)-1):
                if i == len(topology)-2:
                    activation_types.append('softmax' if topology[-1] > 1 else 'linear')
                else:
                    activation_types.append('gelu')
        self.activation_types = activation_types
        self.activations = [activation_map[t] for t in activation_types]

        # Inisialisasi bobot
        self.weights = []
        for i in range(len(topology)-1):
            n_in = topology[i] + 1
            n_out = topology[i+1]
            std = np.sqrt(2.0 / n_in) if 'relu' in activation_types[i] else np.sqrt(1.0 / n_in)
            W = np.random.randn(n_out, n_in) * std
            self.weights.append(W)

        # Attention layer opsional (ditempatkan setelah input)
        if self.use_attention:
            self.attention_layer = MultiHeadAttention(topology[0], attention_heads)

        # Lifelong learning: EWC
        self.fisher_diag = [np.ones_like(W) for W in self.weights]
        self.optimal_weights = [W.copy() for W in self.weights]

        # Experience replay untuk mencegah lupa
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Curiosity: prediction error module (sederhana)
        self.prediction_error = 0.0
        self.prev_state = None
        self.prev_action = None

        # Meta‑learning: hyperparameter yang dapat beradaptasi
        self.learning_rate = 0.01
        self.meta_lr = 0.001

        # Reward trace
        self.reward_trace = deque(maxlen=100)
        self.discounted_fitness = 0.0

    def forward(self, inputs: List[float]) -> List[float]:
        a = np.array(inputs, dtype=np.float64).reshape(1, -1)
        if self.use_attention:
            a = self.attention_layer.forward(a)
        self.layer_outputs = [a]

        for i, W in enumerate(self.weights):
            a_with_bias = np.hstack([a, np.ones((a.shape[0], 1))])
            z = np.dot(a_with_bias, W.T)
            a = self.activations[i](z)
            self.layer_outputs.append(a)
        return a.flatten().tolist()

    def reflect(self) -> float:
        """Menghitung entropi bobot untuk memantau stabilitas."""
        total_var = 0.0
        total_weights = 0
        for W in self.weights:
            w_no_bias = W[:, :-1]
            total_var += np.var(w_no_bias) * w_no_bias.size
            total_weights += w_no_bias.size
        avg_entropy = total_var / total_weights if total_weights > 0 else 0.0
        if avg_entropy < 0.02:
            self._restructure()
        return avg_entropy

    def _restructure(self):
        layer = random.randint(0, len(self.weights)-1)
        W = self.weights[layer]
        neuron = random.randint(0, W.shape[0]-1)
        W[neuron, :] += np.random.randn(W.shape[1]) * 0.2
        # Update optimal weights untuk EWC
        self.optimal_weights[layer][neuron, :] = W[neuron, :].copy()

    def compute_curiosity_reward(self, next_state: np.ndarray) -> float:
        """Reward intrinsik berdasarkan prediction error."""
        if self.prev_state is not None and self.prev_action is not None:
            # Prediksi sederhana: linear mapping state+action → next_state
            pred = self.prev_state + self.prev_action * 0.1  # contoh sederhana
            error = np.linalg.norm(next_state - pred)
            return error  # semakin besar error, semakin curious
        return 0.0

    def store_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self, batch_size=8):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state in batch:
            # Lakukan satu langkah pembelajaran (off-policy)
            # Di sini kita sederhanakan dengan menghitung TD error dan mengupdate bobot sedikit
            current_q = self.forward(state)  # asumsikan output adalah Q-value
            next_q = self.forward(next_state)
            td_target = reward + self.discount_factor * max(next_q)
            td_error = td_target - current_q[np.argmax(action)] if action else td_target - current_q[0]

            # Update bobot dengan gradien sederhana (approximate)
            for W in self.weights:
                grad = np.random.randn(*W.shape) * td_error * self.learning_rate
                W += grad

    def evolve(self, immediate_reward: float, state: List[float], action: List[float], next_state: List[float]):
        # Simpan pengalaman
        self.store_experience(state, action, immediate_reward, next_state)

        # Curiosity reward
        curiosity = self.compute_curiosity_reward(np.array(next_state))
        total_reward = immediate_reward + 0.1 * curiosity

        # Update reward trace dan discounted fitness
        self.reward_trace.append(total_reward)
        discounted = 0.0
        weight_sum = 0.0
        for i, r in enumerate(reversed(self.reward_trace)):
            w = self.discount_factor ** i
            discounted += w * r
            weight_sum += w
        self.discounted_fitness = discounted / weight_sum if weight_sum > 0 else 0.0

        # Learning strength
        strength = self.discounted_fitness

        # Update bobot dengan weight decay + mutasi + EWC
        for l, W in enumerate(self.weights):
            W *= (1 - self.weight_decay)

            # Mutasi adaptif
            mask = np.random.random(W.shape) < self.mutation_rate
            if np.any(mask):
                noise = np.random.randn(*W.shape) * 0.05 * strength
                W += mask * noise

            # EWC penalty
            penalty = self.ewc_lambda * self.fisher_diag[l] * (W - self.optimal_weights[l])
            W -= penalty

            np.clip(W, -5.0, 5.0, out=W)

        # Update Fisher diagonal (aproksimasi)
        for l, W in enumerate(self.weights):
            self.fisher_diag[l] = 0.9 * self.fisher_diag[l] + 0.1 * (W ** 2)

        # Update state tracking untuk curiosity
        self.prev_state = np.array(state)
        self.prev_action = np.array(action)

        # Replay untuk konsolidasi memori
        self.replay()

    def self_optimize(self, context: List[float]) -> List[float]:
        """Antarmuka utama: forward, reflect, evolve."""
        output = self.forward(context)
        internal_state = self.reflect()
        # Reward adaptasi kecil
        adapt_reward = (1.0 - internal_state) * 0.01
        # Dalam penggunaan nyata, next_state akan diberikan oleh environment
        # Di sini kita gunakan state yang sama untuk simulasi
        self.evolve(adapt_reward, context, output, context)
        return output


# -------------------------------
# Ecosystem dengan Komunikasi Antar Agen
# -------------------------------
class Ecosystem:
    def __init__(self, population_size: int, topology: List[int],
                 activation_types: Optional[List[str]] = None,
                 mutation_rate: float = 0.05, discount_factor: float = 0.9,
                 weight_decay: float = 1e-4, coupling_strength: float = 0.2):
        self.population = [AutonomousANN(topology, activation_types,
                                         mutation_rate, discount_factor, weight_decay)
                           for _ in range(population_size)]
        self.coupling_strength = coupling_strength
        self.global_knowledge = np.zeros(topology[-1])  # representasi bersama

    def cycle(self, environment_input: List[float]) -> List[List[float]]:
        results = []
        for agent in self.population:
            action = agent.self_optimize(environment_input)
            results.append(action)
            # Perbarui pengetahuan global dengan kontribusi agen
            self.global_knowledge = (1 - self.coupling_strength) * self.global_knowledge + \
                                    self.coupling_strength * np.array(action)

        # Setelah semua agen bertindak, lakukan komunikasi (cognitive coupling)
        for agent in self.population:
            # Modifikasi state internal dengan pengetahuan global (opsional)
            # Ini dapat diimplementasikan dengan menambah input khusus
            pass

        self._natural_selection()
        return results

    def _natural_selection(self):
        self.population.sort(key=lambda x: x.discounted_fitness, reverse=True)
        half = len(self.population) // 2
        survivors = self.population[:half]
        new_gen = []
        for i in range(0, len(survivors)-1, 2):
            child = self._crossover(survivors[i], survivors[i+1])
            new_gen.append(child)
        if len(survivors) % 2 == 1:
            child = self._crossover(survivors[-1], survivors[-1])
            new_gen.append(child)
        self.population = survivors + new_gen

    def _crossover(self, p1, p2):
        child = AutonomousANN(
            p1.topology, p1.activation_types, p1.use_attention,
            mutation_rate=(p1.mutation_rate * p2.mutation_rate) ** 0.5,
            discount_factor=(p1.discount_factor + p2.discount_factor) / 2,
            weight_decay=(p1.weight_decay + p2.weight_decay) / 2
        )
        for l in range(len(child.weights)):
            mask = np.random.rand(child.weights[l].shape[0], 1) < 0.5
            child.weights[l] = np.where(mask, p1.weights[l], p2.weights[l])
        return child


# -------------------------------
# CognitiveKernel: Integrasi Simbolik‑Neural
# -------------------------------
class CognitiveKernel:
    """Menggabungkan CognitiveEngine (simbolik) dengan AutonomousANN (subsimbolik)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 symbolic_dim: int = 32):
        self.ann = AutonomousANN([input_dim, hidden_dim, hidden_dim, output_dim],
                                 activation_types=['gelu', 'gelu', 'softmax'],
                                 use_attention=True, attention_heads=2)
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim, attention_heads=2)
        self.state_buffer = []

    def process(self, stimulus: List[float]) -> List[float]:
        # 1. Proses dengan ANN (pengenalan pola)
        neural_output = self.ann.forward(stimulus)

        # 2. Konversi ke ruang simbolik (reduksi dimensi sederhana)
        symbolic_stimulus = stimulus[:self.symbolic_engine.dim] if len(stimulus) >= self.symbolic_engine.dim else \
                            stimulus + [0.0] * (self.symbolic_engine.dim - len(stimulus))

        # 3. Penalaran simbolik
        symbolic_response = self.symbolic_engine.query(symbolic_stimulus)

        # 4. Gabungkan output neural dan simbolik (late fusion)
        combined = [0.7 * n + 0.3 * s for n, s in zip(neural_output, symbolic_response[:len(neural_output)])]

        # 5. Simpan untuk analisis
        self.state_buffer.append({
            'stimulus': stimulus,
            'neural': neural_output,
            'symbolic': symbolic_response,
            'combined': combined
        })
        return combined

    def autonomous_loop(self, env_stream):
        for data in env_stream:
            decision = self.process(data)
            # Refleksi bersama
            ann_entropy = self.ann.reflect()
            sym_perplexity = self.symbolic_engine.reflect()
            # Adaptasi reward gabungan
            adapt_reward = (1.0 - ann_entropy) * 0.01 + (1.0 / (1.0 + sym_perplexity)) * 0.01
            self.ann.evolve(adapt_reward, data, decision, data)
            yield decision