# core/reflection.py
import numpy as np
from typing import List, Optional

class Reflection:
    def __init__(self, reward_system):
        self.reward = reward_system  # neuromodulator system
        self.baseline_perplexity = 1.0
        self.adaptation_rate = 0.1
    
    def evaluate(self, thought_trace: List[List[float]], ground_truth: Optional[List[float]] = None) -> float:
        """
        Mengevaluasi kualitas penalaran berdasarkan koherensi internal.
        Mengembalikan skor antara 0 dan 1.
        """
        if len(thought_trace) < 2:
            return 0.5
        
        # 1. Koherensi: seberapa konsisten perubahan antar langkah
        coherence_scores = []
        for i in range(len(thought_trace)-1):
            v1 = np.array(thought_trace[i])
            v2 = np.array(thought_trace[i+1])
            # Cosine similarity
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                sim = dot / (norm1 * norm2)
            else:
                sim = 0.0
            coherence_scores.append(sim)
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        # 2. Surprise (perplexity-like): variansi state akhir
        final_state = np.array(thought_trace[-1])
        variance = np.var(final_state)
        surprise = np.exp(-variance)  # variansi rendah -> surprise rendah
        
        # 3. Progress: apakah state bergerak menjauh dari awal (eksplorasi)
        initial = np.array(thought_trace[0])
        final = np.array(thought_trace[-1])
        progress = np.linalg.norm(final - initial) / np.sqrt(len(initial))
        progress = min(progress, 1.0)  # normalisasi
        
        # Skor gabungan
        score = 0.4 * avg_coherence + 0.3 * surprise + 0.3 * progress
        
        # Update neuromodulator berdasarkan skor
        delta_dopamine = (score - 0.5) * 0.2
        delta_serotonin = (avg_coherence - 0.5) * 0.1
        delta_ach = (progress - 0.3) * 0.1
        delta_ne = (surprise - 0.5) * 0.1
        
        self.reward.update_all({
            "Dopamine": delta_dopamine,
            "Serotonin": delta_serotonin,
            "Acetylcholine": delta_ach,
            "Noradrenaline": delta_ne
        })
        
        # Update baseline
        self.baseline_perplexity = 0.9 * self.baseline_perplexity + 0.1 * (1 - surprise)
        
        return score