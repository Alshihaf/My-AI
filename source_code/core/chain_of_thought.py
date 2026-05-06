# core/chain_of_thought.py (versi diperbaiki)
import numpy as np
from typing import List, Optional

class ChainOfThought:
    def __init__(self, cognitive_engine, neuromodulator_system=None):
        self.engine = cognitive_engine
        self.neuromod = neuromodulator_system
        self.trace = []
    
    def reason(self, problem_text: str, steps: int = 3) -> List[float]:
        # Konversi problem ke vektor dengan embedding yang lebih baik
        vec = self._text_to_vector(problem_text, self.engine.dim)
        
        # Set state awal
        self.engine.state = vec
        self.trace = []
        
        # Modulasi jumlah langkah berdasarkan neuromodulator
        if self.neuromod:
            # Acetylcholine mempengaruhi kedalaman penalaran
            ach = self.neuromod.acetylcholine.level
            effective_steps = int(steps * (1.0 + ach))
        else:
            effective_steps = steps
        
        for step in range(effective_steps):
            # Modulasi jumlah siklus think per langkah
            cycles = 5
            if self.neuromod:
                # Noradrenaline meningkatkan fokus -> lebih banyak siklus
                cycles = int(5 * (1.0 + self.neuromod.noradrenaline.level))
            
            self.engine.think(cycles=cycles)
            thought_vec = self.engine.state.copy()
            self.trace.append(thought_vec)
            
            # Catat perplexity untuk refleksi
            _ = self.engine.reflect()
        
        # Kembalikan state akhir sebagai ringkasan
        return self.engine.state
    
    def _text_to_vector(self, text: str, dim: int) -> List[float]:
        """Embedding teks sederhana menggunakan bag-of-characters + hash."""
        vec = [0.0] * dim
        if not text:
            return vec
        
        # Normalisasi panjang
        text = text[:dim*10]  # batasi
        
        # Gunakan sliding window n-gram karakter
        for i, ch in enumerate(text):
            idx = ord(ch) % dim
            vec[idx] += 1.0
        
        # Normalisasi
        total = sum(vec)
        if total > 0:
            vec = [v/total for v in vec]
        
        # Tambahkan noise kecil untuk variasi
        vec = [v + np.random.normal(0, 0.01) for v in vec]
        return vec