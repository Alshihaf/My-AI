import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# =============================================================================
#  NEUROMODULATOR SYSTEM (No output, pure internal state & modulation logic)
# =============================================================================

class Neuromodulator(ABC):
    """Abstract base class for a single neuromodulator."""
    
    def __init__(self, name: str, initial_level: float = 0.0, 
                 decay_rate: float = 0.01, baseline: float = 0.0):
        self.name = name
        self.level = np.clip(initial_level, 0.0, 1.0)
        self.decay_rate = decay_rate
        self.baseline = baseline
        self._history: List[float] = []
        
    def update(self, delta: float = 0.0, external_input: float = 0.0) -> float:
        """Update level based on delta (e.g., from reward) and decay toward baseline."""
        self.level += delta + external_input
        self.level = np.clip(self.level, 0.0, 1.0)
        # Decay toward baseline
        self.level = self.level * (1 - self.decay_rate) + self.baseline * self.decay_rate
        self._history.append(self.level)
        return self.level
    
    @abstractmethod
    def modulate(self, parameter_name: str, base_value: float) -> float:
        """Return modulated value of a given parameter."""
        pass
    
    def reset(self):
        self.level = self.baseline
        self._history.clear()


class Dopamine(Neuromodulator):
    """Dopamine: modulates learning rate, exploration, and motivation."""
    
    def __init__(self, initial_level: float = 0.1, decay_rate: float = 0.05):
        super().__init__("Dopamine", initial_level, decay_rate, baseline=0.1)
    
    def modulate(self, parameter_name: str, base_value: float) -> float:
        if parameter_name == "learning_rate":
            # Higher dopamine -> higher learning rate (up to 2x)
            return base_value * (1.0 + self.level)
        elif parameter_name == "exploration_noise":
            # Higher dopamine -> more exploration
            return base_value * (1.0 + self.level * 1.5)
        elif parameter_name == "action_bias":
            # Higher dopamine -> positive bias (optimism)
            return base_value + self.level * 0.5
        else:
            return base_value


class Serotonin(Neuromodulator):
    """Serotonin: modulates patience, risk aversion, and mood stability."""
    
    def __init__(self, initial_level: float = 0.3, decay_rate: float = 0.02):
        super().__init__("Serotonin", initial_level, decay_rate, baseline=0.3)
    
    def modulate(self, parameter_name: str, base_value: float) -> float:
        if parameter_name == "discount_factor":
            # Higher serotonin -> more patience (higher discount factor)
            return min(base_value + self.level * 0.2, 0.99)
        elif parameter_name == "risk_aversion":
            # Higher serotonin -> less risk-taking
            return base_value * (1.0 + self.level)
        elif parameter_name == "negative_feedback_sensitivity":
            # Higher serotonin -> lower sensitivity to negative outcomes
            return base_value * (1.0 - self.level * 0.5)
        else:
            return base_value


class Noradrenaline(Neuromodulator):
    """Noradrenaline (Norepinephrine): modulates arousal, attention, and signal-to-noise ratio."""
    
    def __init__(self, initial_level: float = 0.2, decay_rate: float = 0.03):
        super().__init__("Noradrenaline", initial_level, decay_rate, baseline=0.2)
    
    def modulate(self, parameter_name: str, base_value: float) -> float:
        if parameter_name == "gain":
            # Higher NE -> increased neuronal gain
            return base_value * (1.0 + self.level)
        elif parameter_name == "noise_level":
            # Inverted-U: intermediate NE reduces noise; extremes increase noise
            noise_mod = 1.0 - 2.0 * np.abs(self.level - 0.5)
            return base_value * (1.0 + noise_mod * 0.5)
        elif parameter_name == "attention_focus":
            # Higher NE -> narrower attentional focus (increase contrast)
            return base_value * (1.0 + self.level)
        else:
            return base_value


class Acetylcholine(Neuromodulator):
    """Acetylcholine: modulates memory encoding, sensory precision, and learning stability."""
    
    def __init__(self, initial_level: float = 0.15, decay_rate: float = 0.04):
        super().__init__("Acetylcholine", initial_level, decay_rate, baseline=0.15)
    
    def modulate(self, parameter_name: str, base_value: float) -> float:
        if parameter_name == "memory_retention":
            # Higher ACh -> stronger memory retention
            return base_value * (1.0 + self.level)
        elif parameter_name == "sensory_precision":
            # Higher ACh -> more precise sensory representation
            return base_value * (1.0 + self.level * 1.2)
        elif parameter_name == "learning_stability":
            # Higher ACh -> reduced interference (more stable)
            return base_value * (1.0 + self.level * 0.8)
        else:
            return base_value


@dataclass
class NeuromodulatorSystem:
    """Orchestrates multiple neuromodulators and their interactions."""
    
    dopamine: Dopamine = field(default_factory=Dopamine)
    serotonin: Serotonin = field(default_factory=Serotonin)
    noradrenaline: Noradrenaline = field(default_factory=Noradrenaline)
    acetylcholine: Acetylcholine = field(default_factory=Acetylcholine)
    
    def update_all(self, deltas: Optional[Dict[str, float]] = None, 
                   external_inputs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Update all modulators with given deltas and external inputs."""
        deltas = deltas or {}
        external_inputs = external_inputs or {}
        levels = {}
        for mod in [self.dopamine, self.serotonin, self.noradrenaline, self.acetylcholine]:
            delta = deltas.get(mod.name, 0.0)
            ext_in = external_inputs.get(mod.name, 0.0)
            levels[mod.name] = mod.update(delta, ext_in)
        return levels
    
    def get_modulated_value(self, modulator_name: str, param_name: str, base_value: float) -> float:
        """Apply modulation from a specific modulator."""
        mod = getattr(self, modulator_name.lower())
        return mod.modulate(param_name, base_value)
    
    def get_all_levels(self) -> Dict[str, float]:
        """Return current levels of all modulators."""
        return {
            "Dopamine": self.dopamine.level,
            "Serotonin": self.serotonin.level,
            "Noradrenaline": self.noradrenaline.level,
            "Acetylcholine": self.acetylcholine.level
        }
    
    def apply_to_learning_params(self, base_lr: float, base_discount: float,
                                 base_noise: float) -> Tuple[float, float, float]:
        """Convenience: return modulated learning rate, discount factor, exploration noise."""
        lr = self.dopamine.modulate("learning_rate", base_lr)
        discount = self.serotonin.modulate("discount_factor", base_discount)
        noise = self.dopamine.modulate("exploration_noise", base_noise)
        noise = self.noradrenaline.modulate("noise_level", noise)
        return lr, discount, noise
    
    def reset_all(self):
        for mod in [self.dopamine, self.serotonin, self.noradrenaline, self.acetylcholine]:
            mod.reset()


# =============================================================================
#  OPTIONAL: Context-dependent neuromodulatory event handler (no output)
# =============================================================================

class NeuromodulatoryEvent:
    """Define events that trigger neuromodulator release."""
    
    @staticmethod
    def reward_prediction_error(actual: float, expected: float) -> Dict[str, float]:
        """Compute dopamine delta based on reward prediction error."""
        rpe = actual - expected
        return {"Dopamine": np.clip(rpe, -1.0, 1.0) * 0.5}
    
    @staticmethod
    def unexpected_punishment(intensity: float = 0.5) -> Dict[str, float]:
        """Negative event triggers serotonin drop and NE spike."""
        return {
            "Serotonin": -intensity * 0.3,
            "Noradrenaline": intensity * 0.4
        }
    
    @staticmethod
    def novelty_detection(novelty_score: float) -> Dict[str, float]:
        """Novelty increases acetylcholine and noradrenaline."""
        return {
            "Acetylcholine": novelty_score * 0.3,
            "Noradrenaline": novelty_score * 0.2
        }
    
    @staticmethod
    def stress_response(level: float) -> Dict[str, float]:
        """Stress elevates noradrenaline and reduces serotonin."""
        return {
            "Noradrenaline": level * 0.5,
            "Serotonin": -level * 0.2
        }