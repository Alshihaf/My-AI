"""
SWS Logic - Slow Wave Sleep / System Wide Simulation (v2.2)

Action scoring via a learned linear value function.
Replaces hardcoded rules with online reinforcement learning,
while retaining initial domain knowledge for warm-start.

Author: Samre Cognitive Core Team
"""

import random
from typing import Dict, List, Optional

import numpy as np

# ──────────────────────────────────────────────
# All possible actions the agent can take
# ──────────────────────────────────────────────
POSSIBLE_ACTIONS = [
    "EXPLORE",
    "EVOLVE",
    "ORGANIZE",
    "REST",
    "LEARN",
    "REASON",
    "CONSOLIDATE",
    "GEOLOGY_EXPLORE",
    "GEOLOGY_LEARN",
    "IMAGINE",
    "CONTEMPLATE",
]

# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
# Feature indices (for documentation; actual vectors are built by order)
# 0..4 : needs (hunger, boredom, fatigue, messiness, cognitive_load)
# 5..8 : neuromodulators (dopamine, serotonin, noradrenaline, acetylcholine)
# 9    : context_geology   (0/1)
# 10   : context_code      (0/1)
# 11   : ltm_success_rate  (float 0..1)
# 12   : bias              (always 1.0)
_FEATURE_DIM = 13

def _extract_features(
    action: str,
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    recalled_concepts_context: List[str],
    ltm_success_rate: float,
) -> np.ndarray:
    """Convert current cognitive context into a fixed-length feature vector."""
    # Needs (order consistent with InternalNeeds)
    hunger = needs.get("hunger", 0.0)
    boredom = needs.get("boredom", 0.0)
    fatigue = needs.get("fatigue", 0.0)
    messiness = needs.get("messiness", 0.0)
    cognitive_load = needs.get("cognitive_load", 0.0)

    # Neuromodulators – accept both capitalised and lower‑case keys for robustness
    # (FlockOfThought currently sends lower‑case keys; this makes it idempotent)
    dopamine = neuromodulators.get("dopamine", neuromodulators.get("Dopamine", 0.5))
    serotonin = neuromodulators.get("serotonin", neuromodulators.get("Serotonin", 0.5))
    noradrenaline = neuromodulators.get(
        "noradrenaline", neuromodulators.get("Noradrenaline", 0.1)
    )
    acetylcholine = neuromodulators.get(
        "acetylcholine", neuromodulators.get("Acetylcholine", 0.1)
    )

    # Short‑term memory context as binary flags
    context_str = " ".join(recalled_concepts_context).lower()
    context_geology = 1.0 if any(
        w in context_str for w in ("geology", "macrostrat", "rock")
    ) else 0.0
    context_code = 1.0 if any(
        w in context_str for w in ("file", "source", "code")
    ) else 0.0

    # LTM success rate (per action)
    success = ltm_success_rate

    # Build vector (the action is used for weight lookup, not directly in features)
    feat = np.array(
        [
            hunger,
            boredom,
            fatigue,
            messiness,
            cognitive_load,
            dopamine,
            serotonin,
            noradrenaline,
            acetylcholine,
            context_geology,
            context_code,
            success,
            1.0,  # bias
        ],
        dtype=np.float64,
    )
    return feat


# ──────────────────────────────────────────────
# Linear value function (one weight vector per action)
# ──────────────────────────────────────────────

class LinearActionValue:
    """
    Maintains a learned linear value function for each action.
    Supports online gradient‑descent updates and weight initialisation
    from a prior rule‑based policy (so the agent behaves sensibly from the start).
    """

    def __init__(
        self,
        feature_dim: int = _FEATURE_DIM,
        learning_rate: float = 0.1,
        l2_reg: float = 0.001,
    ):
        self.feature_dim = feature_dim
        self.lr = learning_rate
        self.l2_reg = l2_reg

        # One weight vector per possible action
        self.weights: Dict[str, np.ndarray] = {
            a: np.zeros(feature_dim, dtype=np.float64) for a in POSSIBLE_ACTIONS
        }

        # Initialise with prior domain knowledge (warm‑start)
        self._init_priors()

    # ── Prior initialisation ───────────────────────

    def _init_priors(self) -> None:
        """
        Set initial weights to mimic the original hard‑coded scoring rules.
        This gives the agent a head start and keeps it from random walking.
        """
        # Helper to set specific feature weights
        def set_w(action: str, **deltas: float):
            w = self.weights[action]
            for feat_name, val in deltas.items():
                idx = _FEATURE_NAMES.index(feat_name)
                w[idx] += val

        _FEATURE_NAMES = [
            "hunger",
            "boredom",
            "fatigue",
            "messiness",
            "cognitive_load",
            "dopamine",
            "serotonin",
            "noradrenaline",
            "acetylcholine",
            "context_geology",
            "context_code",
            "ltm_success",
            "bias",
        ]

        # CONTEMPLATE
        set_w("CONTEMPLATE", cognitive_load=1.05, boredom=0.7, fatigue=-0.3, bias=0.0)
        # EXPLORE / GEOLOGY_EXPLORE
        for a in ("EXPLORE", "GEOLOGY_EXPLORE"):
            set_w(a, hunger=1.5, boredom=1.0, bias=0.0)
        # LEARN / GEOLOGY_LEARN
        for a in ("LEARN", "GEOLOGY_LEARN"):
            set_w(a, hunger=1.2, bias=0.0)
        # CONSOLIDATE
        set_w("CONSOLIDATE", cognitive_load=0.6, bias=0.2)
        # REST
        set_w("REST", fatigue=2.0, bias=0.0)
        # IMAGINE
        set_w("IMAGINE", cognitive_load=0.8, hunger=0.5, boredom=0.6, fatigue=-0.4, bias=0.0)
        # Neuromodulator interactions (implemented as multiplicative scaling in old code)
        # We approximate them by adding base weight * level
        # dopamine boosts non‑rest actions, serotonin boosts REST, noradrenaline penalises complex actions
        for a in POSSIBLE_ACTIONS:
            if a != "REST":
                # dopamine: weight multiplied by (1 + 0.5*dopamine) -> weight += 0.5*dopamine * base ?
                # We can't easily replicate multiplicative without a non‑linear transform, but as a start
                # we add a positive dopamine coefficient
                set_w(a, dopamine=0.01)  # tiny base that will grow with learning
            else:
                set_w(a, serotonin=0.01)
            if a in ("EVOLVE", "REASON", "CONSOLIDATE"):
                # stress penalty: multiplicative (1 - 0.5*stress) -> weight -= 0.5*stress * constant
                # we set a negative noradrenaline weight, which will be learned further
                set_w(a, noradrenaline=-0.01)

        # Context bonus (geology/code) will be learned, but we can give a small initial boost
        for a in ("GEOLOGY_LEARN", "GEOLOGY_EXPLORE"):
            set_w(a, context_geology=0.5)
        for a in ("LEARN", "EXPLORE"):
            set_w(a, context_code=0.3)

        # LTM success rate scaling: old code used (0.5 + ltm_success_rate)
        # That's equivalent to weight * ltm_success if we treat ltm_success as feature
        # We set ltm_success weight to 1.0 for all actions (since old: score *= (0.5+ltm))
        # and bias to absorb the 0.5 part. But we can learn that instead.
        for a in POSSIBLE_ACTIONS:
            set_w(a, ltm_success=0.8, bias=0.5)

        # Ensure no all‑zero weight vectors (bias at least)
        for a in POSSIBLE_ACTIONS:
            if np.all(self.weights[a] == 0.0):
                self.weights[a][-1] = 0.1  # small bias

    # ── Prediction ────────────────────────────────

    def predict(self, action: str, features: np.ndarray) -> float:
        """Return estimated value (score) for the given action and context features."""
        w = self.weights.get(action)
        if w is None:
            return 0.0
        return float(np.dot(w, features))

    def score_all(
        self,
        needs: Dict[str, float],
        neuromodulators: Dict[str, float],
        recalled_concepts_context: List[str],
        ltm_success_rates: Dict[str, float],
    ) -> Dict[str, float]:
        """Return scores for all possible actions."""
        scores = {}
        for action in POSSIBLE_ACTIONS:
            rate = ltm_success_rates.get(action, 0.5)
            feats = _extract_features(
                action, needs, neuromodulators, recalled_concepts_context, rate
            )
            scores[action] = self.predict(action, feats)
        return scores

    # ── Learning update ───────────────────────────

    def update(
        self,
        action: str,
        features: np.ndarray,
        reward: float,
    ) -> None:
        """
        Perform a gradient‑descent step on the weights for the chosen action.
        target = reward (single‑step return; equivalent to TD(0) with no successor state).
        Loss = 0.5 * (target - prediction)^2 + 0.5 * l2_reg * ||w||^2
        """
        if action not in self.weights:
            return

        w = self.weights[action]
        pred = float(np.dot(w, features))
        error = reward - pred

        # Gradient: -(error) * features + l2_reg * w
        grad = -error * features + self.l2_reg * w
        # SGD update
        w -= self.lr * grad

        # Clip weights to reasonable range to avoid explosion
        np.clip(w, -2.0, 2.0, out=w)

    def save(self, filepath: str) -> None:
        """Save weight vectors as a .npz file."""
        arrays = {a: self.weights[a] for a in POSSIBLE_ACTIONS}
        np.savez(filepath, **arrays)

    def load(self, filepath: str) -> Optional[bool]:
        """Load weight vectors from a .npz file. Returns True on success."""
        try:
            with np.load(filepath) as data:
                for a in POSSIBLE_ACTIONS:
                    if a in data:
                        self.weights[a] = data[a]
                return True
        except Exception:
            return False


# ──────────────────────────────────────────────
# Module‑level instance (singleton)
# ──────────────────────────────────────────────

_learner = LinearActionValue()


def score_all_actions(
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    recalled_concepts_context: List[str] = None,
    ltm_success_rates: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Public interface compatible with FlockOfThought.
    Returns a dictionary {action: score} using the current learned value function.
    """
    return _learner.score_all(
        needs,
        neuromodulators,
        recalled_concepts_context or [],
        ltm_success_rates or {},
    )


def update_action_value(
    action: str,
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    recalled_concepts_context: List[str],
    ltm_success_rates: Dict[str, float],
    reward: float,
) -> None:
    """
    Update the value function after executing an action.
    Should be called by FlockOfThought.execute_action or update_learning_systems.
    """
    rate = ltm_success_rates.get(action, 0.5)
    feats = _extract_features(
        action, needs, neuromodulators, recalled_concepts_context, rate
    )
    _learner.update(action, feats, reward)


def get_learner() -> LinearActionValue:
    """Access the underlying learner for persistence or inspection."""
    return _learner

def save(self, filepath: str) -> None:
    arrays = {a: self.weights[a] for a in POSSIBLE_ACTIONS}
    np.savez(filepath, **arrays)

def load(self, filepath: str) -> bool:
    try:
        with np.load(filepath) as data:
            for a in POSSIBLE_ACTIONS:
                if a in data:
                    self.weights[a] = data[a]
        return True
    except Exception:
        return False

def save_learner(filepath: str):
    _learner.save(filepath)

def load_learner(filepath: str):
    return _learner.load(filepath)