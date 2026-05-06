"""
SWS Logic - Slow Wave Sleep / System Wide Simulation (v2.1)

This module scores actions based on needs, context, and neuromodulators.
Corrected to use proper neuromodulator dictionary keys.
"""
import random
from typing import Dict, List

POSSIBLE_ACTIONS = [
    "EXPLORE", "EVOLVE", "ORGANIZE", "REST", "LEARN", "REASON", 
    "CONSOLIDATE", "GEOLOGY_EXPLORE", "GEOLOGY_LEARN",
]

def foresight_simulation(
    action: str,
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    recalled_concepts_context: List[str],
    ltm_success_rate: float = 0.5,
) -> float:
    """
    Calculates a score for a given action, influenced by context and neuromodulation.
    """
    score = 0.0
    
    # 1. Influence of Internal Needs
    if action == "EXPLORE" or action == "GEOLOGY_EXPLORE":
        score += needs.get("hunger", 0.0) * 1.5
        score += needs.get("boredom", 0.0) * 1.0
    elif action == "LEARN" or action == "GEOLOGY_LEARN":
        score += needs.get("hunger", 0.0) * 1.2
    elif action == "CONSOLIDATE":
        score += needs.get("cognitive_load", 0.0) * 1.8
    elif action == "REST":
        score += needs.get("fatigue", 0.0) * 2.0
        if all(n < 0.2 for n in needs.values()): score += 0.3

    # 2. Influence of Short-Term Memory Context
    context_str = " ".join(recalled_concepts_context).lower()
    if "geology" in context_str or "macrostrat" in context_str or "rock" in context_str:
        if action == "GEOLOGY_LEARN":
            score *= 2.5
        if action == "GEOLOGY_EXPLORE":
            score *= 2.0
    if "file" in context_str or "source" in context_str or "code" in context_str:
        if action == "LEARN":
            score *= 1.8
        if action == "EXPLORE":
            score *= 1.5

    # 3. Influence of Neuromodulators (Corrected Keys)
    # The dictionary passed here uses Capitalized keys from NeuromodulatorSystem.get_all_levels()
    dopamine = neuromodulators.get("Dopamine", 0.5)
    serotonin = neuromodulators.get("Serotonin", 0.5)
    # Using Noradrenaline as a proxy for stress/cortisol for now
    stress = neuromodulators.get("Noradrenaline", 0.1)

    if action not in ["REST"]:
        score *= (1.0 + dopamine * 0.5)
    if action == "REST":
        score *= (1.0 + serotonin * 0.3)
    if action in ["EVOLVE", "REASON", "CONSOLIDATE"]:
        score *= (1.0 - stress * 0.5) # Stress makes complex thought harder

    # 4. Influence of Learning and Randomness
    score *= (0.5 + ltm_success_rate)
    score += random.uniform(-0.05, 0.05)
    
    return max(0, score)

def score_all_actions(
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    recalled_concepts_context: List[str] = [],
    ltm_success_rates: Dict[str, float] = {}
) -> Dict[str, float]:
    """
    Scores all possible actions, passing the full context to the simulation.
    """
    scores = {}
    for action in POSSIBLE_ACTIONS:
        success_rate = ltm_success_rates.get(action, 0.5)
        scores[action] = foresight_simulation(
            action, 
            needs, 
            neuromodulators, 
            recalled_concepts_context, 
            success_rate
        )
    return scores
