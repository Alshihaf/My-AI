"""
Executive Function - The Gatekeeper (v2.1)

This module acts as a final checkpoint before an action is executed.
It evaluates the chosen action based on a set of principles, constraints,
and the overall state of the system to prevent undesirable or risky behavior.

Now includes a confidence threshold and rebalanced EVOLVE criteria.
"""

import random
from typing import Dict, Any, Optional, List

from core.plan import Plan

CONFIDENCE_THRESHOLD = 0.2  # Actions with a score below this are generally rejected.

def evaluate_action(
    action: str,
    score: float,
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    stm_labels: List[str] = None,
    ltm_rates: Dict[str, Dict[str, int]] = None
) -> bool:
    if action != "REST" and score < CONFIDENCE_THRESHOLD:
        print(f"EXECUTIVE: Action '{action}' rejected due to low confidence score ({score:.2f}).")
        return False

    # Hindari ulangi aksi yang baru saja gagal berkali-kali
    if ltm_rates and action in ltm_rates:
        stats = ltm_rates[action]
        if stats["total"] >= 5:
            fail_rate = 1.0 - (stats["success"] / stats["total"])
            if fail_rate > 0.7:
                print(f"EXECUTIVE: Action '{action}' rejected because of high historical failure rate ({fail_rate:.2f}).")
                return False

    # Konteks STM: jika tidak ada sinyal relevan, jangan paksakan aksi khusus
    if action in ["GEOLOGY_EXPLORE", "GEOLOGY_LEARN"]:
        context = " ".join(stm_labels or []).lower()
        if not any(w in context for w in ("geology", "macrostrat", "rock", "formation")):
            print(f"EXECUTIVE: Action '{action}' rejected because geology context not present in STM.")
            return False

    # ... aturan fatigue, dll seperti sebelumnya ...
    if needs.get("fatigue", 0.0) > 0.8:
        if action not in ["REST"] and random.random() > 0.3:
            print(f"EXECUTIVE: Action '{action}' rejected due to high fatigue.")
            return False

    if action == "EVOLVE":
        if neuromodulators.get("Dopamine", 0.5) < 0.5 or neuromodulators.get("Noradrenaline", 0.1) > 0.5:
            print(f"EXECUTIVE: Action '{action}' rejected due to unfavorable neuromodulatory state.")
            return False
        if random.random() > 0.5:
            print(f"EXECUTIVE: Action '{action}' rejected by probabilistic safety gate.")
            return False

    if action == "REST" and needs.get("boredom", 0.0) > 0.7 and needs.get("fatigue", 0.0) < 0.8:
        print(f"EXECUTIVE: Action '{action}' rejected due to high boredom.")
        return False

    print(f"EXECUTIVE: Action '{action}' approved.")
    return True

def evaluate_plan(plan: Plan, neuromodulators: Dict[str, float]) -> bool:
    """
    Evaluates an entire plan before execution begins.

    Args:
        plan: The plan to evaluate.
        neuromodulators: Current neuromodulator levels (uses Capitalized keys).

    Returns:
        True if the plan is approved, False otherwise.
    """
    print(f"EXECUTIVE: Evaluating plan for goal: '{plan.goal}'")
    # Principle: Don't commit to long, dangerous plans in a bad state.
    if "EVOLVE" in plan.actions:
        if neuromodulators.get("Noradrenaline", 0.1) > 0.6:
            print("EXECUTIVE: Plan rejected due to high stress and inclusion of EVOLVE.")
            return False
    
    print("EXECUTIVE: Plan approved.")
    return True
