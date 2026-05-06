"""
Executive Function - The Gatekeeper

This module acts as a final checkpoint before an action is executed.
It evaluates the chosen action based on a set of principles, constraints,
and the overall state of the system to prevent undesirable or risky behavior.
"""

import random
from typing import Dict, Any, Optional

from core.plan import Plan

def evaluate_action(
    action: str,
    score: float,
    needs: Dict[str, float],
    neuromodulators: Dict[str, float]
) -> bool:
    """
    Evaluates whether to approve or reject a selected action.

    Args:
        action: The action selected by the scoring system.
        score: The score given to the action.
        needs: Current internal needs.
        neuromodulators: Current neuromodulator levels.

    Returns:
        True if the action is approved, False if rejected.
    """
    # Principle 1: Conserve energy. If fatigued, be hesitant.
    if needs.get("fatigue", 0.0) > 0.8:
        if action not in ["REST"] and random.random() > 0.3:
            print(f"EXECUTIVE: Action '{action}' rejected due to high fatigue.")
            return False

    # Principle 2: Self-preservation. Be cautious with self-modification.
    if action == "EVOLVE":
        # Requires high motivation (dopamine) and low stress (cortisol)
        if neuromodulators.get("dopamine", 0.5) < 0.6 or neuromodulators.get("cortisol", 0.1) > 0.4:
            print(f"EXECUTIVE: Action '{action}' rejected due to unfavorable neuromodulatory state.")
            return False
        
        # Add a probabilistic gate for extra safety
        if random.random() > 0.4: # Only a 40% chance of approval
            print(f"EXECUTIVE: Action '{action}' rejected by probabilistic safety gate.")
            return False

    # Principle 3: Avoid pointless loops. If boredom is high, don't rest.
    if action == "REST" and needs.get("boredom", 0.0) > 0.7:
         print(f"EXECUTIVE: Action '{action}' rejected due to high boredom.")
         return False

    # Default approval
    print(f"EXECUTIVE: Action '{action}' approved.")
    return True

def evaluate_plan(plan: Plan, neuromodulators: Dict[str, float]) -> bool:
    """
    Evaluates an entire plan before execution begins.

    Args:
        plan: The plan to evaluate.
        neuromodulators: Current neuromodulator levels.

    Returns:
        True if the plan is approved, False otherwise.
    """
    print(f"EXECUTIVE: Evaluating plan for goal: '{plan.goal}'")
    # Principle: Don't commit to long, dangerous plans in a bad state.
    if "EVOLVE" in plan.actions:
        if neuromodulators.get("cortisol", 0.1) > 0.6:
            print("EXECUTIVE: Plan rejected due to high stress and inclusion of EVOLVE.")
            return False
    
    print("EXECUTIVE: Plan approved.")
    return True
