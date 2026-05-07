"""
Metacognition - The Reflective Mind (v2.1)

This module provides the agent with the ability to reason about its own
cognitive processes, performance, and limitations.

Now generates more specific, actionable goals for the Planner.
"""

from typing import Dict, List, Optional, Any

from core.plan import Plan

class Metacognition:
    """
    Analyzes the agent's performance and suggests strategic adjustments.
    """
    def __init__(self, action_success_ltm: Dict[str, Dict[str, int]]):
        self.action_success_ltm = action_success_ltm
        self.plan_history: List[Dict[str, Any]] = []

    def record_plan_outcome(self, plan: Plan, was_successful: bool):
        """
        Records the outcome of a completed plan.

        Args:
            plan: The plan that was executed.
            was_successful: Whether the plan achieved its goal.
        """
        self.plan_history.append({
            "goal": plan.goal,
            "actions": plan.original_actions,
            "success": was_successful,
        })
        print(f"🤔 METACOGNITION: Recorded outcome for plan '{plan.goal}'. Success: {was_successful}")

    def review_performance(self) -> Optional[str]:
        """
        Reviews overall performance and suggests a new high-level goal if needed.

        Returns:
            A string suggesting a new goal, or None.
        """
        # 1. Analyze action failure rates
        for action, stats in self.action_success_ltm.items():
            if stats["total"] > 5:
                failure_rate = 1.0 - (stats["success"] / stats["total"])
                if failure_rate > 0.6:
                    # Be specific for the planner
                    suggestion = f"Improve reliability of {action}"
                    print(f"🤔 METACOGNITION: High failure rate for action '{action}'. Suggesting goal: '{suggestion}'")
                    # Reset stats to give the new plan a chance to work without being immediately overwritten
                    stats["total"] = 0
                    stats["success"] = 0
                    return suggestion

        # 2. Analyze plan failures
        if len(self.plan_history) > 3:
            recent_plans = self.plan_history[-3:]
            failed_plans = [p for p in recent_plans if not p['success']]
            if len(failed_plans) >= 2:
                 # This is still a bit generic, but let's make it a known goal for the planner
                 suggestion = "Refactor SemanticGarden to improve planning"
                 print(f"🤔 METACOGNITION: Multiple recent plan failures. Suggesting goal: '{suggestion}'")
                 return suggestion

        print("🤔 METACOGNITION: Performance review complete. No immediate concerns.")
        return None
