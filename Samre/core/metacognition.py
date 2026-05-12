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

    def review_performance(self, current_needs: Dict[str, float] = None) -> Optional[str]:
        # 1. Cek aksi dengan fail rate tinggi
        for action in list(self.action_success_ltm.keys()):
            stats = self.action_success_ltm[action]
            if stats["total"] > 5:
                failure_rate = 1.0 - (stats["success"] / stats["total"])
                if failure_rate > 0.6:
                    suggestion = f"Improve reliability of {action}"
                    print(f"🤔 METACOGNITION: High failure rate for '{action}' ({failure_rate:.2f}). Suggesting: '{suggestion}'")
                    stats["total"] = 0
                    stats["success"] = 0
                    return suggestion

        # 2. Deteksi kebutuhan yang tidak terpenuhi
        if current_needs:
            if current_needs.get("hunger", 0.0) > 0.7:
                if self.action_success_ltm.get("LEARN", {}).get("success", 0) / max(1, self.action_success_ltm.get("LEARN", {}).get("total", 1)) < 0.5:
                    return "Find and learn from new data sources"
            if current_needs.get("cognitive_load", 0.0) > 0.7:
                return "Prioritize memory consolidation and pruning"

        # 3. Analisis plan terakhir
        if len(self.plan_history) > 3:
            recent = self.plan_history[-3:]
            failed = [p for p in recent if not p['success']]
            if len(failed) >= 2:
                suggestion = "Refactor SamanticGarden to improve planning"
                print(f"🤔 METACOGNITION: Multiple plan failures, suggesting: '{suggestion}'")
                return suggestion

        print("🤔 METACOGNITION: Performance review complete.")
        return None