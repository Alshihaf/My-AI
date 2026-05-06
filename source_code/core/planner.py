"""
Planner - Goal-Oriented Planning Engine

This module is responsible for creating and revising plans to achieve high-level goals.
"""

from typing import List, Optional

from core.plan import Plan

class Planner:
    """
    Creates plans (sequences of actions) to achieve specified goals.
    For now, this is a simple rule-based planner.
    """
    def create_plan(self, goal: str) -> Optional[Plan]:
        """
        Creates a plan based on a high-level goal.

        Args:
            goal: A string describing the goal.

        Returns:
            A Plan object or None if the goal is not recognized.
        """
        print(f"📋 PLANNER: Received new goal: '{goal}'")
        if "refactor" in goal.lower() and "semanticgarden" in goal.lower():
            actions = ["LEARN", "REASON", "EVOLVE", "CONSOLIDATE"]
            return Plan(goal=goal, actions=actions)
        
        # Add more planning rules here for other goals

        print(f"🤔 PLANNER: Don't know how to create a plan for goal: '{goal}'")
        return None

    def revise_plan(self, failed_plan: Plan, error: str) -> Optional[Plan]:
        """
        Revises a plan that failed to execute.

        Args:
            failed_plan: The plan that failed.
            error: A string describing the failure.

        Returns:
            A revised Plan object, or None if no revision is possible.
        """
        print(f"❌ PLANNER: Revising plan for goal '{failed_plan.goal}' due to error: {error}")
        # Simple revision strategy: try a different action
        if failed_plan.get_next_action() == "EVOLVE":
            # If EVOLVE fails, maybe we need to reason more first
            revised_actions = failed_plan.actions[:failed_plan.current_step] + ["REASON", "EVOLVE"]
            return Plan(goal=failed_plan.goal, actions=revised_actions)

        print(f"🤔 PLANNER: Cannot revise the failed plan.")
        return None
