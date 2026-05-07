"""
Planner - Goal-Oriented Planning Engine (v2.1)

This module is responsible for creating and revising plans to achieve high-level goals.
Expanded with more rules to handle metacognitive suggestions.
"""

from typing import List, Optional

from core.plan import Plan

class Planner:
    """
    Creates plans (sequences of actions) to achieve specified goals.
    This is an expanded rule-based planner.
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
        goal_lower = goal.lower()

        if "refactor" in goal_lower and "semanticgarden" in goal_lower:
            actions = ["LEARN", "REASON", "EVOLVE", "CONSOLIDATE"]
            return Plan(goal=goal, actions=actions)
        
        if "improve reliability" in goal_lower:
            if "explore" in goal_lower:
                # If EXPLORE is failing, we need to learn more about the environment.
                actions = ["EXPLORE", "LEARN", "REASON", "CONSOLIDATE"]
                return Plan(goal=goal, actions=actions)
            if "learn" in goal_lower:
                # If LEARN is failing, maybe we need to find more files first.
                actions = ["EXPLORE", "LEARN"]
                return Plan(goal=goal, actions=actions)

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
        
        # Simple revision strategy: try a different action.
        next_action = failed_plan.get_next_action()
        if next_action == "EVOLVE":
            # If EVOLVE fails, maybe we need to reason more first.
            revised_actions = failed_plan.actions[:failed_plan.current_step] + ["REASON", "EVOLVE"]
            return Plan(goal=failed_plan.goal, actions=revised_actions)

        # If any action in an "improve reliability" plan fails, try resting to reset state.
        if "improve reliability" in failed_plan.goal.lower():
            revised_actions = failed_plan.actions[:failed_plan.current_step] + ["REST", next_action]
            print("    -> Inserting REST action to reset state.")
            return Plan(goal=failed_plan.goal, actions=revised_actions)

        print(f"🤔 PLANNER: Cannot revise the failed plan.")
        return None
