"""
Planner - Goal-Oriented Planning Engine (v2.2)

This version significantly expands the rule set to create more nuanced and
context-aware plans for a wider variety of metacognitive suggestions.
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
        Creates a plan based on a high-level goal, now with more comprehensive rules.
        """
        print(f"📋 PLANNER: Received new goal: '{goal}'")
        goal_lower = goal.lower()

        # Rule for general knowledge acquisition
        if "understand" in goal_lower or "learn about" in goal_lower:
            return Plan(goal=goal, actions=["EXPLORE", "LEARN", "REASON", "CONSOLIDATE"])

        # Rule for improving a specific, underperforming action
        if "improve reliability" in goal_lower:
            if "explore" in goal_lower:
                return Plan(goal=goal, actions=["LEARN", "REASON", "EXPLORE", "CONSOLIDATE"])
            if "learn" in goal_lower:
                return Plan(goal=goal, actions=["EXPLORE", "LEARN"])
            if "reason" in goal_lower:
                 return Plan(goal=goal, actions=["LEARN", "CONSOLIDATE", "REASON"])
            return Plan(goal=goal, actions=["EXPLORE", "LEARN", "CONSOLIDATE"]) # General improvement

        # Rule for code or self-improvement tasks
        if "refactor" in goal_lower or "evolve" in goal_lower or "organize" in goal_lower:
            return Plan(goal=goal, actions=["LEARN", "REASON", "EVOLVE", "CONSOLIDATE"])
        
        # Rule for geological tasks
        if "geology" in goal_lower:
            if "find data" in goal_lower or "explore area" in goal_lower:
                return Plan(goal=goal, actions=["GEOLOGY_EXPLORE", "GEOLOGY_LEARN"])
            if "analyze" in goal_lower or "predict" in goal_lower:
                return Plan(goal=goal, actions=["GEOLOGY_LEARN", "REASON", "ORGANIZE"])

        # Rule for when boredom is high - seek novelty
        if "boredom" in goal_lower and "high" in goal_lower:
            return Plan(goal=goal, actions=["EXPLORE", "IMAGINE"])
        
        print(f"🤔 PLANNER: No specific rule found for goal: '{goal}'. Creating a default plan.")
        # Default plan: Learn, Reason, Consolidate
        return Plan(goal=goal, actions=["LEARN", "REASON", "CONSOLIDATE"])

    def revise_plan(self, failed_plan: Plan, error: str) -> Optional[Plan]:
        """
        Revises a plan that failed. Now with more specific revision strategies.
        """
        print(f"❌ PLANNER: Revising plan for goal '{failed_plan.goal}' due to error: {error}")
        next_action = failed_plan.get_next_action()

        # If any action fails, a simple but effective strategy is to consolidate existing knowledge.
        if failed_plan.current_step > 0:
            # Try to consolidate and then re-try the failed action
            revised_actions = failed_plan.actions[:failed_plan.current_step] + ["CONSOLIDATE", next_action] + failed_plan.actions[failed_plan.current_step+1:]
            print(f"    -> Inserting CONSOLIDATE before retrying '{next_action}'.")
            return Plan(goal=failed_plan.goal, actions=revised_actions, current_step=failed_plan.current_step)

        # If the very first action fails, the plan might be flawed. Create a simpler one.
        print("    -> First action of plan failed. Creating a more basic plan.")
        return self.create_plan(f"learn about the environment") # A safe, general goal

