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
        print(f"📋 PLANNER: Received new goal: '{goal}'")
        g = goal.lower()

        # Deteksi domain
        geology_keywords = ["geology", "rock", "mineral", "formation", "sedimentary", "volcanic", "macrostrat"]
        code_keywords = ["code", "refactor", "bug", "fix", "implement", "feature", "debug"]
        data_keywords = ["data", "csv", "json", "dataset", "clean", "preprocess", "analyze"]
        cognitive_keywords = ["planning", "improve planning", "reduce failure", "optimize", "performance"]

        if any(w in g for w in geology_keywords):
            return Plan(goal=goal, actions=["GEOLOGY_EXPLORE", "GEOLOGY_LEARN", "CONSOLIDATE"])
        if any(w in g for w in code_keywords):
            return Plan(goal=goal, actions=["EXPLORE", "LEARN", "REASON", "EVOLVE", "CONSOLIDATE"])
        if any(w in g for w in data_keywords):
            return Plan(goal=goal, actions=["EXPLORE", "LEARN", "REASON", "ORGANIZE"])
        if any(w in g for w in cognitive_keywords):
            return Plan(goal=goal, actions=["LEARN", "REASON", "CONSOLIDATE", "CONTEMPLATE"])

        if "improve reliability" in g:
            if "explore" in g:
                return Plan(goal=goal, actions=["LEARN", "REASON", "EXPLORE", "CONSOLIDATE"])
            if "learn" in g:
                return Plan(goal=goal, actions=["EXPLORE", "LEARN"])
            if "reason" in g:
                return Plan(goal=goal, actions=["LEARN", "CONSOLIDATE", "REASON"])
            return Plan(goal=goal, actions=["EXPLORE", "LEARN", "CONSOLIDATE"])

        if "refactor" in g or "evolve" in g or "organize" in g:
            return Plan(goal=goal, actions=["LEARN", "REASON", "EVOLVE", "CONSOLIDATE"])

        if "boredom" in g and "high" in g:
            return Plan(goal=goal, actions=["EXPLORE", "IMAGINE"])

        print(f"🤔 PLANNER: No specific rule, creating default learning plan.")
        return Plan(goal=goal, actions=["EXPLORE", "LEARN", "REASON", "CONSOLIDATE"])

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

