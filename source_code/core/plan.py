"""
Plan Data Structure

Defines the `Plan` object used by the Planner and FlockOfThought.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Plan:
    """Represents a sequence of actions to achieve a specific goal."""
    goal: str
    actions: List[str]
    current_step: int = 0
    original_actions: List[str] = field(init=False)

    def __post_init__(self):
        # Keep a copy of the original plan for reference
        self.original_actions = list(self.actions)

    def get_next_action(self) -> Optional[str]:
        """Returns the next action to be executed."""
        if self.is_complete():
            return None
        return self.actions[self.current_step]

    def advance(self):
        """Moves the plan to the next step."""
        if not self.is_complete():
            self.current_step += 1

    def is_complete(self) -> bool:
        """Checks if all actions in the plan have been executed."""
        return self.current_step >= len(self.actions)

    def __repr__(self) -> str:
        return f"Plan(goal='{self.goal}', actions={self.actions}, current_step={self.current_step})"
