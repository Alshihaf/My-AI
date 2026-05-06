
"""
Manages Samre's internal needs to drive autonomous behavior.
"""
import random

class InternalNeeds:
    """
    Simulates internal needs like hunger, boredom, and fatigue.
    These needs increase over time and influence action selection.
    """
    def __init__(self):
        self.needs = {
            "hunger": 0.0,          # Drive to seek new information/data
            "boredom": 0.0,         # Drive to engage in different tasks
            "fatigue": 0.0,         # Need to rest and consolidate
            "messiness": 0.0,       # Drive to organize and refactor code
            "cognitive_load": 0.0   # Drive to prune and optimize the connectome
        }
        self.need_growth_rates = {
            "hunger": 0.05,
            "boredom": 0.03,
            "fatigue": 0.02,
            "messiness": 0.04,
            "cognitive_load": 0.06 # Builds up as new knowledge is acquired
        }

    def update_needs(self):
        """
        Increase needs over time to create a sense of urgency.
        Called at the beginning of each cycle.
        """
        for need, value in self.needs.items():
            growth = random.uniform(0, self.need_growth_rates[need])
            self.needs[need] = min(1.0, value + growth)

    def get_need(self, need: str) -> float:
        """Get the current level of a specific need."""
        return self.needs.get(need, 0.0)

    def get_all_needs(self) -> dict:
        """Get a dictionary of all current need levels."""
        return self.needs.copy()

    def satisfy_need(self, need: str, amount: float):
        """
        Reduce a need after a corresponding action is performed.
        """
        if need in self.needs:
            self.needs[need] = max(0.0, self.needs[need] - amount)

    def increase_need(self, need: str, amount: float):
        """Manually increase a need, e.g., when a specific event occurs."""
        if need in self.needs:
            self.needs[need] = min(1.0, self.needs[need] + amount)
