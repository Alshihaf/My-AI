# core/goals_manager.py
import json
import os
from typing import List, Dict, Optional
from datetime import datetime

class GoalsManager:
    def __init__(self, goals_file: str = "goals.json"):
        self.goals_file = goals_file
        self.goals: List[Dict] = []
        self.load_goals()

    def load_goals(self):
        if os.path.exists(self.goals_file):
            try:
                with open(self.goals_file, 'r') as f:
                    data = json.load(f)
                    self.goals = data.get("goals", [])
            except (IOError, json.JSONDecodeError):
                self.goals = []
        else:
            # Initialize with default goals
            self.goals = [
                {
                    "id": "default_1",
                    "description": "Explore and learn from available files",
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                    "priority": 1
                }
            ]
            self.save_goals()

    def save_goals(self):
        with open(self.goals_file, 'w') as f:
            json.dump({"goals": self.goals}, f, indent=4)

    def add_goal(self, description: str, priority: int = 1, status: str = "active"):
        goal = {
            "id": f"goal_{len(self.goals)+1}_{int(datetime.now().timestamp())}",
            "description": description,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "priority": priority
        }
        self.goals.append(goal)
        self.save_goals()
        return goal["id"]

    def get_active_goal(self) -> Optional[str]:
        """Return the highest-priority active goal description."""
        active = [g for g in self.goals if g.get("status") == "active"]
        if not active:
            return None
        # Sort by priority descending
        active.sort(key=lambda x: x.get("priority", 1), reverse=True)
        return active[0]["description"]

    def complete_goal(self, goal_id: str):
        for g in self.goals:
            if g["id"] == goal_id:
                g["status"] = "completed"
                g["completed_at"] = datetime.now().isoformat()
                self.save_goals()
                return

    def generate_goal_from_reflection(self, suggestion: str) -> str:
        """Add a goal suggested by metacognition or other module."""
        return self.add_goal(suggestion, priority=2)