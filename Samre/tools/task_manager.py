# tools/task_manager.py
import json
import os
from typing import Optional, Dict, Any

class TaskManager:
    def __init__(self, task_file="my_task.json"):
        self.task_file = task_file
        self.tasks = []
        self.current_task_index = 0
        self.project_name = "Unnamed Project"
        self.load_tasks()

    def load_tasks(self):
        if not os.path.exists(self.task_file):
            print(f"Task file '{self.task_file}' not found. Starting with no tasks.")
            return
        with open(self.task_file, 'r') as f:
            data = json.load(f)
            self.project_name = data.get("project_name", "Unnamed Project")
            self.tasks = data.get("tasks", [])
            self.current_task_index = 0
            print(f"📋 Task Manager loaded '{self.project_name}' with {len(self.tasks)} tasks.")

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]
            return task
        return None

    def complete_current_task(self):
        print(f"✅ Task {self.current_task_index + 1}/{len(self.tasks)} completed.")
        self.current_task_index += 1

    def is_project_complete(self) -> bool:
        return self.current_task_index >= len(self.tasks)

    def get_project_status(self) -> str:
        return f"Project '{self.project_name}': {self.current_task_index}/{len(self.tasks)} tasks completed."