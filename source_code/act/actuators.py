"""
Actuators - Action Execution Components for Samre.

Each class here represents the agent's ability to perform a specific action
in its environment, such as exploring for new information, learning it,
or reasoning about its knowledge.
"""

import os
import random
from typing import Optional, Set, Tuple, List

from tools.file_manager import FileManager
# Correctly import the full class to avoid circular dependency issues with type hinting
import core.flock_of_thought

class ExploreActuator:
    """Finds new, unread files for the agent to learn from."""
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.ignore_dirs = {'.git', '__pycache__', '.idea', 'log'}
        self.valid_extensions = {
            '.py', '.md', '.txt', '.json', '.xml', '.html', '.css', '.js'
        }

    def execute(self, explored_paths: Set[str]) -> Optional[str]:
        """
        Scans the filesystem for a file that has not been explored yet and
        chooses one at random.
        """
        print("🗺️ EXPLORING: Searching for new information sources...")
        new_files = []
        for root, dirs, files in os.walk(self.file_manager.base_path):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.file_manager.base_path)
                    if relative_path not in explored_paths:
                        new_files.append(relative_path)
        
        if new_files:
            chosen_file = random.choice(new_files)
            print(f"    ✅ Found {len(new_files)} new files. Randomly chose: {chosen_file}")
            return chosen_file
        else:
            print("    ⚠️ No new files found to explore.")
            return None


class LearningActuator:
    """Reads a specific file and processes it for learning."""
    def __init__(self, file_manager: FileManager, flock_of_thought: "core.flock_of_thought.FlockOfThought"):
        self.file_manager = file_manager
        self.flock_of_thought = flock_of_thought

    def execute(self, target_file: str, return_keywords: bool = False) -> Tuple[bool, List[str]]:
        """
        Reads, processes, and stores a file's content in the SamanticGarden.
        Can optionally return the keywords extracted from the content.
        """
        print(f"📚 LEARNING: Attempting to learn from '{target_file}'.")
        read_result = self.file_manager.read_file(target_file)
        if "content" in read_result:
            content = read_result["content"]
            # process_and_store now correctly returns the keywords
            keywords = self.flock_of_thought.process_and_store(content, target_file)
            print(f"    🧠 Processed and stored. Keywords: {keywords}")
            if return_keywords:
                return True, keywords
            return True, [] # Return tuple to maintain consistent signature
        else:
            print(f"    ❌ Failure: Could not read '{target_file}'.")
            # Always return a tuple of (bool, list)
            return False, []

class EvolutionaryActuator:
    """Analyzes own source code to prepare for self-modification (conceptual)."""
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    def execute(self, target_file: str = "core/flock_of_thought.py") -> bool:
        print(f"🧬 EVOLVING: Preparing evolution by analyzing '{target_file}'.")
        read_result = self.file_manager.read_file(target_file)
        if "content" in read_result:
            print(f"    ✅ Analysis: Successfully read {len(read_result['content'])} characters.")
            return True
        else:
            print(f"    ❌ Failure: Could not read source code '{target_file}'.")
            return False
