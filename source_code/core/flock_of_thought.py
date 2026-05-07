"""
Flock of Thought - The Autonomous Cognitive Core of Samre (v2.4)

This version introduces robust, global error handling within the main cognitive
cycle, preventing the agent from crashing due to unhandled exceptions in its
core logic (determine or execute action).
"""
import time
import os
import numpy as np
from typing import Dict, List, Set, Optional
from collections import deque
import json

# Core Components
from core.cognitive_core import CognitiveEngine
from core.imagination import Imagination
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.needs import InternalNeeds
from core.sws_logic import score_all_actions, POSSIBLE_ACTIONS
from core.executive import evaluate_action, evaluate_plan
from core.plan import Plan
from core.planner import Planner
from core.metacognition import Metacognition
from core.samantic_garden import SamanticGarden, KnowledgeNode
from core.neural_ecosystem import NeuralEcosystem
from core.chain_of_thought import ChainOfThought

# Tools & Actuators
from tools.file_manager import FileManager
from tools.text_processor import TextProcessor
from act.actuators import ExploreActuator, LearningActuator, EvolutionaryActuator
from act.reasoning_actuator import ReasoningActuator
from tools.task_manager import TaskManager
from tools.geology_actuator import GeologyActuator

class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128, use_imagination: bool = True):
        # ... (init logic remains the same) ...
        print("🧠 Initializing Flock of Thought (v2.4 - Robust Error Handling)...")
        samre_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(samre_dir, "log")
        self.persistence_dir = os.path.join(samre_dir, "persistence")
        if not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir)
        self.garden_persistence_file = os.path.join(self.persistence_dir, "samantic_garden.json")
        self.ltm_persistence_file = os.path.join(self.persistence_dir, "ltm_action_success.json")
        self.ecosystem = NeuralEcosystem()
        self.ecosystem.register_module("symbolic_engine", CognitiveEngine(dimensionality=symbolic_dim))
        self.ecosystem.register_module("neuromodulators", NeuromodulatorSystem())
        if use_imagination:
            self.ecosystem.register_module("imagination", Imagination(storage_path=os.path.join(self.log_dir, "imagination")))
        self.needs = InternalNeeds()
        task_file = os.path.join(samre_dir, "my_task.json")
        self.task_manager = TaskManager(task_file=task_file)
        self._load_ltm_action_success()
        self.current_task: Optional[Dict] = None
        self.short_term_memory: List[KnowledgeNode] = []
        self.metacognition = Metacognition(self.ltm_action_success)
        self.planner = Planner()
        self.chain_of_thought = ChainOfThought()
        self.current_plan: Optional[Plan] = None
        self.samantic_garden = SamanticGarden(log_dir=self.log_dir, persistence_file=self.garden_persistence_file)
        self.file_manager = FileManager(base_path=samre_dir)
        self.text_processor = TextProcessor(vector_dim=symbolic_dim)
        self.explore_actuator = ExploreActuator(self.file_manager)
        self.learning_actuator = LearningActuator(self.file_manager, self)
        self.reasoning_actuator = ReasoningActuator(self.samantic_garden)
        self.evolutionary_actuator = EvolutionaryActuator(self.file_manager)
        self.geology_actuator = GeologyActuator(self)
        self.explored_paths: Set[str] = set()
        self.learning_queue: deque[str] = deque()
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        if not self.samantic_garden.nodes: self._bootstrap_initial_knowledge()
        print("✅ Flock of Thought initialized.")

    def _load_ltm_action_success(self):
        if os.path.exists(self.ltm_persistence_file):
            try:
                with open(self.ltm_persistence_file, 'r') as f: self.ltm_action_success = json.load(f)
                for action in POSSIBLE_ACTIONS: 
                    if action not in self.ltm_action_success: self.ltm_action_success[action] = {"success": 0, "total": 0}
            except (IOError, json.JSONDecodeError) as e: self.ltm_action_success = {a: {"success": 0, "total": 0} for a in POSSIBLE_ACTIONS}
        else: self.ltm_action_success = {a: {"success": 0, "total": 0} for a in POSSIBLE_ACTIONS}

    def save_state(self):
        print("💾 Saving agent state...")
        self.samantic_garden.save_state()
        try:
            with open(self.ltm_persistence_file, 'w') as f: json.dump(self.ltm_action_success, f, indent=4)
            print(f"✅ LTM saved to {self.ltm_persistence_file}")
        except IOError as e: print(f"❌ Error saving LTM: {e}")
        print("✅ Agent state saving complete.")
    
    def _bootstrap_initial_knowledge(self):
        print("--- Bootstrapping ---")
        source_files = ["core/flock_of_thought.py", "core/cognitive_core.py"]
        corpus = []
        corpus_data = {}
        for file_path in source_files:
            res = self.file_manager.read_file(file_path)
            if "content" in res: corpus.append(res["content"]); corpus_data[file_path] = res["content"]
        if corpus: self.text_processor.update_idf_counts(corpus)
        for file_path, content in corpus_data.items(): self.process_and_store(content, file_path)
        self.samantic_garden.consolidate_memories()
        print("--- Bootstrap Complete ---")

    def process_and_store(self, content: str, source_path: str):
        vector, keywords = self.text_processor.text_to_vector(content)
        concept_vector = self.ecosystem.get_module("symbolic_engine").query(vector)
        self.samantic_garden.ingest_knowledge(concept_vector, f"File:{source_path}", source_path, list(keywords.keys()))

    def determine_action(self) -> str:
        # ... (logic is the same) ...
        if self.current_task: return self.current_task["action"]
        task = self.task_manager.get_next_task()
        if task: 
            self.current_task = task
            if task["action"] == "LEARN" and task.get("target"): self.learning_queue.append(task["target"])
            return self.current_task["action"]
        if self.current_plan and not self.current_plan.is_complete(): return self.current_plan.get_next_action()
        if self.current_plan and self.current_plan.is_complete(): self.metacognition.record_plan_outcome(self.current_plan, True); self.current_plan = None
        self.needs.update_needs()
        stm_labels = [n.label for n in self.short_term_memory]
        ltm_rates = {a: (s["success"] / s["total"] if s["total"] > 0 else 0.5) for a, s in self.ltm_action_success.items()}
        neuro_levels = self.ecosystem.get_module("neuromodulators").get_all_levels()
        action_scores = score_all_actions(self.needs.get_all_needs(), {k.lower():v for k,v in neuro_levels.items()}, stm_labels, ltm_rates)
        return self.evaluate_and_select_action(action_scores)

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        for action, score in sorted_actions:
            if evaluate_action(action, score, self.needs.get_all_needs(), self.ecosystem.get_module("neuromodulators").get_all_levels()): return action
        return "REST"

    def execute_action(self, action: str):
        # ... (logic is the same, but now wrapped in run_cycle) ...
        reward, success, keywords = 0.0, False, []
        # This method is now called within a try-except block in run_cycle
        if action == "EXPLORE":
            new_file = self.explore_actuator.execute(self.explored_paths)
            if new_file: self.learning_queue.append(new_file); self.explored_paths.add(new_file); reward, success = 0.5, True
        elif action == "LEARN":
            if self.learning_queue:
                f_learn = self.learning_queue.popleft()
                s, k = self.learning_actuator.execute(f_learn, return_keywords=True)
                if s: self.needs.satisfy_need("hunger", 0.8); keywords = k; reward, success = 0.9, True
            else: reward, success = -0.3, False
        # ... other actions
        else: reward, success = 0.1, True

        if success:
            if keywords: 
                nodes = [n.id for n in self.samantic_garden.nodes.values() if not set(n.keywords).isdisjoint(keywords)]
                if nodes: self.short_term_memory = self.samantic_garden.spreading_activation(nodes, 1.0, 5)
            if self.current_task and action == self.current_task["action"]: self.task_manager.complete_current_task(); self.current_task = None
        self.cumulative_reward += reward
        self.update_learning_systems(action, success, reward)
        if self.current_plan: 
            if success: self.current_plan.advance()
            else: self.current_plan = None

    def update_learning_systems(self, action: str, success: bool, reward: float):
        # ... (logic is the same) ...
        self.ltm_action_success[action]["total"] += 1
        if success: self.ltm_action_success[action]["success"] += 1
        neuromodulators = self.ecosystem.get_module("neuromodulators")
        deltas = NeuromodulatoryEvent.reward_prediction_error(reward, 0.1)
        neuromodulators.update_all(deltas=deltas)
        levels = neuromodulators.get_all_levels()
        arousal = (levels["Dopamine"] + levels["Noradrenaline"]) / 2.0
        self.samantic_garden.global_learning_rate = 0.01 * (1 + 1.5 * arousal)

    def run_cycle(self):
        """Runs a single cognitive cycle with robust global error handling."""
        print("
--- CYCLE START ---")
        self.cycle_count += 1
        print(f"🔄 Cycle: {self.cycle_count}")
        try:
            if self.task_manager.is_project_complete():
                print("🎉 Project complete!")
                # Potentially trigger a final save or shutdown sequence here
                return

            print(self.task_manager.get_project_status())

            # Metacognition and Planning
            if not self.current_plan and self.cycle_count % 5 == 0:
                suggestion = self.metacognition.review_performance()
                if suggestion:
                    context = {"needs": self.needs.get_all_needs(), "stm": [n.label for n in self.short_term_memory]}
                    steps = self.chain_of_thought.generate_steps(suggestion, context)
                    if steps:
                        new_plan = Plan(goal=suggestion, steps=steps)
                        if evaluate_plan(new_plan, self.ecosystem.get_module("neuromodulators").get_all_levels()):
                            self.current_plan = new_plan
            
            # Action Determination and Execution
            selected_action = self.determine_action()
            self.execute_action(selected_action)

            # System Maintenance
            if self.needs.get_need('cognitive_load') < 0.2:
                self.samantic_garden.consolidate_memories()
            self.ecosystem.update_modules(self.cycle_count)

        except Exception as e:
            # INTEGRATION: Global exception handler
            print(f"💥💥 CRITICAL UNHANDLED EXCEPTION IN CYCLE {self.cycle_count} 💥💥")
            print(f"Error: {e}")
            print("Agent will attempt to recover by resting and increasing cognitive load.")
            # Penalize and attempt to recover
            self.update_learning_systems("CRITICAL_FAILURE", False, -2.0) # Virtual action
            self.needs.increase_need("cognitive_load", 0.8) # Trigger consolidation
            self.current_plan = None # Abandon current plan as it might be the cause
        
        print("--- CYCLE END ---")
