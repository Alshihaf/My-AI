"""
Flock of Thought - The Autonomous Cognitive Core of Samre (v2.1)

Orchestrates the cognitive cycle, integrating a dynamic connectome
(SamanticGarden) for memory and recall, which influences action selection.
All module interfaces are now properly aligned.
"""
import time
import os
import numpy as np
from typing import Dict, List, Set, Optional
from collections import deque

# Core Components
from core.cognitive_core import CognitiveEngine
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.needs import InternalNeeds
from core.sws_logic import score_all_actions, POSSIBLE_ACTIONS
from core.executive import evaluate_action, evaluate_plan
from core.plan import Plan
from core.planner import Planner
from core.metacognition import Metacognition

# Tools & Actuators
from tools.file_manager import FileManager
from tools.text_processor import TextProcessor
from act.actuators import ExploreActuator, LearningActuator, EvolutionaryActuator
from act.reasoning_actuator import ReasoningActuator
from core.samantic_garden import SamanticGarden, KnowledgeNode
from tools.task_manager import TaskManager
from tools.geology_actuator import GeologyActuator

class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128):
        print("🧠 Initializing Flock of Thought (v2.1 - Connectome Enabled & Integral)...")

        # === Path otomatis ===
        samre_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(samre_dir)
        log_dir = os.path.join(samre_dir, "log")
        
        # === FileManager (SEKALI SAJA) ===
        self.file_manager = FileManager(base_path=samre_dir)

        # === Cognitive & Need Architecture ===
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim)
        self.neuromodulators = NeuromodulatorSystem()
        self.needs = InternalNeeds()
        
        task_file = os.path.join(samre_dir, "my_task.json")
        self.task_manager = TaskManager(task_file=task_file)

        # === State & Memory ===
        self.ltm_action_success = {action: {"success": 0, "total": 0} for action in POSSIBLE_ACTIONS}
        self.current_task: Optional[Dict] = None
        self.short_term_memory: List[KnowledgeNode] = []

        # === Metacognition & Planning ===
        self.metacognition = Metacognition(self.ltm_action_success)
        self.planner = Planner()
        self.current_plan: Optional[Plan] = None

        # === Connectome ===
        self.samantic_garden = SamanticGarden(log_dir=log_dir)

        # === Tools & Actuators (pakai FileManager yang sama) ===
        self.text_processor = TextProcessor(vector_dim=symbolic_dim)
        self.explore_actuator = ExploreActuator(self.file_manager)
        self.learning_actuator = LearningActuator(self.file_manager, self)
        self.reasoning_actuator = ReasoningActuator(self.samantic_garden)
        self.evolutionary_actuator = EvolutionaryActuator(self.file_manager)
        self.geology_actuator = GeologyActuator(self)
        
        # === Eksplorasi state ===
        self.explored_paths: Set[str] = set()
        self.learning_queue: deque[str] = deque()
        self.text_file_extensions = {'.py', '.md', '.txt', '.json', '.xml', '.html', '.css', '.js'}

        # === Cycle & Bootstrap ===
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        self._bootstrap_initial_knowledge()
        print("✅ Flock of Thought initialized and bootstrapped.")

    def _bootstrap_initial_knowledge(self):
        """Read own source code to build an initial IDF model and populate the Garden."""
        print("--- Bootstrapping Self-Awareness ---")
        source_files = [
            "core/flock_of_thought.py",
            "core/cognitive_core.py",
            "core/samantic_garden.py",
            "core/needs.py",
            "core/sws_logic.py",
        ]
        
        # Step 1: Collect all documents into a corpus first.
        corpus_data = {} # Store path -> content mapping
        corpus = []
        print("    1. Collecting bootstrap corpus...")
        for file_path in source_files:
            self.explored_paths.add(file_path)
            read_result = self.file_manager.read_file(file_path)
            if "content" in read_result:
                content = read_result["content"]
                corpus.append(content)
                corpus_data[file_path] = content
            else:
                print(f"⚠️  Could not read own file for bootstrap: {file_path}")
        
        # Step 2: Build the IDF model from the complete corpus.
        if corpus:
            print("    2. Building IDF model from corpus...")
            self.text_processor.update_idf_counts(corpus)
            
        # Step 3: Now, process each document to populate the Garden using the built IDF model.
        print("    3. Populating Samantic Garden with initial knowledge...")
        for file_path, content in corpus_data.items():
            self.process_and_store(content, file_path)

        self.samantic_garden.consolidate_memories()  # Initial consolidation
        print("--- Bootstrap Complete ---")

    def process_and_store(self, content: str, source_path: str) -> List[str]:
        """Processes content, stores it in the Samantic Garden, and returns keywords."""
        print(f"    🌱 Processing content from {source_path} for Samantic Garden.")
        vector, keywords = self.text_processor.text_to_vector(content)
        # Use the symbolic engine to get a more abstract representation
        concept_vector = self.symbolic_engine.query(vector)
        self.samantic_garden.ingest_knowledge(concept_vector, f"File:{source_path}", source_path, list(keywords.keys()))
        return list(keywords.keys())

    def determine_action(self) -> str:
        """Determines the next action based on tasks, plans, and reactive scoring influenced by memory."""
        # 1. Check for an active task.
        if self.current_task:
            return self.current_task["action"]

        # 2. If no active task, try to get a new one.
        task = self.task_manager.get_next_task()
        if task:
            self.current_task = task
            print(f"📌 New Task Acquired: {task['id']} - {task['description']}")
            if task["action"] == "LEARN" and task.get("target"):
                target = task["target"]
                
                if target.startswith("Samre/"):
                    target = target[6:]
                self.learning_queue.append(target) # Only append the cleaned path
            return self.current_task["action"]

        # 3. Plan Following (restored logic)
        if self.current_plan and not self.current_plan.is_complete():
            action = self.current_plan.get_next_action()
            print(f"📘 Executing plan '{self.current_plan.goal}': Step {self.current_plan.current_step + 1} -> {action}")
            return action

        if self.current_plan and self.current_plan.is_complete():
            print(f"🎉 Plan '{self.current_plan.goal}' completed successfully!")
            self.metacognition.record_plan_outcome(self.current_plan, was_successful=True)
            self.current_plan = None

        # 4. Reactive action selection, now influenced by short-term memory
        print("🤔 No active task or plan. Resorting to reactive, memory-influenced action selection.")
        self.needs.update_needs()
        
        # Prepare STM context (labels of recalled concepts)
        recalled_concept_labels = [node.label for node in self.short_term_memory]

        # Build LTM success rates for actions
        ltm_rates = {
            action: (stats["success"] / stats["total"] if stats["total"] > 0 else 0.5)
            for action, stats in self.ltm_action_success.items()
        }

        # Neuromodulator levels are provided with lowercase keys for compatibility with sws_logic
        neuro_levels_raw = self.neuromodulators.get_all_levels()
        neuro_levels_compat = {k.lower(): v for k, v in neuro_levels_raw.items()}

        action_scores = score_all_actions(
            self.needs.get_all_needs(),
            neuro_levels_compat,
            recalled_concepts_context=recalled_concept_labels,
            ltm_success_rates=ltm_rates
        )
        print(f"Needs: {self.needs.get_all_needs()}")
        print(f"STM Context: {recalled_concept_labels}")
        print(f"Action Scores: {action_scores}")
        return self.evaluate_and_select_action(action_scores)

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        """Evaluates scored actions through the executive gate and returns the chosen one."""
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        selected_action = "REST"
        for action, score in sorted_actions:
            # Pass neuromodulator levels with original keys to executive
            if evaluate_action(action, score, self.needs.get_all_needs(), self.neuromodulators.get_all_levels()):
                selected_action = action
                break
        print(f"Action Evaluated & Selected: {selected_action}")
        return selected_action

    def execute_action(self, action: str) -> None:
        """Executes the selected action and triggers memory and learning updates."""
        print(f"⚡️ Delegating execution for action: {action}")
        reward = 0.0
        execution_success = False
        action_keywords: List[str] = []  # Keywords generated by the action's content

        try:
            if action == "EXPLORE":
                new_file_path = self.explore_actuator.execute(self.explored_paths)
                if new_file_path:
                    self.learning_queue.append(new_file_path)
                    self.explored_paths.add(new_file_path)
                    # Can't get keywords from explore directly, but we can infer from file name later
                    reward, execution_success = 0.5, True
                else:
                    reward, execution_success = -0.1, False

            elif action == "LEARN":
                if not self.learning_queue:
                    print("🔎 LEARNING queue is empty. Nothing to learn.")
                    reward, execution_success = -0.3, False
                else:
                    file_to_learn = self.learning_queue.popleft()
                    # Actuator now returns (success, keywords) when return_keywords=True
                    success, keywords = self.learning_actuator.execute(file_to_learn, return_keywords=True)
                    if success:
                        self.needs.satisfy_need("hunger", 0.8)
                        self.needs.increase_need("cognitive_load", 0.4)
                        action_keywords = keywords
                        reward, execution_success = 0.9, True
                    else:
                        reward, execution_success = -0.5, False

            elif action == "GEOLOGY_EXPLORE":
                if self.current_task and action == self.current_task["action"]:
                    target = self.current_task["target"]
                    print(f"🗺️ GEOLOGY_EXPLORE: Checking data availability at {target}.")
                    # This is where the actual action would be performed.
                    # For now, we simulate success.
                    execution_success = True 
                    reward = 0.6
                else:
                    print("⚠️ GEOLOGY_EXPLORE called without a corresponding task.")
                    reward, execution_success = -0.2, False

            elif action == "GEOLOGY_LEARN":
                if self.current_task and action == self.current_task["action"]:
                    target = self.current_task["target"]
                    try:
                        parts = target.split(',')
                        lat = float(parts[0].split(':')[1])
                        lng = float(parts[1].split(':')[1])
                        label = f"Task_{self.current_task['id']}_location"
                        # Actuator returns (success, keywords)
                        success, keywords = self.geology_actuator.learn_from_location(lat, lng, label, return_keywords=True)
                        if success:
                            action_keywords = keywords
                            reward, execution_success = 0.9, True
                        else:
                            reward, execution_success = -0.5, False
                    except (ValueError, IndexError) as e:
                        print(f"💥 ERROR parsing GEOLOGY_LEARN target '{target}': {e}")
                        reward, execution_success = -0.5, False
                else:
                    reward, execution_success = -0.2, False

            elif action == "CONSOLIDATE":
                self.samantic_garden.consolidate_memories()
                self.needs.satisfy_need("cognitive_load", 0.9)
                reward, execution_success = 0.7, True

            elif action == "REASON":
                execution_success = self.reasoning_actuator.execute()
                reward = 0.4 if execution_success else -0.2

            elif action == "EVOLVE":
                execution_success = self.evolutionary_actuator.execute()
                reward = 0.3 if execution_success else -0.6

            elif action == "ORGANIZE":
                print("✍️ ORGANIZING: Writing reasoning results to prediction.csv")
                csv_header = "ID,Predicted_Lithology"
                csv_content = f"{csv_header}\n1,Sandstone"  # Placeholder
                result = self.file_manager.write_file("prediction.csv", csv_content)
                if "succeeded" in result.get("status", ""):
                    execution_success = True
                    reward = 0.8
                else:
                    print(f"💥 Failed to write prediction file: {result}")
                    execution_success = False
                    reward = -0.4

            elif action == "REST":
                # Mind wandering: activate random important nodes gently
                all_ids = list(self.samantic_garden.nodes.keys())
                if all_ids:
                    stimulus_ids = np.random.choice(all_ids, size=min(3, len(all_ids)), replace=False).tolist()
                else:
                    stimulus_ids = []
                self.short_term_memory = self.samantic_garden.spreading_activation(
                    stimulus_ids, initial_signal=0.2, depth=4
                )
                self.needs.satisfy_need("fatigue", 0.9)
                reward, execution_success = 0.2, True

            else:
                print(f"Action '{action}' execution logic is basic. No keywords generated.")
                reward, execution_success = 0.1, True

        except Exception as e:
            print(f"💥 CRITICAL ERROR during {action} execution: {e}")
            reward, execution_success = -1.0, False

        # --- Post-Action Memory and Learning Updates ---
        if execution_success:
            # If the action produced new information, trigger a memory recall cycle
            if action_keywords:
                print(f"🧠 Action produced keywords: {action_keywords}. Triggering recall.")
                # Find nodes that contain any of these keywords to start the activation
                stimulus_nodes = [
                    node.id for node in self.samantic_garden.nodes.values()
                    if not set(node.keywords).isdisjoint(action_keywords)
                ]
                if stimulus_nodes:
                    self.short_term_memory = self.samantic_garden.spreading_activation(
                        stimulus_nodes, initial_signal=1.0, depth=5
                    )
                    print(f"STM Updated. Recalled concepts: {[n.label for n in self.short_term_memory]}")

            # Complete the task if it was the one executed
            if self.current_task and action == self.current_task["action"]:
                print(f"✅ Task {self.current_task['id']} ({action}) completed.")
                self.task_manager.complete_current_task()
                self.current_task = None

        self.cumulative_reward += reward
        print(f"📊 Outcome: Success={execution_success}, Reward={reward:.2f}")
        self.update_learning_systems(action, execution_success, reward)

        # Plan maintenance
        if self.current_plan:
            if execution_success:
                self.current_plan.advance()
            else:
                print(f"Plan '{self.current_plan.goal}' failed at action '{action}'. Attempting to revise.")
                self.metacognition.record_plan_outcome(self.current_plan, was_successful=False)
                revised_plan = self.planner.revise_plan(self.current_plan, f"Action '{action}' failed.")
                self.current_plan = revised_plan

    def update_learning_systems(self, action: str, success: bool, reward: float):
        """Updates LTM, neuromodulators, and Garden plasticity."""
        # 1. Update long-term memory about action success
        self.ltm_action_success[action]["total"] += 1
        if success:
            self.ltm_action_success[action]["success"] += 1

        # 2. Update neuromodulators based on reward prediction error
        expected_reward = 0.1  # Simple baseline
        deltas = NeuromodulatoryEvent.reward_prediction_error(reward, expected_reward)
        self.neuromodulators.update_all(deltas=deltas)

        # 3. Modulate the Garden's plasticity based on arousal (dopamine & noradrenaline)
        levels = self.neuromodulators.get_all_levels()
        dopamine = levels["Dopamine"]
        noradrenaline = levels["Noradrenaline"]
        arousal = (dopamine + noradrenaline) / 2.0
        self.samantic_garden.global_learning_rate = 0.01 * (1 + 1.5 * arousal)

        print(f"Neuromodulators Updated: {levels}")
        print(f"Garden Plasticity (LR): {self.samantic_garden.global_learning_rate:.4f}")

    def run_cycle(self):
        """Runs a single cognitive cycle."""
        print("\n--- CYCLE START ---")
        self.cycle_count += 1
        print(f"🔄 Cycle: {self.cycle_count}")

        if self.task_manager.is_project_complete():
            print("🎉 Project complete!")
            return

        print(self.task_manager.get_project_status())

        # Metacognitive Review & Goal Generation (still available, needs planner rules)
        if not self.current_plan and self.cycle_count % 5 == 0:
            suggestion = self.metacognition.review_performance()
            if suggestion:
                new_plan = self.planner.create_plan(suggestion)
                if new_plan and evaluate_plan(new_plan, self.neuromodulators.get_all_levels()):
                    self.current_plan = new_plan

        selected_action = self.determine_action()
        self.execute_action(selected_action)

        # Periodic consolidation during low-need states
        if self.needs.get_need('cognitive_load') < 0.2 and self.cycle_count % 10 == 0:
            self.samantic_garden.consolidate_memories()

        print("--- CYCLE END ---")