"""
Flock of Thought - The Autonomous Cognitive Core of Samre (v2.5.0)

Changelog v2.5.0:
- Refactored planning: Metacognition → Planner creates Plan directly.
- Introduced CONTEMPLATE action: deep reflective reasoning using ReasoningEngine,
  FileManager, and Imagination. Symbolic facts are extracted from strong synapses
  in the SamanticGarden and used for logical inference.
- Replaced self.chain_of_thought (ChainOfThought) with self.reasoning_engine (ReasoningEngine).
- Fixed neuromodulator key mismatch and LEARN keyword extraction (inherited from v2.4.1).
"""

import time
import os
import random
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import deque
import json

# Core Components
from .cognitive_core import CognitiveEngine
from .imagination import Imagination
from .neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from .needs import InternalNeeds
from .sws_logic import (
    score_all_actions,
    POSSIBLE_ACTIONS,
    update_action_value,
    load_learner,
    save_learner,
)
from .executive import evaluate_action, evaluate_plan
from .plan import Plan
from .planner import Planner
from .metacognition import Metacognition
from .samantic_garden import SamanticGarden, NeuralNode
from .neural_ecosystem import NeuralEcosystem
from .chain_of_thought import ReasoningEngine

# Tools & Actuators
from tools.file_manager import FileManager
from tools.text_processor import TextProcessor
from act.actuators import ExploreActuator, LearningActuator, EvolutionaryActuator
from act.reasoning_actuator import ReasoningActuator
from tools.task_manager import TaskManager
from tools.geology_actuator import GeologyActuator


class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128, use_imagination: bool = True):
        print("🧠 Initializing Flock of Thought (v2.5.0 - Contemplative Reasoning)...")
        samre_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Buat direktori data jika belum ada
        self.data_dir = os.path.join(samre_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.log_dir = os.path.join(samre_dir, "log")
        self.persistence_dir = os.path.join(samre_dir, "persistence")
        if not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir)

        self.garden_persistence_file = os.path.join(self.persistence_dir, "samantic_garden.json")
        self.ltm_persistence_file = os.path.join(self.persistence_dir, "ltm_action_success.json")
        self.sws_file = os.path.join(self.persistence_dir, "sws_weight.npz")

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
        self.short_term_memory: List[NeuralNode] = []
        self.metacognition = Metacognition(self.ltm_action_success)
        self.planner = Planner()
        self.reasoning_engine = ReasoningEngine(search_strategy="depth_first")
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
        self.path_access_times: Dict[str, float] = {}
        self.learning_queue: deque[str] = deque()
        self.cycle_count = 0
        self.cumulative_reward = 0.0

        if not self.samantic_garden.nodes:
            self._bootstrap_initial_knowledge()

        # Muat bobot SWS jika ada
        if os.path.exists(self.sws_file):
            load_learner(self.sws_file)

        print("✅ Flock of Thought initialized.")

    # -----------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------
    def _load_ltm_action_success(self):
        if os.path.exists(self.ltm_persistence_file):
            try:
                with open(self.ltm_persistence_file, 'r') as f:
                    self.ltm_action_success = json.load(f)
                for action in POSSIBLE_ACTIONS:
                    if action not in self.ltm_action_success:
                        self.ltm_action_success[action] = {"success": 0, "total": 0}
            except (IOError, json.JSONDecodeError):
                self.ltm_action_success = {a: {"success": 0, "total": 0} for a in POSSIBLE_ACTIONS}
        else:
            self.ltm_action_success = {a: {"success": 0, "total": 0} for a in POSSIBLE_ACTIONS}

        if "CRITICAL_FAILURE" not in self.ltm_action_success:
            self.ltm_action_success["CRITICAL_FAILURE"] = {"success": 0, "total": 0}

    def save_state(self):
        print("💾 Saving agent state...")
        self.samantic_garden.save_state()
        try:
            with open(self.ltm_persistence_file, 'w') as f:
                json.dump(self.ltm_action_success, f, indent=4)
            print(f"✅ LTM saved to {self.ltm_persistence_file}")
        except IOError as e:
            print(f"❌ Error saving LTM: {e}")

        # Simpan bobot SWS
        save_learner(self.sws_file)
        print("✅ SWS weights saved.")

        # Pangkas explored_paths jika terlalu besar
        if len(self.explored_paths) > 1000:
            keep_size = min(len(self.explored_paths) // 2, 500)
            # Ambil sampel acak karena kita tidak mencatat waktu akses dengan baik
            self.explored_paths = set(random.sample(list(self.explored_paths), keep_size))
            print(f"✂️ Trimmed explored_paths to {len(self.explored_paths)} entries.")

        print("✅ Agent state saving complete.")

    # -----------------------------------------------------------------
    # Auto detect file format
    # -----------------------------------------------------------------
    def _prepare_gourmet_data(self, file_path: str, content: str) -> str:
        """
        Mengubah isi file data mentah menjadi teks naratif yang kaya konteks
        untuk dicerna oleh SamanticGarden.
        """
        ext = os.path.splitext(file_path)[1].lower()
        enhanced_text = content

        try:
            if ext == '.csv':
                import csv, io
                reader = csv.DictReader(io.StringIO(content))
                rows = list(reader)
                if not rows:
                    return content

                headers = reader.fieldnames
                sample_rows = rows[:50]

                descriptions = []
                descriptions.append(f"Dataset file: '{os.path.basename(file_path)}'.")
                descriptions.append(f"Columns: {', '.join(headers)}.")
                descriptions.append("Sample data:")

                for i, row in enumerate(sample_rows):
                    parts = []
                    for key, val in row.items():
                        if val and val.strip():
                            parts.append(f"{key} is {val}")
                    if parts:
                        descriptions.append(f"Row {i+1}: {'. '.join(parts)}.")

                enhanced_text = " ".join(descriptions)

            elif ext == '.json':
                import json
                data = json.loads(content)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    headers = list(data[0].keys())
                    sample = data[:50]
                    descriptions = []
                    descriptions.append(f"JSON dataset: '{os.path.basename(file_path)}'.")
                    descriptions.append(f"Contains {len(data)} records with keys: {headers}.")
                    descriptions.append("Sample entries:")
                    for i, item in enumerate(sample):
                        parts = [f"{k} = {v}" for k, v in item.items()]
                        descriptions.append(f"Entry {i+1}: {', '.join(parts)}.")
                    enhanced_text = " ".join(descriptions)

        except Exception as e:
            print(f"⚠️ Gagal menyempurnakan data dari {file_path}: {e}")

        return enhanced_text

    # -----------------------------------------------------------------
    # Learn from the data/ directory
    # -----------------------------------------------------------------
    def _ingest_data_files(self):
        valid_extensions = {'.py', '.md', '.txt', '.json', '.xml', '.html', '.css', '.js', '.csv', '.log'}

        result = self.file_manager.list_files("data")
        if "error" in result:
            return

        files = result.get("files", [])
        if not files:
            return

        new_files_processed = 0
        for filename in files:
            relative_path = f"data/{filename}"
            if relative_path in self.explored_paths:
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue

            read_result = self.file_manager.read_file(relative_path)
            if "content" in read_result:
                content = read_result["content"]
                try:
                    keywords = self.process_and_store(content, relative_path)
                    self.explored_paths.add(relative_path)
                    new_files_processed += 1
                    print(f"📂 DATA INGEST: Learned from '{relative_path}' ({len(keywords)} keywords)")
                except Exception as e:
                    print(f"⚠️ Gagal memproses '{relative_path}': {e}")

        if new_files_processed > 0:
            print(f"✅ Data ingestion: {new_files_processed} new file(s) processed.")
            self.needs.satisfy_need("hunger", 0.3 * new_files_processed)

    # -----------------------------------------------------------------
    # Bootstrapping
    # -----------------------------------------------------------------
    def _bootstrap_initial_knowledge(self):
        print("--- Bootstrapping ---")
        source_files = [
            "core/flock_of_thought.py",
            "core/cognitive_core.py",
            "core/samantic_garden.py",
            "core/neural_ecosystem.py",
            "core/imagination.py",
            "core/sws_logic.py",
            "core/executive.py",
            "core/planner.py",
            "core/metacognition.py",
            "core/needs.py",
            "core/neuromodulator.py",
            "core/plan.py",
            "data/test_data (1).csv",
            "data/train_data (1).csv",
            "data/train-v1.1.json",
            "data/SP500_Stock_Data.csv",
        ]
        corpus = []
        corpus_data = {}
        for file_path in source_files:
            res = self.file_manager.read_file(file_path)
            if "content" in res:
                corpus.append(res["content"])
                corpus_data[file_path] = res["content"]
        if corpus:
            self.text_processor.update_idf_counts(corpus)
        for file_path, content in corpus_data.items():
            self.process_and_store(content, file_path)
        self.samantic_garden.consolidate_memories()

        if self.samantic_garden.nodes:
            self.short_term_memory = list(self.samantic_garden.nodes.values())[:5]
            print(f"🧠 Short‑term memory initialized with {len(self.short_term_memory)} nodes.")

        print("--- Bootstrap Complete ---")

    # -----------------------------------------------------------------
    # Knowledge ingestion
    # -----------------------------------------------------------------
    def process_and_store(self, content: str, source_path: str) -> List[str]:
        vector, keywords = self.text_processor.text_to_vector(content)
        concept_vector = self.ecosystem.get_module("symbolic_engine").query(vector)
        self.samantic_garden.ingest_knowledge(
            concept_vector,
            f"File:{source_path}",
            source_path,
            list(keywords.keys())
        )
        return list(keywords.keys())

    # -----------------------------------------------------------------
    # Reward System yang dinamis
    # -----------------------------------------------------------------
    def _compute_learning_reward(self, keywords_before: Set[str], keywords_after: Set[str],
                                 source_path: str) -> float:
        """
        Menghitung reward berdasarkan bertambahnya keyword baru dan penguatan node.
        """
        new_keywords = keywords_after - keywords_before
        if not new_keywords:
            return -0.1

        related_nodes = [node for node in self.samantic_garden.nodes.values()
                         if not new_keywords.isdisjoint(node.keywords)]
        if not related_nodes:
            return 0.2

        importance_gain = sum(max(0, node.importance - 0.5) for node in related_nodes)
        novelty_bonus = min(0.5, len(new_keywords) * 0.05)
        reward = importance_gain * 0.4 + novelty_bonus
        return max(0.0, min(1.0, reward))

    # -----------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------
    def determine_action(self) -> str:
        if self.current_task:
            return self.current_task["action"]
        task = self.task_manager.get_next_task()
        if task:
            self.current_task = task
            if task["action"] == "LEARN" and task.get("target"):
                self.learning_queue.append(task["target"])
            return self.current_task["action"]
        if self.current_plan and not self.current_plan.is_complete():
            return self.current_plan.get_next_action()
        if self.current_plan and self.current_plan.is_complete():
            self.metacognition.record_plan_outcome(self.current_plan, True)
            self.current_plan = None

        stm_labels = [n.label for n in self.short_term_memory]
        ltm_rates = {
            a: (s["success"] / s["total"] if s["total"] > 0 else 0.5)
            for a, s in self.ltm_action_success.items()
        }
        neuro_levels = self.ecosystem.get_module("neuromodulators").get_all_levels()
        action_scores = score_all_actions(
            self.needs.get_all_needs(),
            neuro_levels,
            stm_labels,
            ltm_rates
        )
        return self.evaluate_and_select_action(action_scores)

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        if len(self.samantic_garden.nodes) < 3 and self.cycle_count > 3:
            print("🔍 FORCING EXPLORE: Garden has too few nodes")
            return "EXPLORE"

        if self.cycle_count > 8 and not self.explored_paths:
            print("🔍 FORCING EXPLORE: No file discovered yet.")
            return "EXPLORE"

        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        neuro_levels = self.ecosystem.get_module("neuromodulators").get_all_levels()
        for action, score in sorted_actions:
            if evaluate_action(action, score, self.needs.get_all_needs(), neuro_levels,
                               [n.label for n in self.short_term_memory], self.ltm_action_success):
                return action
        return "REST"

    # -----------------------------------------------------------------
    # Action execution
    # -----------------------------------------------------------------
    def execute_action(self, action: str):
        reward, success, keywords = 0.0, False, []

        if action == "CONSOLIDATE":
            self.samantic_garden.consolidate_memories()
            self.needs.satisfy_need("cognitive_load", 0.5)
            reward, success = 0.9, True

        elif action == "EXPLORE":
            new_file = self.explore_actuator.execute(self.explored_paths)
            if new_file:
                self.learning_queue.append(new_file)
                self.explored_paths.add(new_file)
                reward, success = 0.5, True

        elif action == "LEARN":
            if self.learning_queue:
                f_learn = self.learning_queue.popleft()
                keywords_before = set().union(*[node.keywords for node in self.samantic_garden.nodes.values()])
                s, k = self.learning_actuator.execute(f_learn, return_keywords=True)
                if s:
                    keywords_after = set().union(*[node.keywords for node in self.samantic_garden.nodes.values()])
                    reward = self._compute_learning_reward(keywords_before, keywords_after, f_learn)
                    self.needs.satisfy_need("hunger", 0.8)
                    keywords = k
                    success = True
                else:
                    reward, success = -0.3, False
            else:
                reward, success = -0.3, False

        elif action in ["GEOLOGY_EXPLORE", "GEOLOGY_LEARN"]:
            success_geo, kw = self.geology_actuator.execute(action)
            if success_geo and kw:
                keywords_before = set().union(*[node.keywords for node in self.samantic_garden.nodes.values()])
                keywords_after = set().union(*[node.keywords for node in self.samantic_garden.nodes.values()])
                reward = self._compute_learning_reward(keywords_before, keywords_after, "geology_ingestion")
                keywords = kw
                success = True
            else:
                reward = 0.8 if success_geo else -0.1
                success = success_geo

        elif action == "IMAGINE":
            imagination = self.ecosystem.get_module("imagination")
            if imagination:
                if self.short_term_memory:
                    concepts = [n.label for n in self.short_term_memory[:3]]
                    scenario = f"Exploring the connections between: {', '.join(concepts)}"
                else:
                    scenario = "Random exploration of the latent space"
                try:
                    simulation_result = imagination.simulate(scenario)
                    reward, success = 0.7, True
                    print(f"✨ IMAGINE: Simulated '{scenario}' with avg risk {simulation_result.risk_assessment.get('rata_rata', 0):.2f}")
                except Exception as e:
                    print(f"⚠️ IMAGINE gagal: {e}")
                    reward, success = -0.2, False
            else:
                reward, success = -0.1, False

        elif action == "CONTEMPLATE":
            reward, success, keywords = self._contemplate()

        else:
            # Default fallback
            reward, success = 0.1, True

        if success:
            if keywords:
                nodes = [n.id for n in self.samantic_garden.nodes.values()
                         if not set(n.keywords).isdisjoint(keywords)]
                if nodes:
                    self.short_term_memory = self.samantic_garden.spreading_activation(
                        nodes, 1.0, 5
                    )
            if self.current_task and action == self.current_task["action"]:
                self.task_manager.complete_current_task()
                self.current_task = None

        self.cumulative_reward += reward
        self.update_learning_systems(action, success, reward)

        if self.current_plan:
            if success:
                self.current_plan.advance()
            else:
                self.current_plan = None

    # -----------------------------------------------------------------
    # CONTEMPLATE
    # -----------------------------------------------------------------
    def _contemplate(self) -> Tuple[float, bool, List[str]]:
        print("🧘 CONTEMPLATING: Starting deep reflective reasoning...")

        def _sanitize_term(raw: str) -> str:
            import re
            sanitized = re.sub(r'\W+', '_', raw)
            sanitized = sanitized.strip('_')
            return sanitized if sanitized else "unknown"

        self.reasoning_engine = ReasoningEngine(search_strategy="depth_first")

        for node in self.samantic_garden.nodes.values():
            for syn in node.synapses:
                if syn.strength > 0.7:
                    src = _sanitize_term(node.label)
                    dst = _sanitize_term(syn.target.label)
                    fact_str = f"related({src}, {dst})"
                    try:
                        self.reasoning_engine.add_fact(fact_str)
                    except Exception as e:
                        print(f"    ⚠️ Gagal menambah fakta: {fact_str} -> {e}")

        if len(self.short_term_memory) >= 2:
            node_a, node_b = self.short_term_memory[0], self.short_term_memory[1]
        else:
            all_nodes = list(self.samantic_garden.nodes.values())
            if len(all_nodes) >= 2:
                node_a, node_b = all_nodes[0], all_nodes[1]
            else:
                print("    ⚠️ Not enough concepts to contemplate.")
                return -0.2, False, []

        label_a = _sanitize_term(node_a.label)
        label_b = _sanitize_term(node_b.label)
        query = f"related({label_a}, {label_b})"

        print(f"    🔍 Query: {query}")
        try:
            proven = self.reasoning_engine.reason(query)
        except Exception as e:
            print(f"    ❌ Gagal mem‑parse query: {e}")
            proven = False

        if proven:
            print(f"    ✅ Proved: {label_a} is related to {label_b}")
            syn_a = node_a.get_synapse_to(node_b.id)
            syn_b = node_b.get_synapse_to(node_a.id)
            if syn_a:
                syn_a.strength = min(1.0, syn_a.strength + 0.1)
            if syn_b:
                syn_b.strength = min(1.0, syn_b.strength + 0.1)
            reward = 0.9
        else:
            print(f"    ❌ Could not prove relation.")
            reward = 0.2

        imagination = self.ecosystem.get_module("imagination")
        if imagination:
            scenario_text = f"If {node_a.label} and {node_b.label} are directly connected"
            try:
                sim_result = imagination.simulate(
                    scenario_text,
                    parameters={"node_a": node_a.label, "node_b": node_b.label},
                )
                avg_score = sim_result.risk_assessment.get("rata_rata", 0)
                reward += 0.2 if avg_score > 0 else -0.1
                print(f"    🌌 Imagination: {sim_result.summary()[:120]}...")
            except Exception as e:
                print(f"    ⚠️ Imagination gagal: {e}")
                reward -= 0.1

        keywords = list(node_a.keywords.union(node_b.keywords))
        return reward, True, keywords

    # -----------------------------------------------------------------
    # Learning & Neuromodulation update
    # -----------------------------------------------------------------
    def update_learning_systems(self, action: str, success: bool, reward: float):
        if action not in self.ltm_action_success:
            self.ltm_action_success[action] = {"success": 0, "total": 0}

        self.ltm_action_success[action]["total"] += 1
        if success:
            self.ltm_action_success[action]["success"] += 1

        neuromodulators = self.ecosystem.get_module("neuromodulators")
        deltas = NeuromodulatoryEvent.reward_prediction_error(reward, 0.1)
        neuromodulators.update_all(deltas=deltas)

        levels = neuromodulators.get_all_levels()
        arousal = (levels["Dopamine"] + levels["Noradrenaline"]) / 2.0
        self.samantic_garden.global_learning_rate = 0.01 * (1 + 1.5 * arousal)

        ltm_rates = {
            a: (s["success"] / s["total"] if s["total"] > 0 else 0.5)
            for a, s in self.ltm_action_success.items()
        }
        update_action_value(
            action=action,
            needs=self.needs.get_all_needs(),
            neuromodulators=self.ecosystem.get_module("neuromodulators").get_all_levels(),
            recalled_concepts_context=[n.label for n in self.short_term_memory],
            ltm_success_rates=ltm_rates,
            reward=reward,
        )

    # -----------------------------------------------------------------
    # Main cognitive cycle
    # -----------------------------------------------------------------
    def run_cycle(self):
        print("\n--- CYCLE START ---")
        self.cycle_count += 1
        print(f"🔄 Cycle: {self.cycle_count}")
        try:
            self.needs.update_needs()

            neuromodulators = self.ecosystem.get_module("neuromodulators")
            neuromodulators.apply_physical_condition(self.needs.get_all_needs())

            if self.cycle_count % 10 == 0:
                self._ingest_data_files()
            if self.task_manager.is_project_complete():
                print("🎉 Project complete!")

            print(self.task_manager.get_project_status())

            if not self.current_plan and self.cycle_count % 5 == 0:
                suggestion = self.metacognition.review_performance(self.needs.get_all_needs())
                if suggestion:
                    new_plan = self.planner.create_plan(suggestion)
                    if new_plan and evaluate_plan(
                        new_plan,
                        self.ecosystem.get_module("neuromodulators").get_all_levels()
                    ):
                        self.current_plan = new_plan
                        print(f"📋 New plan adopted: {new_plan}")

            selected_action = self.determine_action()
            self.execute_action(selected_action)
            self.ecosystem.update_modules(self.cycle_count)

        except Exception as e:
            print(f"💥💥 CRITICAL UNHANDLED EXCEPTION IN CYCLE {self.cycle_count} 💥💥")
            print(f"Error: {e}")
            self.update_learning_systems("CRITICAL_FAILURE", False, -2.0)
            self.needs.increase_need("cognitive_load", 0.8)
            self.current_plan = None

        print("--- CYCLE END ---")