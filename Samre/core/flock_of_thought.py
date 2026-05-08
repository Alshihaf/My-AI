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
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import deque
import json

# Core Components
from .cognitive_core import CognitiveEngine
from .imagination import Imagination
from .neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from .needs import InternalNeeds
from .sws_logic import score_all_actions, POSSIBLE_ACTIONS
from .executive import evaluate_action, evaluate_plan
from .plan import Plan
from .planner import Planner
from .metacognition import Metacognition
from .samantic_garden import SamanticGarden, NeuralNode
from .neural_ecosystem import NeuralEcosystem
from .chain_of_thought import ReasoningEngine      # <-- now directly used

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
        self.short_term_memory: List[NeuralNode] = []
        self.metacognition = Metacognition(self.ltm_action_success)
        self.planner = Planner()
        # Reasoning engine sebagai pengganti ChainOfThought untuk penalaran mendalam
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
        self.learning_queue: deque[str] = deque()
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        if not self.samantic_garden.nodes:
            self._bootstrap_initial_knowledge()
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

    def save_state(self):
        print("💾 Saving agent state...")
        self.samantic_garden.save_state()
        try:
            with open(self.ltm_persistence_file, 'w') as f:
                json.dump(self.ltm_action_success, f, indent=4)
            print(f"✅ LTM saved to {self.ltm_persistence_file}")
        except IOError as e:
            print(f"❌ Error saving LTM: {e}")
        print("✅ Agent state saving complete.")

    # -----------------------------------------------------------------
    # Bootstrapping
    # -----------------------------------------------------------------
    def _bootstrap_initial_knowledge(self):
        print("--- Bootstrapping ---")
        source_files = ["core/flock_of_thought.py", "core/cognitive_core.py"]
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
            self.process_and_store(content, file_path)  # return value ignored here
        self.samantic_garden.consolidate_memories()
        print("--- Bootstrap Complete ---")

    # -----------------------------------------------------------------
    # Knowledge ingestion
    # -----------------------------------------------------------------
    def process_and_store(self, content: str, source_path: str) -> List[str]:
        """
        Processes raw text content into vectors and stores it in the Samantic Garden.
        Returns the list of extracted keywords.
        """
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
        self.needs.update_needs()
        stm_labels = [n.label for n in self.short_term_memory]
        ltm_rates = {
            a: (s["success"] / s["total"] if s["total"] > 0 else 0.5)
            for a, s in self.ltm_action_success.items()
        }
        neuro_levels = self.ecosystem.get_module("neuromodulators").get_all_levels()
        # FIX: jangan ubah ke lowercase, sws_logic butuh kapital
        action_scores = score_all_actions(
            self.needs.get_all_needs(),
            neuro_levels,
            stm_labels,
            ltm_rates
        )
        return self.evaluate_and_select_action(action_scores)

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        neuro_levels = self.ecosystem.get_module("neuromodulators").get_all_levels()
        for action, score in sorted_actions:
            if evaluate_action(action, score, self.needs.get_all_needs(), neuro_levels):
                return action
        return "REST"

    # -----------------------------------------------------------------
    # Action execution
    # -----------------------------------------------------------------
    def execute_action(self, action: str):
        reward, success, keywords = 0.0, False, []
        if action == "CONSOLIDATE":
           # Jalankan konsolidasi sesungguhnya
           self.samantic_garden.consolidate_memories()
           # Kurangi beban kognitif secara signifikan
           self.needs.satisfy_need("cognitive_load", 0.8)
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
                s, k = self.learning_actuator.execute(f_learn, return_keywords=True)
                if s:
                    self.needs.satisfy_need("hunger", 0.8)
                    keywords = k
                    reward, success = 0.9, True
            else:
                reward, success = -0.3, False
        elif action in ["GEOLOGY_EXPLORE", "GEOLOGY_LEARN"]:
            success = self.geology_actuator.execute(action)
            reward = 0.8 if success else -0.1
        elif action == "IMAGINE":
            imagination = self.ecosystem.get_module("imagination")
            if imagination:
                context = [n.vector.tolist() for n in self.short_term_memory]
                if not context:
                    context = [np.random.randn(self.text_processor.vector_dim).tolist()]
                simulation_result = imagination.simulate(context[0])
                reward, success = 0.7, True
                print(f"✨ IMAGINE: Simulated scenario with outcome score {simulation_result.get('outcome', 0):.2f}")
            else:
                reward, success = -0.1, False
        elif action == "CONTEMPLATE":
            # Aksi berpikir reflektif baru
            reward, success, keywords = self._contemplate()
        else:
            # Default fallback (termasuk aksi seperti ORGANIZE, EVOLVE, dll yang belum spesifik)
            reward, success = 0.1, True

        if success:
            if keywords:
                # Aktifkan kembali node yang relevan di memori jangka pendek
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
    # CONTEMPLATE: Deep reflective reasoning
    # -----------------------------------------------------------------
    def _contemplate(self) -> Tuple[float, bool, List[str]]:
        """
        Deep reflection using the reasoning engine, file manager, and imagination.
        1. Load symbolic facts from strong synapses in the SamanticGarden.
        2. Pick two highly activated concepts from short‑term memory.
        3. Use the reasoning engine to prove if a transitive relationship exists.
        4. Simulate the outcome with Imagination.
        """
        print("🧘 CONTEMPLATING: Starting deep reflective reasoning...")
        # Reset mesin penalaran setiap kali untuk kondisi bersih
        self.reasoning_engine = ReasoningEngine(search_strategy="depth_first")
        # 1. Bangun basis pengetahuan simbolik dari sinapsis kuat
        for node in self.samantic_garden.nodes.values():
            for syn in node.synapses:
                if syn.strength > 0.7:
                    # Hindari karakter yang tidak valid dalam term
                    src = node.label.replace(' ', '_').replace("'", "")
                    dst = syn.target.label.replace(' ', '_').replace("'", "")
                    fact_str = f"related({src}, {dst})"
                    try:
                        self.reasoning_engine.add_fact(fact_str)
                    except Exception:
                        continue

        # 2. Pilih dua konsep yang sedang aktif di STM
        if len(self.short_term_memory) >= 2:
            node_a, node_b = self.short_term_memory[0], self.short_term_memory[1]
        else:
            all_nodes = list(self.samantic_garden.nodes.values())
            if len(all_nodes) >= 2:
                node_a, node_b = all_nodes[0], all_nodes[1]
            else:
                print("    ⚠️ Not enough concepts to contemplate.")
                return -0.2, False, []

        label_a = node_a.label.replace(' ', '_')
        label_b = node_b.label.replace(' ', '_')
        query = f"related({label_a}, {label_b})"

        # 3. Buktikan dengan backward chaining
        print(f"    🔍 Query: {query}")
        proven = self.reasoning_engine.reason(query)
        if proven:
            print(f"    ✅ Proved: {label_a} is related to {label_b}")
            # Perkuat koneksi di garden
            syn_a = node_a.get_synapse_to(node_b.id)
            syn_b = node_b.get_synapse_to(node_a.id)
            if syn_a:
                syn_a.strength = min(1.0, syn_a.strength + 0.1)
            if syn_b:
                syn_b.strength = min(1.0, syn_b.strength + 0.1)
            reward = 0.9
        else:
            print(f"    ❌ Could not prove relation.")
            reward = 0.2  # reward kecil karena upaya berpikir

        # 4. Simulasi dampak dengan Imagination
        imagination = self.ecosystem.get_module("imagination")
        if imagination:
            scenario = f"If {node_a.label} and {node_b.label} are directly connected"
            sim_result = imagination.simulate(scenario, parameters={
                "node_a": node_a.label,
                "node_b": node_b.label
            })
            avg_score = sim_result.risk_assessment.get("rata_rata", 0)
            reward += 0.2 if avg_score > 0 else -0.1
            print(f"    🌌 Imagination: {sim_result.summary()[:120]}...")

        keywords = list(node_a.keywords.union(node_b.keywords))
        return reward, True, keywords

    # -----------------------------------------------------------------
    # Learning & Neuromodulation update
    # -----------------------------------------------------------------
    def update_learning_systems(self, action: str, success: bool, reward: float):
        """Perbarui statistik LTM, neuromodulator, dan laju belajar."""
        self.ltm_action_success[action]["total"] += 1
        if success:
            self.ltm_action_success[action]["success"] += 1

        neuromodulators = self.ecosystem.get_module("neuromodulators")
        deltas = NeuromodulatoryEvent.reward_prediction_error(reward, 0.1)
        neuromodulators.update_all(deltas=deltas)

        levels = neuromodulators.get_all_levels()
        arousal = (levels["Dopamine"] + levels["Noradrenaline"]) / 2.0
        self.samantic_garden.global_learning_rate = 0.01 * (1 + 1.5 * arousal)

    # -----------------------------------------------------------------
    # Main cognitive cycle
    # -----------------------------------------------------------------
    def run_cycle(self):
        """Satu siklus kognitif lengkap dengan error handling."""
        print("\n--- CYCLE START ---")
        self.cycle_count += 1
        print(f"🔄 Cycle: {self.cycle_count}")
        try:
            self.needs.update_needs()
            if self.task_manager.is_project_complete():
                print("🎉 Project complete!")
                return

            print(self.task_manager.get_project_status())

            # Metacognition & Planning (tidak lagi menggunakan ChainOfThought)
            if not self.current_plan and self.cycle_count % 5 == 0:
                suggestion = self.metacognition.review_performance()
                if suggestion:
                    # Planner langsung membuat Plan dari saran
                    new_plan = self.planner.create_plan(suggestion)
                    if new_plan and evaluate_plan(
                        new_plan,
                        self.ecosystem.get_module("neuromodulators").get_all_levels()
                    ):
                        self.current_plan = new_plan
                        print(f"📋 New plan adopted: {new_plan}")

            # Pilih dan eksekusi aksi
            selected_action = self.determine_action()
            self.execute_action(selected_action)

            # Konsolidasi periodik
            if self.needs.get_need('cognitive_load') < 0.2:
                self.samantic_garden.consolidate_memories()

            self.ecosystem.update_modules(self.cycle_count)

        except Exception as e:
            print(f"💥💥 CRITICAL UNHANDLED EXCEPTION IN CYCLE {self.cycle_count} 💥💥")
            print(f"Error: {e}")
            self.update_learning_systems("CRITICAL_FAILURE", False, -2.0)
            self.needs.increase_need("cognitive_load", 0.8)
            self.current_plan = None

        print("--- CYCLE END ---")