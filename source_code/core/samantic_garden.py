"""
Samantic Garden - A Connectome-Inspired Knowledge Graph (v3.0 - Neural Backbone)

This module implements a dynamic knowledge graph where each node is an active,
learning agent. It fuses the concepts of a knowledge graph with a neural ecosystem,
where each concept (NeuralNode) houses a population of competing sub-neural-networks.

Core Concepts:
- NeuralNode: A replacement for KnowledgeNode. Each node is a self-contained ecosystem
  of AutonomousANNs ("sub-minds").
- Internal Competition: When a node is stimulated, its sub-minds compete. The one
  with the lowest prediction error (highest curiosity match) wins and dictates the
  node's output. This allows a single node to represent multiple contexts of a concept.
- Internal Evolution: During memory consolidation, a process of natural selection occurs
  *within* each NeuralNode. High-performing sub-minds are cross-bred, and poor
  performers are eliminated.
"""

import time
import numpy as np
import random
import copy
from typing import List, Optional, Tuple, Dict, Set
import os
from datetime import datetime

# --- Local Dependencies ---
# Each node now contains a population of neural networks
from .neural_ecosystem import AutonomousANN

# --- Visualization Imports (with fallback) ---
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    print("⚠️ Matplotlib or NetworkX not found. Visualization is disabled.")


# --- Core Building Blocks: Synapse and the new NeuralNode ---

class Synapse:
    """
    Represents a directional, plastic connection between two NeuralNodes.
    Uses an eligibility trace for delayed, reward-modulated Hebbian learning.
    """
    def __init__(self, target_node: 'NeuralNode', initial_strength: float = 0.1):
        self.target = target_node
        self.strength = np.clip(initial_strength, 0.0, 1.0)
        self.trace = 0.0  # Eligibility trace
        self.trace_decay = 0.95

    def hebbian_update(self, pre_activation: float, post_activation: float, learning_rate: float):
        """
        Updates the eligibility trace based on pre- and post-synaptic activity.
        "Neurons that fire together, wire together."
        """
        self.trace = (self.trace * self.trace_decay) + (pre_activation * post_activation)
        delta = learning_rate * self.trace
        self.strength = np.clip(self.strength + delta, 0.01, 1.0)

    def decay(self, decay_factor: float):
        """Applies a slow, homeostatic decay to prevent saturation."""
        self.strength *= (1.0 - decay_factor)
        self.trace *= self.trace_decay * 0.5


class NeuralNode:
    """
    Represents a single concept as a competitive ecosystem of neural networks (sub-minds).
    """
    def __init__(self, concept_vector: np.ndarray, label: str, source: str, keywords: List[str] = [],
                 population_size: int = 4, ann_topology: List[int] = None):
        self.id = f"{label.replace(' ', '_')}_{int(time.time()*1000)}"
        self.vector = concept_vector
        self.label = label
        self.source = source
        self.keywords = set(keywords)
        self.synapses: List[Synapse] = []
        self.created_at = time.time()
        self.importance = 0.1
        self.output_activation = 0.0 # The final "firing" strength of the node for the current cycle

        # --- Neural Backbone ---
        self.population_size = population_size
        self.ann_topology = ann_topology or [self.vector.shape[0] + 1, 16, 1] # Input: vector + signal, Output: activation
        
        # Create the initial population of sub-minds
        self.sub_minds: List[AutonomousANN] = [
            AutonomousANN(topology=self.ann_topology, activation_types=['gelu', 'tanh'])
            for _ in range(population_size)
        ]
        self.active_sub_mind_index = 0
        self.last_prediction_error = 0.0

    def reset_state(self):
        """Resets the node's activation and sub-mind states for a new cycle."""
        self.output_activation = 0.0
        # We don't reset the sub-minds' internal states, as they are persistent learners

    def process_stimulus(self, total_input: float) -> float:
        """
        Processes an incoming signal. All sub-minds compete, and the winner's
        output becomes the node's activation.
        """
        if not self.sub_minds:
            return 0.0

        ann_inputs = np.concatenate((self.vector, [total_input])).tolist()
        outputs = []
        errors = []

        # 1. Parallel Processing & Competition
        for ann in self.sub_minds:
            # The ANN's output is interpreted as the node's potential activation
            output = ann.forward(ann_inputs)[0]
            outputs.append(output)
            
            # Use a simplified prediction error as a measure of "surprise" or "fit"
            # Here, we model it as the difference between the ANN's output and the input signal
            error = (output - total_input) ** 2
            errors.append(error)

        # 2. Winner-Takes-All
        winner_index = np.argmin(errors)
        self.active_sub_mind_index = winner_index
        self.output_activation = outputs[winner_index]
        self.last_prediction_error = errors[winner_index]
        
        # The winning sub-mind's output is the node's output for this cycle
        return self.output_activation

    def learn_from_interaction(self, hebbian_reward: float, learning_rate_modifier: float):
        """
        The winning sub-mind learns from the interaction.
        The reward is based on the Hebbian principle ("fire together, wire together").
        """
        if not self.sub_minds:
            return

        winner_ann = self.sub_minds[self.active_sub_mind_index]
        
        # Curiosity-driven learning: Higher prediction error increases learning rate
        curiosity_bonus = 1.0 + np.clip(self.last_prediction_error, 0, 5)
        
        # The ANN "evolves" based on the Hebbian reward from the graph interaction
        # We simulate a simple state transition for the ANN's internal evolution
        dummy_state = winner_ann.prev_state or np.zeros(self.ann_topology[0])
        dummy_action = [self.output_activation]
        
        winner_ann.evolve(
            immediate_reward=hebbian_reward,
            state=dummy_state.tolist(),
            action=dummy_action,
            next_state=dummy_state.tolist() # Simplified for this context
        )
        # Adjust learning rate based on graph-level arousal and local curiosity
        winner_ann.learning_rate = np.clip(winner_ann.learning_rate * learning_rate_modifier * curiosity_bonus, 0.001, 0.1)


    def evolve_population(self):
        """
        Performs natural selection on the sub-mind population.
        Called during memory consolidation. Borrows logic from the Ecosystem class.
        """
        if len(self.sub_minds) <= 2:
            return # Not enough diversity to evolve

        # 1. Evaluate and Sort
        # Fitness is the inverse of prediction error (a simple proxy) or discounted_fitness from ANN
        self.sub_minds.sort(key=lambda ann: ann.discounted_fitness, reverse=True)

        # 2. Selection: Keep the top half
        half_point = max(1, len(self.sub_minds) // 2)
        survivors = self.sub_minds[:half_point]
        
        # 3. Crossover and Mutation
        new_generation = []
        while len(survivors) + len(new_generation) < self.population_size:
            # Select two parents from the survivors
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            
            # Create a child
            child = self._crossover(parent1, parent2)
            
            # Apply mutation to the child
            for l, W in enumerate(child.weights):
                 mask = np.random.random(W.shape) < child.mutation_rate
                 if np.any(mask):
                     noise = np.random.randn(*W.shape) * 0.1 # Stronger initial mutation
                     child.weights[l] += mask * noise
            
            new_generation.append(child)

        self.sub_minds = survivors + new_generation

    def _crossover(self, p1: AutonomousANN, p2: AutonomousANN) -> AutonomousANN:
        """Creates a new ANN by crossing over the weights of two parents."""
        child = AutonomousANN(
            topology=p1.topology,
            activation_types=p1.activation_types,
            mutation_rate=(p1.mutation_rate + p2.mutation_rate) / 2,
            discount_factor=(p1.discount_factor + p2.discount_factor) / 2,
        )
        # Weight crossover
        for i in range(len(child.weights)):
            mask = np.random.rand(*p1.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, p1.weights[i], p2.weights[i])
        return child
        
    def add_synapse(self, target_node: 'NeuralNode', strength: float):
        if not self.get_synapse_to(target_node.id):
            self.synapses.append(Synapse(target_node, strength))

    def get_synapse_to(self, target_id: str) -> Optional[Synapse]:
        for synapse in self.synapses:
            if synapse.target.id == target_id:
                return synapse
        return None


# --- The Connectome Manager: SamanticGarden ---

class SamanticGarden:
    """
    Manages the entire connectome of NeuralNodes, simulating learning, recall, and evolution.
    """
    def __init__(self, log_dir="Samre/log", persistence_file=None):
        self.nodes: Dict[str, NeuralNode] = {}
        self.log_dir = log_dir
        self.persistence_file = persistence_file
        self.global_learning_rate = 0.01 # Modulated by cognitive state (arousal)
        self.config = {
            "ingestion_reinforcement_threshold": 0.9,
            "connection_similarity_threshold": 0.75,
            "consolidation_pruning_threshold": 0.02,
            "consolidation_decay_factor": 0.005,
            "abstraction_cos_threshold": 0.8,
            "neural_node_population": 4, # Default sub-mind population
        }
        if VISUALIZATION_ENABLED and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.persistence_file:
            self.load_state()

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        return dot / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0.0

    def _find_most_similar_node(self, vector: np.ndarray) -> Optional[Tuple[NeuralNode, float]]:
        if not self.nodes: return None
        node_list = list(self.nodes.values())
        similarities = [self._calculate_similarity(vector, node.vector) for node in node_list]
        best_match_idx = np.argmax(similarities)
        return node_list[best_match_idx], similarities[best_match_idx]

    def ingest_knowledge(self, concept_vector: List[float], label: str, source: str, keywords: List[str]):
        """Ingests a new piece of information, creating or reinforcing nodes."""
        vector_np = np.array(concept_vector)
        best_match, similarity = self._find_most_similar_node(vector_np) or (None, 0)
        
        if similarity > self.config["ingestion_reinforcement_threshold"]:
            node_to_focus = best_match
            node_to_focus.vector = (node_to_focus.vector * 0.9 + vector_np * 0.1)
            node_to_focus.keywords.update(keywords)
            node_to_focus.importance = min(1.0, node_to_focus.importance + 0.1)
            print(f"🧠 Knowledge Reinforced: Node '{node_to_focus.label}' strengthened.")
        else:
            node_to_focus = NeuralNode(
                vector_np, label, source, keywords,
                population_size=self.config["neural_node_population"]
            )
            self.nodes[node_to_focus.id] = node_to_focus
            print(f"🌱 New Knowledge Born: Node '{label}' created with {len(node_to_focus.sub_minds)} sub-minds.")

        for existing_node in self.nodes.values():
            if existing_node == node_to_focus: continue
            cos_sim = self._calculate_similarity(node_to_focus.vector, existing_node.vector)
            if cos_sim > self.config["connection_similarity_threshold"]:
                node_to_focus.add_synapse(existing_node, strength=cos_sim)
                existing_node.add_synapse(node_to_focus, strength=cos_sim)
                print(f"🔗 Synapse Formed: '{node_to_focus.label}' <-> '{existing_node.label}' (sim: {cos_sim:.2f})")
        
        self.spreading_activation(stimulus_node_ids=[node_to_focus.id], initial_signal=0.7, depth=3)


    def spreading_activation(self, stimulus_node_ids: List[str], initial_signal: float, depth: int = 5) -> List[NeuralNode]:
        """
        Simulates recall. Activation spreads through the connectome, and each node's
        internal neural networks process signals and learn.
        """
        print(f"⚡️ Spreading Activation started from {len(stimulus_node_ids)} nodes...")
        for node in self.nodes.values():
            node.reset_state()
            for synapse in node.synapses:
                synapse.trace = 0.0

        active_nodes_this_cycle = {nid: self.nodes[nid] for nid in stimulus_node_ids if nid in self.nodes}
        for node in active_nodes_this_cycle.values():
            node.output_activation = initial_signal # Directly set initial activation for stimulus

        for cycle in range(depth):
            pre_activations = {nid: n.output_activation for nid, n in self.nodes.items()}
            inputs_this_cycle = {nid: 0.0 for nid in self.nodes}

            # 1. Calculate all signals being sent based on the PREVIOUS cycle's output
            for node in self.nodes.values():
                if pre_activations.get(node.id, 0.0) > 0: # If node was active
                    for synapse in node.synapses:
                        signal = pre_activations[node.id] * synapse.strength
                        inputs_this_cycle[synapse.target.id] += signal
            
            # 2. Update all node activations based on the collected inputs
            # Each node internally decides its new activation via its sub-minds
            post_activations = {}
            for node_id, total_input in inputs_this_cycle.items():
                node = self.nodes[node_id]
                new_activation = node.process_stimulus(total_input)
                post_activations[node_id] = new_activation

            # 3. Perform Hebbian updates for both synapses and internal ANNs
            for node in self.nodes.values():
                total_hebbian_reward = 0.0
                for synapse in node.synapses:
                    pre_act = pre_activations.get(node.id, 0.0)
                    post_act = post_activations.get(synapse.target.id, 0.0)
                    
                    if pre_act > 0 and post_act > 0:
                        # Update the synapse
                        synapse.hebbian_update(pre_act, post_act, self.global_learning_rate)
                        
                        # Accumulate reward for the source node's winning ANN
                        total_hebbian_reward += pre_act * post_act
                
                # Each node learns once per cycle based on its total synaptic success
                if total_hebbian_reward > 0:
                    node.learn_from_interaction(total_hebbian_reward, self.global_learning_rate)
        
        print("⚡️ Spreading Activation complete.")
        return self.get_top_activated_nodes()

    def consolidate_memories(self):
        """
        Simulates a sleep cycle: prunes weak connections, decays synapses,
        replays important memories, and triggers internal evolution within each node.
        """
        print("\n--- 🌙 Initiating Memory Consolidation (Sleep Cycle) ---")
        pruned_count = 0
        total_synapses = sum(len(n.synapses) for n in self.nodes.values())

        # 1. Pruning and homeostatic decay
        for node in self.nodes.values():
            original_syn_count = len(node.synapses)
            node.synapses[:] = [s for s in node.synapses if s.strength > self.config["consolidation_pruning_threshold"]]
            pruned_count += original_syn_count - len(node.synapses)
            for synapse in node.synapses:
                synapse.decay(self.config["consolidation_decay_factor"])

        print(f" pruned synapses: {pruned_count} out of {total_synapses}.")
        
        # 2. Memory Replay
        important_nodes = [n for n in self.nodes.values() if n.importance > 0.5]
        if important_nodes:
            print(f"Replaying {len(important_nodes)} important memories...")
            replay_nodes_ids = [n.id for n in np.random.choice(important_nodes, size=min(len(important_nodes), 5), replace=False)]
            self.spreading_activation(stimulus_node_ids=replay_nodes_ids, initial_signal=0.8, depth=4)

        # 3. Internal Evolution within each Node
        print("🧬 Triggering internal evolution for all concepts...")
        for node in self.nodes.values():
            node.evolve_population()

        # 4. Create new abstractions from co-active nodes
        self.create_abstract_nodes()

        if VISUALIZATION_ENABLED:
            self.visualize_and_save_graph()
        print("--- ☀️ Memory Consolidation Complete ---\n")

    def create_abstract_nodes(self):
        """
        Creates a new "interneuron" or abstract concept node to represent the
        relationship between two strongly, bidirectionally connected nodes.
        """
        print("🤔 Searching for new abstractions...")
        created_count = 0
        nodes_list = list(self.nodes.values())
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                node_a = nodes_list[i]
                node_b = nodes_list[j]
                
                # Check for strong bidirectional connection
                syn_a_b = node_a.get_synapse_to(node_b.id)
                syn_b_a = node_b.get_synapse_to(node_a.id)
                
                if syn_a_b and syn_b_a:
                    combined_strength = (syn_a_b.strength + syn_b_a.strength) / 2
                    if combined_strength > self.config["abstraction_cos_threshold"]:
                        # Create abstract label
                        abstract_label = f"Abstract({node_a.label} & {node_b.label})"
                        
                        # Check if already exists
                        if any(n.label == abstract_label for n in self.nodes.values()):
                            continue
                            
                        # Create abstract vector (mean)
                        abstract_vector = (node_a.vector + node_b.vector) / 2
                        
                        # Create new NeuralNode
                        new_node = NeuralNode(
                            abstract_vector, abstract_label, "AbstractionEngine",
                            keywords=list(node_a.keywords.union(node_b.keywords)),
                            population_size=self.config["neural_node_population"]
                        )
                        self.nodes[new_node.id] = new_node
                        
                        # Connect abstraction to its components
                        new_node.add_synapse(node_a, 0.8)
                        new_node.add_synapse(node_b, 0.8)
                        node_a.add_synapse(new_node, 0.5)
                        node_b.add_synapse(new_node, 0.5)
                        
                        created_count += 1
                        
        if created_count > 0:
            print(f"✨ Created {created_count} new abstract concepts.")


    def get_top_activated_nodes(self, k: int = 5) -> List[NeuralNode]:
        """Returns the top k most activated nodes after a recall cycle."""
        if not self.nodes: return []
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.output_activation, reverse=True)
        return sorted_nodes[:k]

    def visualize_and_save_graph(self):
        """Creates and saves a visual representation of the knowledge graph."""
        if not VISUALIZATION_ENABLED or not self.nodes: return

        G = nx.DiGraph()
        node_labels = {n.id: f"{n.label[:15]}\\n{n.output_activation:.2f}\\n(Pop: {len(n.sub_minds)})" for n in self.nodes.values()}
        node_sizes = [1000 + (n.importance * 4000) for n in self.nodes.values()]
        node_colors = [n.output_activation for n in self.nodes.values()]

        for node in self.nodes.values():
            G.add_node(node.id)

        edge_widths = [s.strength * 5 for n in self.nodes.values() for s in n.synapses]
        edge_colors = [s.trace for n in self.nodes.values() for s in n.synapses]
        edges = [(n.id, s.target.id) for n in self.nodes.values() for s in n.synapses]
        G.add_edges_from(edges)
        
        print("📊 Generating graph visualization...")
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=1.5 / np.sqrt(len(self.nodes) or 1), iterations=70)

        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cividis, alpha=0.6, arrows=True, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

        if node_colors:
            plt.colorbar(nodes, label="Activation Level")

        plt.title("Samre's Samantic Garden (Neural Backbone)", fontsize=24)
        plt.axis('off')
        filename = os.path.join(self.log_dir, f"samantic_garden_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        try:
            plt.savefig(filename)
            print(f"✅ Graph visualization saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")
        plt.close()

    def get_garden_state(self) -> dict:
        num_synapses = sum(len(n.synapses) for n in self.nodes.values())
        return {
            "jumlah_node": len(self.nodes),
            "jumlah_sinapsis": num_synapses,
            "rata-rata_kekuatan_sinapsis": np.mean([s.strength for n in self.nodes.values() for s in n.synapses if num_synapses > 0]) if num_synapses > 0 else 0,
            "rata-rata_aktivasi": np.mean([n.output_activation for n in self.nodes.values()]) if self.nodes else 0
        }

    def save_state(self):
        """Saves the garden state to a JSON file."""
        if not self.persistence_file:
            return
        print(f"💾 Saving SamanticGarden to {self.persistence_file}...")
        data = {
            "nodes": {}
        }
        for node_id, node in self.nodes.items():
            data["nodes"][node_id] = {
                "label": node.label,
                "vector": node.vector.tolist(),
                "source": node.source,
                "keywords": list(node.keywords),
                "importance": node.importance,
                "synapses": [{"target": s.target.id, "strength": s.strength} for s in node.synapses]
            }
        try:
            import json
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=4)
            print("✅ SamanticGarden saved.")
        except Exception as e:
            print(f"❌ Error saving SamanticGarden: {e}")

    def load_state(self):
        """Loads the garden state from a JSON file."""
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return
        print(f"📂 Loading SamanticGarden from {self.persistence_file}...")
        try:
            import json
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
            
            # First pass: create nodes
            for node_id, node_data in data.get("nodes", {}).items():
                node = NeuralNode(
                    np.array(node_data["vector"]),
                    node_data["label"],
                    node_data["source"],
                    node_data["keywords"],
                    population_size=self.config["neural_node_population"]
                )
                node.id = node_id
                node.importance = node_data["importance"]
                self.nodes[node_id] = node
            
            # Second pass: create synapses
            for node_id, node_data in data.get("nodes", {}).items():
                node = self.nodes[node_id]
                for syn_data in node_data.get("synapses", []):
                    target_id = syn_data["target"]
                    if target_id in self.nodes:
                        node.add_synapse(self.nodes[target_id], syn_data["strength"])
            
            print(f"✅ Loaded {len(self.nodes)} nodes.")
        except Exception as e:
            print(f"❌ Error loading SamanticGarden: {e}")

