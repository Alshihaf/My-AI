"""
Samantic Garden - A Connectome-Inspired Knowledge Graph (v2.0)

This module simulates a dynamic mind-map based on the principles of a neural
connectome. It features Hebbian learning, eligibility traces, spreading activation,
memory consolidation with replay, and automated abstraction.
"""

import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Set
import os
from datetime import datetime

# --- Visualization Imports (with fallback) ---
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    print("⚠️  Matplotlib or NetworkX not found. Visualization is disabled.")

# --- Core Building Blocks: Synapse and KnowledgeNode ---

class Synapse:
    """
    Represents a directional, plastic connection between two KnowledgeNodes.
    Uses an eligibility trace for delayed, reward-modulated Hebbian learning.
    """
    def __init__(self, target_node: 'KnowledgeNode', initial_strength: float = 0.1):
        self.target = target_node
        self.strength = np.clip(initial_strength, 0.0, 1.0)
        self.trace = 0.0  # Eligibility trace
        self.trace_decay = 0.95 # Slower decay for trace allows for longer credit assignment

    def hebbian_update(self, pre_activation: float, post_activation: float, learning_rate: float):
        """
        Updates the eligibility trace based on pre- and post-synaptic activity.
        The actual weight update is modulated by this trace.
        "Neurons that fire together, wire together."
        """
        # 1. Update the eligibility trace based on recent causal activity.
        self.trace = (self.trace * self.trace_decay) + (pre_activation * post_activation)
        
        # 2. Update the synaptic strength based on the trace.
        # This allows a delayed global reward signal to strengthen a whole chain of recent activity.
        delta = learning_rate * self.trace
        self.strength = np.clip(self.strength + delta, 0.01, 1.0)

    def decay(self, decay_factor: float):
        """Applies a slow, homeostatic decay to prevent saturation."""
        self.strength *= (1.0 - decay_factor)
        # Also decay the trace when not in use
        self.trace *= self.trace_decay * 0.5

class KnowledgeNode:
    """
    Represents a single concept or unit of knowledge, functioning like a neuron.
    """
    def __init__(self, concept_vector: np.ndarray, label: str, source: str, keywords: List[str] = []):
        self.id = f"{label.replace(' ', '_')}_{int(time.time()*1000)}"
        self.vector = concept_vector
        self.label = label
        self.source = source
        self.keywords = set(keywords)
        self.synapses: List[Synapse] = []
        self.activation_level = 0.0
        self.importance = 0.1
        self.created_at = time.time()
        self.firing_threshold = 0.4 # Node must reach this activation to fire
        self.activation_decay = 0.85 # How quickly activation fades per cycle

    def reset_activation(self):
        """Resets the node's activation to zero."""
        self.activation_level = 0.0

    def update_activation(self, total_input: float):
        """
        Updates the node's activation based on summed inputs and internal decay.
        A non-linear function (tanh) is used to keep activation bounded.
        """
        self.activation_level = np.tanh((self.activation_level * self.activation_decay) + total_input)

    def add_synapse(self, target_node: 'KnowledgeNode', strength: float):
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
    Manages the entire connectome, simulating learning, recall, and growth.
    """
    def __init__(self, log_dir="Samre/log"):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.log_dir = log_dir
        self.global_learning_rate = 0.01 # Modulated by cognitive state (arousal)
        self.config = {
            "ingestion_reinforcement_threshold": 0.9,
            "connection_similarity_threshold": 0.75,
            "consolidation_pruning_threshold": 0.02,
            "consolidation_decay_factor": 0.005,
            "abstraction_cos_threshold": 0.8,
        }
        if VISUALIZATION_ENABLED and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        return dot / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0.0

    def _find_most_similar_node(self, vector: np.ndarray) -> Optional[Tuple[KnowledgeNode, float]]:
        if not self.nodes: return None
        node_list = list(self.nodes.values())
        similarities = [self._calculate_similarity(vector, node.vector) for node in node_list]
        best_match_idx = np.argmax(similarities)
        return node_list[best_match_idx], similarities[best_match_idx]

    def ingest_knowledge(self, concept_vector: List[float], label: str, source: str, keywords: List[str]):
        """Ingests a new piece of information, creating or reinforcing nodes."""
        # (Logic is similar to v1 but uses the new classes)
        vector_np = np.array(concept_vector)
        best_match, similarity = self._find_most_similar_node(vector_np) or (None, 0)
        
        if similarity > self.config["ingestion_reinforcement_threshold"]:
            node_to_focus = best_match
            node_to_focus.vector = (node_to_focus.vector * 0.9 + vector_np * 0.1) # Reinforce vector
            node_to_focus.keywords.update(keywords)
            node_to_focus.importance = min(1.0, node_to_focus.importance + 0.1)
            print(f"🧠 Knowledge Reinforced: Node '{node_to_focus.label}' strengthened.")
        else:
            node_to_focus = KnowledgeNode(vector_np, label, source, keywords)
            self.nodes[node_to_focus.id] = node_to_focus
            print(f"🌱 New Knowledge Born: Node '{label}' created.")

        # Form synapses with other related nodes
        for existing_node in self.nodes.values():
            if existing_node == node_to_focus: continue
            cos_sim = self._calculate_similarity(node_to_focus.vector, existing_node.vector)
            if cos_sim > self.config["connection_similarity_threshold"]:
                node_to_focus.add_synapse(existing_node, strength=cos_sim)
                existing_node.add_synapse(node_to_focus, strength=cos_sim)
                print(f"🔗 Synapse Formed: '{node_to_focus.label}' <-> '{existing_node.label}' (sim: {cos_sim:.2f})")
        
        # Trigger a small, localized spreading activation to integrate the new knowledge
        self.spreading_activation(stimulus_node_ids=[node_to_focus.id], initial_signal=0.7, depth=3)


    def spreading_activation(self, stimulus_node_ids: List[str], initial_signal: float, depth: int = 5) -> List[KnowledgeNode]:
        """
        Simulates recall. Activates stimulus nodes and lets the activation spread
        through the connectome over several cycles, performing Hebbian learning.
        """
        print(f"⚡️ Spreading Activation started from {len(stimulus_node_ids)} nodes...")
        # 1. Reset all activations and traces for a clean slate
        for node in self.nodes.values():
            node.reset_activation()
            for synapse in node.synapses:
                synapse.trace = 0.0

        # 2. Apply the initial stimulus
        for node_id in stimulus_node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].activation_level = initial_signal

        # 3. Main propagation loop (over discrete time steps/cycles)
        for cycle in range(depth):
            pre_activations = {nid: n.activation_level for nid, n in self.nodes.items()}
            inputs_this_cycle = {nid: 0.0 for nid in self.nodes}

            # First, calculate all signals being sent in this cycle
            for node in self.nodes.values():
                if node.activation_level > node.firing_threshold:
                    for synapse in node.synapses:
                        signal = node.activation_level * synapse.strength
                        inputs_this_cycle[synapse.target.id] += signal
            
            # Second, update all node activations based on the collected inputs
            for node in self.nodes.values():
                node.update_activation(inputs_this_cycle[node.id])

            # Third, perform Hebbian updates for all synapses based on pre- and post-synaptic activity
            post_activations = {nid: n.activation_level for nid, n in self.nodes.items()}
            for node in self.nodes.values():
                for synapse in node.synapses:
                    pre_act = pre_activations[node.id]
                    post_act = post_activations[synapse.target.id]
                    if pre_act > node.firing_threshold and post_act > 0:
                        synapse.hebbian_update(pre_act, post_act, self.global_learning_rate)
        
        print("⚡️ Spreading Activation complete.")
        return self.get_top_activated_nodes()


    def consolidate_memories(self):
        """
        Simulates a sleep cycle: prunes weak connections, decays all synapses slightly,
        and replays important memories to strengthen their pathways.
        """
        print("\n--- 🌙 Initiating Memory Consolidation (Sleep Cycle) ---")
        pruned_count = 0
        total_synapses = sum(len(n.synapses) for n in self.nodes.values())

        # 1. Pruning and homeostatic decay
        for node in self.nodes.values():
            node.synapses[:] = [s for s in node.synapses if s.strength > self.config["consolidation_pruning_threshold"]]
            pruned_count += len(node.synapses) - len(node.synapses)
            for synapse in node.synapses:
                synapse.decay(self.config["consolidation_decay_factor"])

        print(f" pruned synapses: {pruned_count} out of {total_synapses}.")
        
        # 2. Memory Replay: activate important nodes and let them fire
        important_nodes = [n for n in self.nodes.values() if n.importance > 0.5]
        if important_nodes:
            print(f"Replaying {len(important_nodes)} important memories...")
            replay_nodes_ids = np.random.choice(important_nodes, size=min(len(important_nodes), 5), replace=False)
            self.spreading_activation(stimulus_node_ids=[n.id for n in replay_nodes_ids], initial_signal=0.8, depth=4)

        # 3. Create new abstractions from co-active nodes
        self.create_abstract_nodes()

        if VISUALIZATION_ENABLED:
            self.visualize_and_save_graph()
        print("--- ☀️ Memory Consolidation Complete ---\n")

    def create_abstract_nodes(self):
        """
        Looks for pairs of strongly, bidirectionally connected nodes and creates
        a new "interneuron" or abstract concept node to represent their relationship.
        """
        print("🤔 Searching for new abstractions...")
        created_count = 0
        nodes_list = list(self.nodes.values())
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                node_a, node_b = nodes_list[i], nodes_list[j]
                syn_ab = node_a.get_synapse_to(node_b.id)
                syn_ba = node_b.get_synapse_to(node_a.id)

                if syn_ab and syn_ba and syn_ab.strength > self.config["abstraction_cos_threshold"] and syn_ba.strength > self.config["abstraction_cos_threshold"]:
                    # Check if an abstraction already exists
                    # An existing abstraction would be strongly connected to both A and B
                    is_abstracted = False
                    for potential_interneuron in node_a.synapses:
                        inter_node = potential_interneuron.target
                        if inter_node.get_synapse_to(node_b.id):
                            is_abstracted = True
                            break
                    if is_abstracted: continue

                    # Create the new abstract node
                    new_vec = (node_a.vector + node_b.vector) / 2.0
                    new_label = f"Abstract({node_a.label[:10]},{node_b.label[:10]})"
                    new_node = KnowledgeNode(new_vec, new_label, source="abstraction", keywords=[node_a.label, node_b.label])
                    self.nodes[new_node.id] = new_node
                    
                    # Connect the new node to its parents
                    new_node.add_synapse(node_a, strength=0.9)
                    new_node.add_synapse(node_b, strength=0.9)
                    node_a.add_synapse(new_node, strength=0.9)
                    node_b.add_synapse(new_node, strength=0.9)

                    # Slightly weaken the direct connection to encourage use of the abstract path
                    syn_ab.strength *= 0.8
                    syn_ba.strength *= 0.8
                    created_count += 1
        
        if created_count > 0:
            print(f"✨ Created {created_count} new abstract concepts.")


    def get_top_activated_nodes(self, k: int = 5) -> List[KnowledgeNode]:
        """Returns the top k most activated nodes after a recall cycle."""
        if not self.nodes: return []
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.activation_level, reverse=True)
        return sorted_nodes[:k]

    def visualize_and_save_graph(self):
        """Creates and saves a visual representation of the knowledge graph."""
        if not VISUALIZATION_ENABLED or not self.nodes:
            return

        G = nx.DiGraph()
        node_labels = {n.id: f"{n.label[:20]}\n{n.activation_level:.2f}" for n in self.nodes.values()}
        node_sizes = [1000 + (n.importance * 4000) for n in self.nodes.values()]
        node_colors = [n.activation_level for n in self.nodes.values()]

        for node in self.nodes.values():
            G.add_node(node.id)

        # Kumpulkan edge widths dan colors
        edge_widths = []
        edge_colors = []
        for n in self.nodes.values():
            for s in n.synapses:
                G.add_edge(n.id, s.target.id)
                edge_widths.append(s.strength * 5)
                edge_colors.append(s.trace)

        print("📊 Generating graph visualization...")
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=1.5 / np.sqrt(len(self.nodes)), iterations=70)

        # Gambar node
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9)
        # Gambar edges (tanpa colorbar, warna dari trace)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cividis, alpha=0.6, arrows=True, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

        # Colorbar hanya untuk node (aktivasi)
        if node_colors:
            plt.colorbar(nodes, label="Activation Level")

        plt.title("Samre's Samantic Garden", fontsize=24)
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
            "rata-rata_kekuatan_sinapsis": np.mean([s.strength for n in self.nodes.values() for s in n.synapses if s]) if num_synapses > 0 else 0,
            "rata-rata_aktivasi": np.mean([n.activation_level for n in self.nodes.values()]) if self.nodes else 0
        }
