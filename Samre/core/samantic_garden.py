import time
import numpy as np
import random
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


# =============================================================================
#  Synapse: Hebbian connection with eligibility trace (TD-λ style)
# =============================================================================

class Synapse:
    """
    Directional, plastic connection between two NeuralNodes.
    Uses an eligibility trace for delayed, reward-modulated Hebbian learning.
    """
    def __init__(self, target_node: 'NeuralNode', initial_strength: float = 0.1):
        self.target = target_node
        self.strength = np.clip(initial_strength, 0.01, 1.0)
        self.trace = 0.0          # Eligibility trace (exponential decay)
        self.trace_decay = 0.95

    def hebbian_update(self, pre_activation: float, post_activation: float,
                       learning_rate: float):
        """
        Update eligibility trace and synaptic strength.
        The trace accumulates pre*post products; strength changes proportional
        to trace * learning_rate.
        """
        self.trace = (self.trace * self.trace_decay) + (pre_activation * post_activation)
        delta = learning_rate * self.trace
        self.strength = np.clip(self.strength + delta, 0.01, 1.0)

    def decay(self, decay_factor: float):
        """Homeostatic decay to prevent saturation."""
        self.strength *= (1.0 - decay_factor)
        self.trace *= self.trace_decay * 0.5


# =============================================================================
#  NeuralNode: RL-based concept node with multi-arm policy
# =============================================================================

class NeuralNode:
    """
    Represents a single concept as a Reinforcement Learning agent.

    The node maintains a population of "arms" (sub-policies). At each activation:
      1. Compute state features from the incoming signal and the concept vector.
      2. Select an arm via softmax on learned preferences (bandit).
      3. Sample a continuous output activation from a Gaussian policy whose
         mean is given by the chosen arm's linear function.
      4. Later, receive a reward and update:
         - The arm preferences (multi-armed bandit).
         - The linear weights of the chosen arm (policy gradient).
         - A running baseline for variance reduction.

    This design preserves the idea of internal competition but makes it fully
    driven by Reinforcement Learning.
    """

    # State dimension: 1 (total_input) + 1 (constant 1.0) + N_FEATURES from concept vector
    N_VECTOR_FEATURES = 8   # number of vector dimensions used as state features

    def __init__(self, concept_vector: np.ndarray, label: str, source: str,
                 keywords: List[str] = None,
                 num_arms: int = 4,
                 learning_rate_w: float = 0.02,    # for arm weights
                 learning_rate_pref: float = 0.1,  # for bandit preferences
                 sigma: float = 0.15):             # exploration noise std

        # Identity
        self.id = f"{label.replace(' ', '_')}_{int(time.time()*1000)}"
        self.vector = concept_vector.astype(np.float32)
        self.label = label
        self.source = source
        self.keywords = set(keywords or [])
        self.synapses: List[Synapse] = []
        self.created_at = time.time()
        self.importance = 0.1
        self.output_activation = 0.0

        # --- RL parameters ---
        self.num_arms = num_arms
        self.sigma = sigma
        self.lr_w = learning_rate_w
        self.lr_pref = learning_rate_pref

        # State feature size
        self.state_dim = 1 + 1 + min(self.N_VECTOR_FEATURES, len(self.vector))

        # Arms: each arm has a weight vector W (state_dim,) and a bias b
        self.W = np.random.randn(self.num_arms, self.state_dim).astype(np.float32) * 0.01
        self.b = np.zeros(self.num_arms, dtype=np.float32)

        # Bandit preferences (logits for softmax selection)
        self.preferences = np.zeros(self.num_arms, dtype=np.float32)

        # Running average reward (baseline for REINFORCE)
        self.baseline = 0.0
        self.baseline_momentum = 0.9

        # Selected arm & stats from last forward pass (for learning)
        self.last_chosen_arm: Optional[int] = None
        self.last_mean: float = 0.0
        self.last_state: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    #  State construction
    # -------------------------------------------------------------------------
    def _build_state(self, total_input: float) -> np.ndarray:
        """
        Build the state vector for the RL agent.
        It consists of:
          - total_input (scalar)
          - a constant 1.0 (bias)
          - up to N_VECTOR_FEATURES entries from the concept vector.
        """
        vlen = min(self.N_VECTOR_FEATURES, len(self.vector))
        state = np.empty(self.state_dim, dtype=np.float32)
        state[0] = total_input
        state[1] = 1.0
        state[2:2+vlen] = self.vector[:vlen]
        return state

    # -------------------------------------------------------------------------
    #  Forward pass (action selection & sampling)
    # -------------------------------------------------------------------------
    def process_stimulus(self, total_input: float) -> float:
        """
        Process incoming signal. Select an arm stochastically, sample output
        activation from Gaussian policy, and store data for later learning.

        Returns:
            output_activation (float between 0 and 1).
        """
        if self.num_arms == 0:
            return 0.0

        state = self._build_state(total_input)
        self.last_state = state

        # 1. Select arm via softmax on preferences
        prefs = self.preferences  # size (num_arms,)
        # Subtract max for numerical stability
        prefs_shifted = prefs - np.max(prefs)
        exp_prefs = np.exp(prefs_shifted)
        probs = exp_prefs / exp_prefs.sum()

        chosen = np.random.choice(self.num_arms, p=probs)
        self.last_chosen_arm = chosen

        # 2. Compute mean activation for the chosen arm: sigmoid( W·s + b )
        linear = np.dot(self.W[chosen], state) + self.b[chosen]
        mean = 1.0 / (1.0 + np.exp(-linear))  # sigmoid, output in (0,1)
        self.last_mean = mean

        # 3. Sample output from Gaussian around mean, clip to [0,1]
        activation = mean + np.random.normal(0.0, self.sigma)
        activation = max(0.0, min(1.0, activation))

        self.output_activation = activation
        return activation

    # -------------------------------------------------------------------------
    #  Learning update (REINFORCE + Bandit)
    # -------------------------------------------------------------------------
    def learn_from_interaction(self, reward: float, learning_rate_modifier: float = 1.0):
        """
        Perform a learning step using the stored last choice.

        Args:
            reward: scalar reward signal (e.g. Hebbian co-activation).
            learning_rate_modifier: global factor (e.g. arousal).
        """
        if self.last_chosen_arm is None or self.last_state is None:
            return

        # 1. Update baseline (running average reward)
        self.baseline = (self.baseline_momentum * self.baseline +
                         (1.0 - self.baseline_momentum) * reward)
        advantage = reward - self.baseline

        # 2. Update arm preferences (multi-armed bandit)
        # Recompute probabilities for the same state (probs unchanged in this call)
        prefs = self.preferences
        prefs_shifted = prefs - np.max(prefs)
        exp_prefs = np.exp(prefs_shifted)
        probs = exp_prefs / exp_prefs.sum()

        i = self.last_chosen_arm
        # Increase log-preference of chosen arm, decrease others
        self.preferences[i] += self.lr_pref * advantage * (1.0 - probs[i])
        for j in range(self.num_arms):
            if j != i:
                self.preferences[j] -= self.lr_pref * advantage * probs[j]

        # 3. Policy gradient update for the chosen arm's weights
        # Log-policy gradient for Gaussian: (a - mean) / sigma^2 * state
        a = self.output_activation
        mean = self.last_mean
        grad_w = (a - mean) / (self.sigma ** 2) * self.last_state
        grad_b = (a - mean) / (self.sigma ** 2)

        # Effective learning rate
        lr = self.lr_w * learning_rate_modifier
        self.W[i] += lr * advantage * grad_w
        self.b[i] += lr * advantage * grad_b

        # Optional: clamp weights to avoid explosion
        self.W[i] = np.clip(self.W[i], -2.0, 2.0)
        self.b[i] = np.clip(self.b[i], -2.0, 2.0)

        # Clear last info (ready for next cycle)
        self.last_chosen_arm = None
        self.last_state = None

    # -------------------------------------------------------------------------
    #  Periodic consolidation (evolution)
    # -------------------------------------------------------------------------
    def evolve_population(self):
        """
        Called during memory consolidation.
        Here we slightly decay preferences toward zero and reduce sigma,
        simulating stabilisation of well-learned concepts.
        """
        # Decay preferences toward zero (avoid extreme values)
        self.preferences *= 0.9
        # Reduce exploration noise (sigma) down to a minimum
        self.sigma = max(0.05, self.sigma * 0.99)
        # Decay learning rates very slowly
        self.lr_w *= 0.999
        self.lr_pref *= 0.999

    # -------------------------------------------------------------------------
    #  State reset between activation cycles
    # -------------------------------------------------------------------------
    def reset_state(self):
        """Reset output activation and clear temporary learning data."""
        self.output_activation = 0.0
        self.last_chosen_arm = None
        self.last_mean = 0.0
        self.last_state = None

    # -------------------------------------------------------------------------
    #  Synapse management
    # -------------------------------------------------------------------------
    def add_synapse(self, target_node: 'NeuralNode', strength: float):
        if not self.get_synapse_to(target_node.id):
            self.synapses.append(Synapse(target_node, strength))

    def get_synapse_to(self, target_id: str) -> Optional[Synapse]:
        for synapse in self.synapses:
            if synapse.target.id == target_id:
                return synapse
        return None


# =============================================================================
#  SamanticGarden: the connectome manager
# =============================================================================

class SamanticGarden:
    """
    Manages the entire connectome of RL-powered NeuralNodes.
    Orchestrates ingestion, spreading activation, consolidation, abstraction,
    persistence, and visualization.
    """

    def __init__(self, log_dir: str = "Samre/log", persistence_file: Optional[str] = None):
        self.nodes: Dict[str, NeuralNode] = {}
        self.log_dir = log_dir
        self.persistence_file = persistence_file
        self.global_learning_rate = 0.01  # Modulated externally (e.g. by arousal)
        self.config = {
            "ingestion_reinforcement_threshold": 0.5,
            "connection_similarity_threshold": 0.5,
            "consolidation_pruning_threshold": 0.02,
            "consolidation_decay_factor": 0.005,
            "abstraction_cos_threshold": 0.8,
            "num_arms_per_node": 4,       # default arm count for new nodes
        }
        if VISUALIZATION_ENABLED and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.persistence_file:
            self.load_state()

    # -------------------------------------------------------------------------
    #  Utility: cosine similarity
    # -------------------------------------------------------------------------
    @staticmethod
    def _calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot / norm if norm > 0 else 0.0

    def _find_most_similar_node(self, vector: np.ndarray) -> Optional[Tuple[NeuralNode, float]]:
        if not self.nodes:
            return None
        node_list = list(self.nodes.values())
        sims = [self._calculate_similarity(vector, n.vector) for n in node_list]
        idx = np.argmax(sims)
        return node_list[idx], sims[idx]

    # -------------------------------------------------------------------------
    #  Knowledge ingestion
    # -------------------------------------------------------------------------
    def ingest_knowledge(self, concept_vector: List[float], label: str,
                         source: str, keywords: List[str]):
        """Ingest a piece of information, creating or reinforcing a node."""
        vec = np.array(concept_vector, dtype=np.float32)
        best_match, similarity = self._find_most_similar_node(vec) or (None, 0.0)

        if similarity > self.config["ingestion_reinforcement_threshold"]:
            # Reinforce existing concept
            node = best_match
            node.vector = node.vector * 0.9 + vec * 0.1
            node.keywords.update(keywords)
            node.importance = min(1.0, node.importance + 0.1)
            print(f"🧠 Knowledge Reinforced: Node '{node.label}' strengthened.")
        else:
            # Create new concept
            node = NeuralNode(
                vec, label, source, keywords,
                num_arms=self.config["num_arms_per_node"]
            )
            self.nodes[node.id] = node
            print(f"🌱 New Knowledge Born: Node '{label}' created with {node.num_arms} arms.")

        # Connect to similar existing nodes
        for existing in self.nodes.values():
            if existing == node:
                continue
            sim = self._calculate_similarity(node.vector, existing.vector)
            if sim > self.config["connection_similarity_threshold"]:
                node.add_synapse(existing, strength=sim)
                existing.add_synapse(node, strength=sim)
                print(f"🔗 Synapse Formed: '{node.label}' <-> '{existing.label}' (sim: {sim:.2f})")

        # Trigger a short recall to integrate the new node
        self.spreading_activation([node.id], initial_signal=0.7, depth=3)

    # -------------------------------------------------------------------------
    #  Spreading activation (recall)
    # -------------------------------------------------------------------------
    def spreading_activation(self, stimulus_node_ids: List[str],
                             initial_signal: float, depth: int = 5) -> List[NeuralNode]:
        """
        Simulate recall. Activation spreads through the connectome. Each node
        processes input with its RL policy and learns from Hebbian rewards.
        """
        print(f"⚡️ Spreading Activation started from {len(stimulus_node_ids)} nodes...")
        # Reset all nodes
        for node in self.nodes.values():
            node.reset_state()
            for syn in node.synapses:
                syn.trace = 0.0

        # Set initial activation on stimulus nodes
        stimulus = {nid: self.nodes[nid] for nid in stimulus_node_ids if nid in self.nodes}
        for node in stimulus.values():
            node.output_activation = initial_signal

        for _ in range(depth):
            pre_activations = {nid: n.output_activation for nid, n in self.nodes.items()}
            inputs_this_cycle = {nid: 0.0 for nid in self.nodes}

            # Accumulate incoming signals from active nodes
            for nid, node in self.nodes.items():
                if pre_activations.get(nid, 0.0) > 0:
                    for syn in node.synapses:
                        signal = pre_activations[nid] * syn.strength
                        inputs_this_cycle[syn.target.id] += signal

            # Process stimulus in each node (RL action selection)
            post_activations = {}
            for nid, total_in in inputs_this_cycle.items():
                node = self.nodes[nid]
                post_activations[nid] = node.process_stimulus(total_in)

            # Hebbian update for synapses and reward delivery
            for node in self.nodes.values():
                total_hebbian_reward = 0.0
                pre_act = pre_activations.get(node.id, 0.0)
                if pre_act > 0:
                    for syn in node.synapses:
                        post_act = post_activations.get(syn.target.id, 0.0)
                        if post_act > 0:
                            # Update synapse
                            syn.hebbian_update(pre_act, post_act, self.global_learning_rate)
                            total_hebbian_reward += pre_act * post_act
                # Provide reward to the source node for its chosen action
                if total_hebbian_reward > 0:
                    node.learn_from_interaction(total_hebbian_reward,
                                                self.global_learning_rate)

        print("⚡️ Spreading Activation complete.")
        return self.get_top_activated_nodes()

    # -------------------------------------------------------------------------
    #  Memory consolidation (sleep-like cycle)
    # -------------------------------------------------------------------------
    def consolidate_memories(self):
        """
        Prune weak connections, replay important memories, evolve node policies,
        and create abstractions.
        """
        print("\n--- 🌙 Initiating Memory Consolidation (Sleep Cycle) ---")
        total_syn = sum(len(n.synapses) for n in self.nodes.values())

        # 1. Prune weak synapses
        pruned = 0
        for node in self.nodes.values():
            before = len(node.synapses)
            node.synapses[:] = [s for s in node.synapses
                                if s.strength > self.config["consolidation_pruning_threshold"]]
            pruned += before - len(node.synapses)
            for syn in node.synapses:
                syn.decay(self.config["consolidation_decay_factor"])

        print(f"🧹 Pruned {pruned} synapses out of {total_syn}.")

        # 2. Replay important memories
        important = [n for n in self.nodes.values() if n.importance > 0.5]
        if important:
            print(f"🔄 Replaying {min(5, len(important))} important memories...")
            ids = [n.id for n in np.random.choice(important,
                   size=min(len(important), 5), replace=False)]
            self.spreading_activation(stimulus_node_ids=ids, initial_signal=0.8, depth=4)

        # 3. Evolve each node's RL policy
        print("🧬 Evolving node policies...")
        for node in self.nodes.values():
            node.evolve_population()

        # 4. Create abstractions from strongly connected pairs
        self.create_abstract_nodes()

        if VISUALIZATION_ENABLED:
            self.visualize_and_save_graph()
        print("--- ☀️ Memory Consolidation Complete ---\n")

    # -------------------------------------------------------------------------
    #  Abstraction creation
    # -------------------------------------------------------------------------
    def create_abstract_nodes(self):
        """
        Create abstract nodes for strongly, bidirectionally connected pairs.
        To avoid label collisions, incorporate node ids into the label hash.
        """
        print("🤔 Searching for new abstractions...")
        created = 0
        nodes_list = list(self.nodes.values())
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                node_a = nodes_list[i]
                node_b = nodes_list[j]
                syn_ab = node_a.get_synapse_to(node_b.id)
                syn_ba = node_b.get_synapse_to(node_a.id)
                if syn_ab and syn_ba:
                    strength = (syn_ab.strength + syn_ba.strength) / 2
                    if strength > self.config["abstraction_cos_threshold"]:
                        # Unique label using truncated id hashes
                        tail_a = node_a.id[-6:]
                        tail_b = node_b.id[-6:]
                        # Nama singkat yang aman
                        short_a = node_a.label[:8].replace(' ', '_')
                        short_b = node_b.label[:8].replace(' ', '_')
                        abstract_label = f"Abs_{short_a}_{short_b}_{tail_a}_{tail_b}"
                        # Check for duplicate
                        if any(n.label == abstract_label for n in self.nodes.values()):
                            continue
                        abstract_vec = (node_a.vector + node_b.vector) / 2
                        new_node = NeuralNode(
                            abstract_vec, abstract_label, "AbstractionEngine",
                            keywords=list(node_a.keywords.union(node_b.keywords)),
                            num_arms=self.config["num_arms_per_node"]
                        )
                        self.nodes[new_node.id] = new_node
                        # Connect
                        new_node.add_synapse(node_a, 0.8)
                        new_node.add_synapse(node_b, 0.8)
                        node_a.add_synapse(new_node, 0.5)
                        node_b.add_synapse(new_node, 0.5)
                        created += 1
        if created:
            print(f"✨ Created {created} new abstract concepts.")

    # -------------------------------------------------------------------------
    #  Querying
    # -------------------------------------------------------------------------
    def get_top_activated_nodes(self, k: int = 5) -> List[NeuralNode]:
        if not self.nodes:
            return []
        return sorted(self.nodes.values(),
                      key=lambda n: n.output_activation, reverse=True)[:k]

    def get_garden_state(self) -> dict:
        num_syn = sum(len(n.synapses) for n in self.nodes.values())
        strengths = [s.strength for n in self.nodes.values() for s in n.synapses]
        return {
            "jumlah_node": len(self.nodes),
            "jumlah_sinapsis": num_syn,
            "rata_rata_kekuatan_sinapsis": np.mean(strengths) if strengths else 0,
            "rata_rata_aktivasi": np.mean([n.output_activation for n in self.nodes.values()]) if self.nodes else 0,
        }

    # -------------------------------------------------------------------------
    #  Visualization
    # -------------------------------------------------------------------------
    def visualize_and_save_graph(self):
        if not VISUALIZATION_ENABLED or not self.nodes:
            return
        G = nx.DiGraph()
        labels = {n.id: f"{n.label[:15]}\n{n.output_activation:.2f}\n(Arms:{n.num_arms})"
                  for n in self.nodes.values()}
        sizes = [1000 + n.importance * 4000 for n in self.nodes.values()]
        colors = [n.output_activation for n in self.nodes.values()]

        for n in self.nodes.values():
            G.add_node(n.id)

        edges = []
        edge_widths = []
        edge_colors = []
        for n in self.nodes.values():
            for s in n.synapses:
                edges.append((n.id, s.target.id))
                edge_widths.append(s.strength * 5)
                edge_colors.append(s.trace)
        G.add_edges_from(edges)

        print("📊 Generating graph visualization...")
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=1.5 / np.sqrt(len(self.nodes) or 1), iterations=70)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes,
                                       node_color=colors, cmap=plt.cm.viridis, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=edge_widths,
                               edge_color=edge_colors, edge_cmap=plt.cm.cividis,
                               alpha=0.6, arrows=True, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
        if colors:
            plt.colorbar(nodes, label="Activation Level")
        plt.title("Samre's Samantic Garden (RL Backbone)", fontsize=24)
        plt.axis('off')
        filename = os.path.join(self.log_dir,
                                f"samantic_garden_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        try:
            plt.savefig(filename)
            print(f"✅ Graph visualization saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")
        plt.close()

    # -------------------------------------------------------------------------
    #  Persistence
    # -------------------------------------------------------------------------
    def save_state(self):
        if not self.persistence_file:
            return
        print(f"💾 Saving SamanticGarden to {self.persistence_file}...")
        data = {"nodes": {}}
        for nid, node in self.nodes.items():
            data["nodes"][nid] = {
                "label": node.label,
                "vector": node.vector.tolist(),
                "source": node.source,
                "keywords": list(node.keywords),
                "importance": node.importance,
                "synapses": [{"target": s.target.id, "strength": s.strength} for s in node.synapses],
                "num_arms": node.num_arms,
            }
        try:
            import json
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=4)
            print("✅ SamanticGarden saved.")
        except Exception as e:
            print(f"❌ Error saving SamanticGarden: {e}")

    def load_state(self):
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return
        print(f"📂 Loading SamanticGarden from {self.persistence_file}...")
        try:
            import json
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
            # First pass: create nodes
            for nid, ndata in data.get("nodes", {}).items():
                node = NeuralNode(
                    np.array(ndata["vector"], dtype=np.float32),
                    ndata["label"],
                    ndata["source"],
                    ndata.get("keywords", []),
                    num_arms=ndata.get("num_arms", self.config["num_arms_per_node"])
                )
                node.id = nid
                node.importance = ndata["importance"]
                self.nodes[nid] = node
            # Second pass: restore synapses
            for nid, ndata in data.get("nodes", {}).items():
                node = self.nodes[nid]
                for syn in ndata.get("synapses", []):
                    tid = syn["target"]
                    if tid in self.nodes:
                        node.add_synapse(self.nodes[tid], syn["strength"])
            print(f"✅ Loaded {len(self.nodes)} nodes.")
        except Exception as e:
            print(f"❌ Error loading SamanticGarden: {e}")