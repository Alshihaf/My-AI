"""
Reasoning Actuator - Inferensi Aktif pada Graf Pengetahuan

Modul ini bertanggung jawab untuk melakukan penalaran tingkat lanjut pada
SamanticGarden untuk menyimpulkan hubungan baru dan menghasilkan pengetahuan.
"""

from typing import List, Tuple, Optional
import networkx as nx

from core.samantic_garden import SamanticGarden, KnowledgeNode

class ReasoningActuator:
    """
    Menjalankan kueri inferensi pada SamanticGarden untuk menemukan
    pengetahuan dan hubungan yang tersembunyi.
    """
    def __init__(self, garden: SamanticGarden):
        self.garden = garden
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        """Membangun representasi NetworkX dari SamanticGarden untuk analisis."""
        G = nx.Graph()
        for node in self.garden.nodes.values():
            G.add_node(node.id, label=node.label)
        for node in self.garden.nodes.values():
            for synapse in node.synapses:
                G.add_edge(node.id, synapse.target.id, weight=synapse.strength)
        return G

    def update_graph(self):
        """Memperbarui graf internal jika SamanticGarden telah berubah."""
        print("🔄 ReasoningActuator: Memperbarui representasi graf...")
        self.graph = self._build_graph()

    def find_shortest_path(self, start_label: str, end_label: str) -> Optional[List[str]]:
        """
        Menemukan jalur terpendek antara dua konsep di graf pengetahuan.
        Mengembalikan daftar label node di sepanjang jalur.
        """
        self.update_graph()
        try:
            start_node_id = next(node_id for node_id, data in self.graph.nodes(data=True) if data['label'] == start_label)
            end_node_id = next(node_id for node_id, data in self.graph.nodes(data=True) if data['label'] == end_label)

            path_ids = nx.shortest_path(self.graph, source=start_node_id, target=end_node_id, weight='weight')
            path_labels = [self.graph.nodes[node_id]['label'] for node_id in path_ids]
            print(f"🧠 INFERENSI (Jalur Terpendek): Ditemukan jalur antara '{start_label}' dan '{end_label}': {' -> '.join(path_labels)}")
            return path_labels
        except (StopIteration, nx.NetworkXNoPath):
            print(f"🤔 INFERENSI (Jalur Terpendek): Tidak ada jalur yang ditemukan antara '{start_label}' dan '{end_label}'.")
            return None

    def find_central_concepts(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Mengidentifikasi konsep yang paling sentral (hub) di graf.
        Menggunakan PageRank sebagai proksi untuk sentralitas.
        """
        if len(self.graph) < top_n:
            return []
        self.update_graph()
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Urutkan node berdasarkan skor PageRank
        sorted_nodes = sorted(pagerank.items(), key=lambda item: item[1], reverse=True)
        
        # Ambil top N
        central_concepts = []
        for node_id, score in sorted_nodes[:top_n]:
            label = self.graph.nodes[node_id]['label']
            central_concepts.append((label, score))
            print(f"🧠 INFERENSI (Hub): Konsep sentral teridentifikasi: '{label}' (Skor: {score:.4f})")
        
        return central_concepts

    def find_transitive_relations(self, start_label: str) -> List[Tuple[str, str, str]]:
        """
        Menemukan hubungan transitif sederhana: Jika A -> B dan B -> C, laporkan (A, B, C).
        """
        self.update_graph()
        inferred_relations = []
        try:
            start_node_id = next(node_id for node_id, data in self.graph.nodes(data=True) if data['label'] == start_label)
            
            # Temukan semua tetangga B dari A (start_node)
            for b_id in self.graph.neighbors(start_node_id):
                # Temukan semua tetangga C dari B
                for c_id in self.graph.neighbors(b_id):
                    # Pastikan C bukan A dan tidak ada hubungan langsung antara A dan C
                    if c_id != start_node_id and not self.graph.has_edge(start_node_id, c_id):
                        a_label = self.graph.nodes[start_node_id]['label']
                        b_label = self.graph.nodes[b_id]['label']
                        c_label = self.graph.nodes[c_id]['label']
                        inferred_relations.append((a_label, b_label, c_label))
                        print(f"🧠 INFERENSI (Transitif): Ditemukan hubungan: {a_label} -> {b_label} -> {c_label}")

        except StopIteration:
             print(f"🤔 INFERENSI (Transitif): Node awal '{start_label}' tidak ditemukan.")

        return inferred_relations

    def execute(self) -> bool:
        """
        Menjalankan siklus penalaran: memilih jenis inferensi secara acak
        dan mengeksekusinya.
        """
        if not self.garden.nodes:
            print("🤔 REASONING: Tidak ada node di SamanticGarden untuk dinalar.")
            return False

        # Pilih node acak sebagai titik awal
        start_node = list(self.garden.nodes.values())[0]

        # Lakukan berbagai jenis penalaran
        self.find_central_concepts()
        relations = self.find_transitive_relations(start_node.label)

        # Contoh: Buat node baru berdasarkan inferensi transitif
        if relations:
            a, b, c = relations[0]
            new_concept_label = f"Inferred: {a} -> {c}"
            # Logika untuk membuat atau memperkuat node baru di SamanticGarden akan ditambahkan di sini
            print(f"🌱 PENGETAHUAN BARU (Disimpulkan): Hubungan '{a} -> {c}' melalui '{b}' dicatat.")

        return True

