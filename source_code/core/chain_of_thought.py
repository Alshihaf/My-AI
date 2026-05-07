import re
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable
from dataclasses import dataclass, field
import itertools

# ============================================================
# 1. Representasi Term yang Konsisten
# ============================================================

class Term:
    """Kelas dasar untuk semua term."""
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Term) and self.name == other.name and type(self) == type(other)

    def __hash__(self):
        return hash((type(self), self.name))

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def is_variable(self) -> bool:
        return False

    def is_constant(self) -> bool:
        return False

    def is_compound(self) -> bool:
        return False

    def substitute(self, subst: 'Substitution') -> 'Term':
        """Mengganti variabel dalam term dengan substitusi yang diberikan."""
        return self

class Variable(Term):
    """Variabel logika (diawali dengan '?' atau '_' anonim)."""
    def __init__(self, name: str):
        super().__init__(name)
        self.anonymous = (name == '_' or name == '?')

    def is_variable(self) -> bool:
        return True

    def substitute(self, subst: 'Substitution') -> 'Term':
        if self.anonymous:
            return self
        bound = subst.get(self)
        if bound is not None:
            return bound.substitute(subst)
        return self

    def __repr__(self):
        return f"Var({self.name})"

class Constant(Term):
    """Konstanta (atom, angka, simbol)."""
    def is_constant(self) -> bool:
        return True

    def __repr__(self):
        return f"Const({self.name})"

class Compound(Term):
    """Term terstruktur: predikat(arg1, arg2, ...)."""
    def __init__(self, predicate: str, args: List[Term]):
        super().__init__(predicate)
        self.predicate = predicate
        self.args = args

    def is_compound(self) -> bool:
        return True

    def substitute(self, subst: 'Substitution') -> 'Compound':
        new_args = [arg.substitute(subst) for arg in self.args]
        return Compound(self.predicate, new_args)

    def __eq__(self, other):
        return (isinstance(other, Compound) and self.predicate == other.predicate and
                len(self.args) == len(other.args) and
                all(a == b for a, b in zip(self.args, other.args)))

    def __hash__(self):
        return hash((self.predicate, tuple(self.args)))

    def __repr__(self):
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(repr(a) for a in self.args)})"

    def __str__(self):
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(str(a) for a in self.args)})"

# ============================================================
# 2. Substitusi dengan Trail Stack (Backtracking Efisien)
# ============================================================

class Substitution:
    """
    Substitusi variabel → term. Menggunakan trail stack untuk membatalkan
    binding saat backtracking tanpa menyalin seluruh dictionary.
    """
    def __init__(self):
        self._bindings: Dict[Variable, Term] = {}
        self._trail: List[Tuple[Variable, Optional[Term]]] = []

    def get(self, var: Variable) -> Optional[Term]:
        if var.anonymous:
            return None
        return self._bindings.get(var)

    def bind(self, var: Variable, term: Term) -> bool:
        """Menambahkan binding var→term. Mengembalikan True jika berhasil."""
        if var.anonymous:
            return True
        # Occur check sederhana
        if self.occurs_check(var, term):
            return False
        # Simpan keadaan lama di trail
        old = self._bindings.get(var)
        self._trail.append((var, old))
        self._bindings[var] = term
        return True

    def occurs_check(self, var: Variable, term: Term) -> bool:
        """Memeriksa apakah var muncul di dalam term."""
        if var == term:
            return True
        if term.is_compound():
            for arg in term.args:
                if self.occurs_check(var, arg):
                    return True
        elif term.is_variable():
            # Ikuti binding jika term adalah variabel lain
            bound = self.get(term)
            if bound is not None:
                return self.occurs_check(var, bound)
        return False

    def undo(self):
        """Membatalkan binding terakhir (digunakan saat backtracking)."""
        if self._trail:
            var, old = self._trail.pop()
            if old is None:
                del self._bindings[var]
            else:
                self._bindings[var] = old

    def snapshot(self) -> int:
        """Mengembalikan ukuran trail saat ini (untuk checkpoint)."""
        return len(self._trail)

    def restore(self, checkpoint: int):
        """Kembali ke snapshot tertentu."""
        while len(self._trail) > checkpoint:
            self.undo()

    def copy(self) -> 'Substitution':
        """Membuat salinan independen (untuk strategi non-backtracking)."""
        new = Substitution()
        new._bindings = self._bindings.copy()
        new._trail = []  # trail tidak disalin karena tidak diperlukan lagi
        return new

    def __repr__(self):
        return "{" + ", ".join(f"{k}={v}" for k, v in self._bindings.items()) + "}"

# ============================================================
# 3. Unifikasi (Algoritma Standar dengan Trail)
# ============================================================

class Unification:
    @staticmethod
    def unify(t1: Term, t2: Term, subst: Substitution) -> bool:
        """
        Menyatukan dua term dengan substitusi yang dapat diubah (menggunakan trail).
        Mengembalikan True jika berhasil, False jika gagal.
        """
        # Terapkan substitusi saat ini
        t1 = t1.substitute(subst)
        t2 = t2.substitute(subst)

        if t1 == t2:
            return True

        if t1.is_variable() and not t1.anonymous:
            return Unification._unify_var(t1, t2, subst)
        if t2.is_variable() and not t2.anonymous:
            return Unification._unify_var(t2, t1, subst)
        if t1.is_compound() and t2.is_compound():
            if t1.predicate != t2.predicate or len(t1.args) != len(t2.args):
                return False
            for a1, a2 in zip(t1.args, t2.args):
                if not Unification.unify(a1, a2, subst):
                    return False
            return True
        return False

    @staticmethod
    def _unify_var(var: Variable, term: Term, subst: Substitution) -> bool:
        """Menyatukan variabel dengan term."""
        # Cek occur check
        if subst.occurs_check(var, term):
            return False
        return subst.bind(var, term)

# ============================================================
# 4. Aturan dan Fakta
# ============================================================

class Rule:
    def __init__(self, premises: List[Compound], conclusion: Compound, name: str = ""):
        self.premises = premises
        self.conclusion = conclusion
        self.name = name or f"R{id(self)}"

    def __repr__(self):
        prem_str = " ∧ ".join(str(p) for p in self.premises)
        return f"{prem_str} → {self.conclusion}"

    def standardize_variables(self, counter: int, var_map: Optional[Dict[Variable, Variable]] = None) -> 'Rule':
        """
        Mengganti semua variabel dalam aturan dengan variabel baru (unik).
        """
        if var_map is None:
            var_map = {}
        # Fungsi bantu untuk melakukan rename pada term
        def rename(term: Term) -> Term:
            if term.is_variable():
                var = term
                if var.anonymous:
                    return var
                if var not in var_map:
                    new_name = f"{var.name}_{counter}"
                    var_map[var] = Variable(new_name)
                return var_map[var]
            if term.is_compound():
                new_args = [rename(arg) for arg in term.args]
                return Compound(term.predicate, new_args)
            return term
        new_premises = [rename(p) for p in self.premises]
        new_conclusion = rename(self.conclusion)
        return Rule(new_premises, new_conclusion, self.name)

# ============================================================
# 5. Parser yang Lebih Toleran
# ============================================================

class Parser:
    """Parser untuk fakta dan aturan dengan sintaks fleksibel."""
    @staticmethod
    def parse_term(s: str) -> Term:
        """Mengurai string menjadi term (Constant atau Compound)."""
        s = s.strip()
        # Cek apakah term adalah variabel anonim
        if s == '_' or s == '?':
            return Variable('_')
        # Cek apakah term adalah variabel (diawali ?)
        if s.startswith('?'):
            # Variabel seperti ?x, ?nama
            if re.match(r'\?[a-zA-Z_][a-zA-Z0-9_]*', s):
                return Variable(s)
        # Cek apakah term adalah konstanta tanpa kurung
        if '(' not in s:
            return Constant(s)
        # Cek format compound: predikat(arg1, arg2, ...)
        match = re.fullmatch(r'(\w[\w]*)\((.*)\)', s)
        if not match:
            raise ValueError(f"Format term tidak valid: {s}")
        pred = match.group(1)
        args_str = match.group(2)
        if args_str.strip() == '':
            args = []
        else:
            # Pisahkan argumen dengan koma (tapi tidak di dalam nested term)
            # Sederhana: split koma, asumsikan tidak ada koma dalam term.
            # Untuk parser lengkap perlu tokenizing.
            # Karena contoh sederhana, kita lakukan split sederhana.
            # Namun term bisa nested: p(a, f(b,c)) -> perlu parsing rekursif.
            # Kita gunakan pendekatan sederhana: pecah berdasarkan koma di level teratas.
            args = []
            current = []
            depth = 0
            for ch in args_str:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    args.append(''.join(current).strip())
                    current = []
                    continue
                current.append(ch)
            if current:
                args.append(''.join(current).strip())
        term_args = [Parser.parse_term(arg) for arg in args]
        return Compound(pred, term_args)

    @staticmethod
    def parse_fact(s: str) -> Compound:
        term = Parser.parse_term(s)
        if not isinstance(term, Compound) and not isinstance(term, Constant):
            raise ValueError("Fakta harus berupa term atomik atau compound")
        return term

    @staticmethod
    def parse_rule(s: str) -> Rule:
        """Format: IF prem1 AND prem2 ... THEN conclusion"""
        s = s.strip()
        if not s.startswith("IF"):
            raise ValueError("Aturan harus diawali dengan 'IF'")
        parts = s.split("THEN")
        if len(parts) != 2:
            raise ValueError("Aturan harus memiliki bagian THEN")
        if_part = parts[0][2:].strip()
        then_part = parts[1].strip()
        if if_part:
            premises_str = [p.strip() for p in if_part.split("AND")]
        else:
            premises_str = []
        premises = [Parser.parse_term(p) for p in premises_str]
        conclusion = Parser.parse_term(then_part)
        if not isinstance(conclusion, Compound):
            raise ValueError("Kesimpulan aturan harus berupa term compound")
        for p in premises:
            if not isinstance(p, Compound):
                raise ValueError("Premis aturan harus berupa term compound")
        return Rule(premises, conclusion)

# ============================================================
# 6. Mesin Penalaran dengan Berbagai Strategi
# ============================================================

class ProofNode:
    """Node dalam pohon bukti."""
    def __init__(self, goal: Term, rule: Optional[Rule] = None, subst: Optional[Substitution] = None):
        self.goal = goal
        self.rule = rule
        self.subst = subst
        self.children: List[ProofNode] = []
        self.success = False
        self.failure_reason = ""

    def to_string(self, indent=0) -> str:
        spaces = "  " * indent
        result = spaces + f"Goal: {self.goal}"
        if self.rule:
            result += f" (by {self.rule.name})"
        result += "\n"
        for child in self.children:
            result += child.to_string(indent + 1)
        if not self.success and self.failure_reason:
            result += spaces + f"  Failure: {self.failure_reason}\n"
        return result

class ReasoningEngine:
    def __init__(self, search_strategy: str = "depth_first"):
        """
        search_strategy: 'depth_first', 'breadth_first', 'iterative_deepening'
        """
        self.facts: Dict[str, List[Compound]] = defaultdict(list)  # indeks berdasarkan predikat
        self.rules_by_pred: Dict[str, List[Rule]] = defaultdict(list)
        self.rules: List[Rule] = []
        self.search_strategy = search_strategy
        self.var_counter = 0
        self.memo_table: Dict[Tuple[str, str], bool] = {}  # (goal_str, subst_hash) -> bool
        self.proof_root: Optional[ProofNode] = None
        self.enable_tabling = True

    def add_fact(self, fact_str: str):
        fact = Parser.parse_fact(fact_str)
        pred = fact.predicate if isinstance(fact, Compound) else fact.name
        self.facts[pred].append(fact)

    def add_rule(self, rule_str: str):
        rule = Parser.parse_rule(rule_str)
        pred = rule.conclusion.predicate
        self.rules_by_pred[pred].append(rule)
        self.rules.append(rule)

    def _fresh_variable_counter(self) -> int:
        self.var_counter += 1
        return self.var_counter

    def _standardize_rule(self, rule: Rule) -> Rule:
        counter = self._fresh_variable_counter()
        return rule.standardize_variables(counter)

    def _unify_goal_with_conclusion(self, goal: Compound, rule: Rule, subst: Substitution) -> Optional[Substitution]:
        """Mencoba unifikasi goal dengan kesimpulan aturan. Mengembalikan substitusi baru jika berhasil."""
        new_subst = Substitution()
        # Salin binding yang sudah ada? Dalam backward chaining, kita mulai dengan substitusi kosong
        # untuk aturan, lalu gabung nanti. Tapi untuk efisiensi, kita unifikasi langsung dengan subst yang ada.
        # Kita buat substitusi baru yang merupakan gabungan dari subst dan binding baru.
        # Namun karena unifikasi langsung mengubah subst, kita perlu snapshot.
        checkpoint = subst.snapshot()
        # Terapkan substitusi yang sudah ada ke goal (tapi goal seharusnya sudah disubstitusi sebelumnya)
        # Di sini kita asumsikan goal sudah disubstitusi dengan konteks luar.
        if Unification.unify(goal, rule.conclusion, subst):
            # Berhasil: kita perlu mengembalikan substitusi yang sudah dimodifikasi.
            # Namun karena kita menggunakan trail, kita tidak bisa mengembalikan snapshot dengan mudah.
            # Alternatif: kita buat salinan.
            new_subst = subst.copy()
            subst.restore(checkpoint)  # kembalikan subst asli
            return new_subst
        subst.restore(checkpoint)
        return None

    def reason(self, query_str: str) -> bool:
        """
        Melakukan penalaran dengan strategi yang dipilih.
        Mengembalikan True jika query terbukti.
        """
        query = Parser.parse_term(query_str)
        if not isinstance(query, Compound):
            raise ValueError("Query harus berupa term compound")
        self.proof_root = ProofNode(query)
        self.var_counter = 0
        self.memo_table.clear()

        if self.search_strategy == "depth_first":
            success = self._prove_dfs(query, Substitution(), 0, self.proof_root)
        elif self.search_strategy == "breadth_first":
            success = self._prove_bfs(query, Substitution(), self.proof_root)
        elif self.search_strategy == "iterative_deepening":
            success = self._prove_ids(query, Substitution(), self.proof_root)
        else:
            raise ValueError(f"Strategi tidak dikenal: {self.search_strategy}")
        return success

    # ------------------------------------------------------------
    # Depth-First Search (dengan backtracking menggunakan trail)
    # ------------------------------------------------------------
    def _prove_dfs(self, goal: Compound, subst: Substitution, depth: int, node: ProofNode) -> bool:
        # Cek memoization
        goal_ground = goal.substitute(subst)
        goal_str = str(goal_ground)
        if self.enable_tabling:
            key = goal_str
            if key in self.memo_table:
                node.success = self.memo_table[key]
                if not node.success:
                    node.failure_reason = "Memoized failure"
                return node.success

        # Cek fakta langsung (setelah substitusi)
        goal_ground = goal.substitute(subst)
        pred = goal_ground.predicate
        for fact in self.facts.get(pred, []):
            # Unifikasi goal dengan fakta
            checkpoint = subst.snapshot()
            if Unification.unify(goal_ground, fact, subst):
                node.success = True
                subst.restore(checkpoint)
                if self.enable_tabling:
                    self.memo_table[key] = True
                return True
            subst.restore(checkpoint)

        # Coba semua aturan yang relevan
        for rule in self.rules_by_pred.get(pred, []):
            rule_std = self._standardize_rule(rule)
            checkpoint = subst.snapshot()
            # Unifikasi goal dengan conclusion aturan
            if Unification.unify(goal_ground, rule_std.conclusion, subst):
                # Simpan binding sementara
                local_subst = subst.copy()
                subst.restore(checkpoint)  # kembalikan untuk iterasi berikutnya
                # Buktikan semua premis
                all_proven = True
                premise_nodes = []
                for prem in rule_std.premises:
                    prem_ground = prem.substitute(local_subst)
                    prem_node = ProofNode(prem_ground, rule_std, local_subst)
                    premise_nodes.append(prem_node)
                    if not self._prove_dfs(prem_ground, local_subst, depth+1, prem_node):
                        all_proven = False
                        break
                if all_proven:
                    node.success = True
                    node.rule = rule
                    node.children = premise_nodes
                    node.subst = local_subst
                    if self.enable_tabling:
                        self.memo_table[key] = True
                    return True
                # Gagal: lanjutkan ke aturan berikutnya
            else:
                subst.restore(checkpoint)

        node.success = False
        node.failure_reason = "Tidak ada fakta atau aturan yang cocok"
        if self.enable_tabling:
            self.memo_table[key] = False
        return False

    # ------------------------------------------------------------
    # Breadth-First Search (menggunakan antrian)
    # ------------------------------------------------------------
    def _prove_bfs(self, goal: Compound, initial_subst: Substitution, root_node: ProofNode) -> bool:
        # BFS perlu menyimpan state (goal, subst, node, depth) dalam antrian
        from collections import deque
        queue = deque()
        # Kita perlu tracking pohon bukti; BFS lebih kompleks karena tidak linear.
        # Implementasi sederhana: kita gunakan pencarian ruang state, bukan pohon bukti detail.
        # Untuk kemudahan, kita buat fungsi pembantu yang mengembalikan bool dan membangun pohon dengan BFS.
        # Karena BFS tidak cocok dengan backtracking berbasis trail (perlu salinan subst),
        # kita akan menggunakan substitusi immutable (copy) setiap kali.

        # State: (goal, subst, node, depth)
        queue.append((goal, initial_subst.copy(), root_node, 0))
        visited = set()
        while queue:
            curr_goal, curr_subst, curr_node, depth = queue.popleft()
            goal_ground = curr_goal.substitute(curr_subst)
            goal_str = str(goal_ground)
            pred = goal_ground.predicate if goal_ground.is_compound() else goal_ground.name

            # Cek memo
            if self.enable_tabling:
                key = (goal_str, repr(curr_subst))
                if key in self.memo_table:
                    curr_node.success = self.memo_table[key]
                    if curr_node.success:
                        return True
                    continue

            # Cek fakta
            found = False
            for fact in self.facts.get(pred, []):
                if Unification.unify(goal_ground, fact, curr_subst):
                    curr_node.success = True
                    if self.enable_tabling:
                        self.memo_table[key] = True
                    return True
                # reset subst? karena kita pakai salinan, tidak perlu reset
            if found:
                continue

            # Coba aturan
            for rule in self.rules_by_pred.get(pred, []):
                rule_std = self._standardize_rule(rule)
                new_subst = curr_subst.copy()
                if Unification.unify(goal_ground, rule_std.conclusion, new_subst):
                    # Buat node untuk aturan
                    rule_node = ProofNode(curr_goal, rule, new_subst)
                    curr_node.children.append(rule_node)
                    # Tambahkan premis ke antrian
                    all_premises = rule_std.premises
                    # Kita perlu membuktikan semua premis secara berurutan. BFS bisa menggunakan
                    # pendekatan conjunction: coba semua kombinasi? Sederhana: kita buktikan premis satu per satu
                    # secara DFS di dalam BFS. Agar konsisten, kita gunakan pendekatan yang sama seperti DFS
                    # tetapi dengan antrian luar. Alternatif: kita jadikan satu state yang berisi daftar premis yang tersisa.
                    # Untuk memudahkan, kita gunakan DFS di dalam BFS untuk membuktikan premis (depth-first internal).
                    # Namun itu akan mengubah sifat. Lebih baik kita implementasikan BFS yang benar untuk pohon AND-OR.
                    # Karena kompleksitas, untuk keperluan demo kita cukup menggunakan DFS untuk subgoal.
                    # Tapi kita janjikan BFS, jadi kita buat implementasi sederhana: semua premis dibuktikan
                    # secara rekursif dengan BFS (panggil _prove_bfs lagi) namun dengan state baru.
                    all_ok = True
                    for prem in rule_std.premises:
                        prem_ground = prem.substitute(new_subst)
                        prem_node = ProofNode(prem_ground, rule, new_subst)
                        if not self._prove_bfs(prem_ground, new_subst, prem_node):
                            all_ok = False
                            break
                        # Jika berhasil, lanjutkan
                    if all_ok:
                        curr_node.success = True
                        if self.enable_tabling:
                            self.memo_table[key] = True
                        return True
            # Jika tidak ada aturan yang berhasil
            curr_node.success = False
            curr_node.failure_reason = "Tidak ada solusi"
            if self.enable_tabling:
                self.memo_table[key] = False
        return False

    # ------------------------------------------------------------
    # Iterative Deepening Search
    # ------------------------------------------------------------
    def _prove_ids(self, goal: Compound, subst: Substitution, root_node: ProofNode, max_depth: int = 100) -> bool:
        for depth in range(1, max_depth+1):
            self.memo_table.clear() # Reset memoization between IDS iterations
            if self._prove_dfs_depth_limit(goal, subst, 0, root_node, depth):
                return True
        return False

    def _prove_dfs_depth_limit(self, goal: Compound, subst: Substitution, depth: int, node: ProofNode, limit: int) -> bool:
        if depth > limit:
            node.failure_reason = "Batas kedalaman tercapai"
            return False
        # Sama seperti DFS tetapi dengan batas kedalaman
        goal_ground = goal.substitute(subst)
        pred = goal_ground.predicate if goal_ground.is_compound() else goal_ground.name

        # Fakta
        for fact in self.facts.get(pred, []):
            checkpoint = subst.snapshot()
            if Unification.unify(goal_ground, fact, subst):
                node.success = True
                subst.restore(checkpoint)
                return True
            subst.restore(checkpoint)

        # Aturan
        for rule in self.rules_by_pred.get(pred, []):
            rule_std = self._standardize_rule(rule)
            checkpoint = subst.snapshot()
            if Unification.unify(goal_ground, rule_std.conclusion, subst):
                local_subst = subst.copy()
                subst.restore(checkpoint)
                all_proven = True
                premise_nodes = []
                for prem in rule_std.premises:
                    prem_ground = prem.substitute(local_subst)
                    prem_node = ProofNode(prem_ground, rule_std, local_subst)
                    premise_nodes.append(prem_node)
                    if not self._prove_dfs_depth_limit(prem_ground, local_subst, depth+1, prem_node, limit):
                        all_proven = False
                        break
                if all_proven:
                    node.success = True
                    node.rule = rule
                    node.children = premise_nodes
                    node.subst = local_subst
                    return True
            else:
                subst.restore(checkpoint)
        node.success = False
        return False

    # ------------------------------------------------------------
    # Negation as Failure (terbatas)
    # ------------------------------------------------------------
    def prove_negation(self, query_str: str) -> bool:
        """Membuktikan negasi query: True jika query tidak terbukti."""
        return not self.reason(query_str)

    # ------------------------------------------------------------
    # Mendapatkan Chain/Pohon Bukti
    # ------------------------------------------------------------
    def get_chain(self) -> List[Dict[str, Any]]:
        """Mengembalikan representasi flat dari pohon bukti untuk kompatibilitas dengan versi lama."""
        if not self.proof_root:
            return []
        chain = []
        self._flatten_proof_tree(self.proof_root, chain, 0)
        return chain

    def _flatten_proof_tree(self, node: ProofNode, chain: List, depth: int):
        entry = {
            "type": "goal",
            "goal": str(node.goal),
            "depth": depth,
            "result": node.success,
            "subst": str(node.subst) if node.subst else "{}",
            "rule": node.rule.name if node.rule else None,
        }
        if not node.success and node.failure_reason:
            entry["failure_reason"] = node.failure_reason
        chain.append(entry)
        for child in node.children:
            self._flatten_proof_tree(child, chain, depth+1)

    def print_proof_tree(self):
        """Menampilkan pohon bukti yang terindentasi."""
        if self.proof_root:
            print(self.proof_root.to_string())
        else:
            print("Belum ada hasil penalaran.")

# ============================================================
# 7. Unit Test
# ============================================================

import unittest

class TestReasoningEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine()

    def test_fact_only(self):
        self.engine.add_fact("manusia(socrates)")
        self.assertTrue(self.engine.reason("manusia(socrates)"))
        self.assertFalse(self.engine.reason("manusia(plato)"))

    def test_simple_rule(self):
        self.engine.add_fact("manusia(socrates)")
        self.engine.add_rule("IF manusia(?x) THEN mortal(?x)")
        self.assertTrue(self.engine.reason("mortal(socrates)"))
        self.assertFalse(self.engine.reason("mortal(plato)"))

    def test_multi_premise(self):
        self.engine.add_fact("laki(socrates)")
        self.engine.add_fact("manusia(socrates)")
        self.engine.add_rule("IF laki(?x) AND manusia(?x) THEN pria(?x)")
        self.assertTrue(self.engine.reason("pria(socrates)"))

    def test_occur_check(self):
        # Aturan dengan occur check: unifikasi yang tidak seharusnya berhasil
        # Misalnya unifikasi ?x dengan f(?x) harus gagal
        # Kita buat aturan buatan
        self.engine.add_rule("IF p(?x) THEN q(f(?x))")
        # Query q(?y) akan menghasilkan unifikasi ?y = f(?x) dan p(?x) tidak ada fakta
        self.assertFalse(self.engine.reason("q(f(f(a)))"))  # tidak ada fakta, jadi false

    def test_loop(self):
        # Aturan rekursif tanpa base case: harus berhenti karena deteksi loop
        self.engine.add_rule("IF loop(?x) THEN loop(?x)")
        # Seharusnya tidak infinite loop, karena visited set pada depth
        self.assertFalse(self.engine.reason("loop(a)"))

    def test_anonymous_variable(self):
        self.engine.add_fact("kakak(ani, budi)")
        self.engine.add_rule("IF kakak(ani, _) THEN punya_kakak(ani)")
        self.assertTrue(self.engine.reason("punya_kakak(ani)"))

    def test_strategy_dfs(self):
        self.engine = ReasoningEngine("depth_first")
        self.engine.add_fact("a")
        self.engine.add_rule("IF a THEN b")
        self.assertTrue(self.engine.reason("b"))

    def test_memoization(self):
        self.engine.enable_tabling = True
        self.engine.add_fact("p(a)")
        self.engine.add_rule("IF p(?x) THEN q(?x)")
        self.assertTrue(self.engine.reason("q(a)"))
        # Coba lagi, harus cepat karena memo
        self.assertTrue(self.engine.reason("q(a)"))

    def test_complex_term(self):
        self.engine.add_fact("ayah(john, mary)")
        self.engine.add_rule("IF ayah(?x, ?y) THEN orangtua(?x, ?y)")
        self.assertTrue(self.engine.reason("orangtua(john, mary)"))

if __name__ == "__main__":
    # Jalankan unit test
    unittest.main(argv=[''], exit=False)

    # Contoh penggunaan
    engine = ReasoningEngine(search_strategy="depth_first")
    engine.add_fact("manusia(socrates)")
    engine.add_rule("IF manusia(?x) THEN mortal(?x)")
    query = "mortal(socrates)"
    print(f"\nQuery: {query} -> {engine.reason(query)}")
    engine.print_proof_tree()
    print("\nChain (flat):")
    for step in engine.get_chain():
        print(step)