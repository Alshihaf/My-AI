import re
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union, Any
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
        return self

class Variable(Term):
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
    def is_constant(self) -> bool:
        return True

    def __repr__(self):
        return f"Const({self.name})"

class Compound(Term):
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
# 2. Substitusi dengan Trail Stack
# ============================================================

class Substitution:
    def __init__(self):
        self._bindings: Dict[Variable, Term] = {}
        self._trail: List[Tuple[Variable, Optional[Term]]] = []

    def get(self, var: Variable) -> Optional[Term]:
        if var.anonymous:
            return None
        return self._bindings.get(var)

    def bind(self, var: Variable, term: Term) -> bool:
        if var.anonymous:
            return True
        if self.occurs_check(var, term):
            return False
        old = self._bindings.get(var)
        self._trail.append((var, old))
        self._bindings[var] = term
        return True

    def occurs_check(self, var: Variable, term: Term) -> bool:
        if var == term:
            return True
        if term.is_compound():
            for arg in term.args:
                if self.occurs_check(var, arg):
                    return True
        elif term.is_variable():
            bound = self.get(term)
            if bound is not None:
                return self.occurs_check(var, bound)
        return False

    def undo(self):
        if self._trail:
            var, old = self._trail.pop()
            if old is None:
                del self._bindings[var]
            else:
                self._bindings[var] = old

    def snapshot(self) -> int:
        return len(self._trail)

    def restore(self, checkpoint: int):
        while len(self._trail) > checkpoint:
            self.undo()

    def copy(self) -> 'Substitution':
        new = Substitution()
        new._bindings = self._bindings.copy()
        new._trail = []  # trail tidak disalin karena tidak diperlukan untuk salinan
        return new

    def __repr__(self):
        return "{" + ", ".join(f"{k}={v}" for k, v in self._bindings.items()) + "}"

# ============================================================
# 3. Unifikasi
# ============================================================

class Unification:
    @staticmethod
    def unify(t1: Term, t2: Term, subst: Substitution) -> bool:
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
        if var_map is None:
            var_map = {}

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
# 5. Parser
# ============================================================

class Parser:
    @staticmethod
    def parse_term(s: str) -> Term:
        s = s.strip()
        if s == '_' or s == '?':
            return Variable('_')
        if s.startswith('?'):
            if re.match(r'\?[a-zA-Z_][a-zA-Z0-9_]*', s):
                return Variable(s)
        if '(' not in s:
            return Constant(s)
        match = re.fullmatch(r'(\w[\w]*)\((.*)\)', s)
        if not match:
            raise ValueError(f"Format term tidak valid: {s}")
        pred = match.group(1)
        args_str = match.group(2)
        if args_str.strip() == '':
            args = []
        else:
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
        s = s.strip()
        if not s.startswith("IF"):
            raise ValueError("Aturan harus diawali dengan 'IF'")
        parts = s.split("THEN")
        if len(parts) != 2:
            raise ValueError("Aturan harus memiliki bagian THEN")
        if_part = parts[0][2:].strip()
        then_part = parts[1].strip()
        premises_str = [p.strip() for p in if_part.split("AND")] if if_part else []
        premises = [Parser.parse_term(p) for p in premises_str]
        conclusion = Parser.parse_term(then_part)
        if not isinstance(conclusion, Compound):
            raise ValueError("Kesimpulan aturan harus berupa term compound")
        for p in premises:
            if not isinstance(p, Compound):
                raise ValueError("Premis aturan harus berupa term compound")
        return Rule(premises, conclusion)

# ============================================================
# 6. Mesin Penalaran
# ============================================================

class ProofNode:
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
        self.facts: Dict[str, List[Compound]] = defaultdict(list)
        self.rules_by_pred: Dict[str, List[Rule]] = defaultdict(list)
        self.rules: List[Rule] = []
        self.search_strategy = search_strategy
        self.var_counter = 0
        self.memo_table: Dict[Tuple[str, str], bool] = {}
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

    def reason(self, query_str: str) -> bool:
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
    # Depth-First Search (dengan backtracking trail)
    # ------------------------------------------------------------
    def _prove_dfs(self, goal: Compound, subst: Substitution, depth: int, node: ProofNode) -> bool:
        goal_ground = goal.substitute(subst)
        goal_str = str(goal_ground)
        if self.enable_tabling:
            key = (goal_str, repr(subst))
            if key in self.memo_table:
                node.success = self.memo_table[key]
                if not node.success:
                    node.failure_reason = "Memoized failure"
                return node.success

        pred = goal_ground.predicate

        # Cek fakta
        for fact in self.facts.get(pred, []):
            checkpoint = subst.snapshot()
            if Unification.unify(goal_ground, fact, subst):
                node.success = True
                subst.restore(checkpoint)  # kembalikan subst, bukti tak perlu ikatan
                if self.enable_tabling:
                    self.memo_table[(goal_str, repr(subst))] = True
                return True
            subst.restore(checkpoint)

        # Cek aturan
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
                    if not self._prove_dfs(prem_ground, local_subst, depth + 1, prem_node):
                        all_proven = False
                        break

                if all_proven:
                    node.success = True
                    node.rule = rule_std
                    node.children = premise_nodes
                    node.subst = local_subst
                    if self.enable_tabling:
                        self.memo_table[(goal_str, repr(subst))] = True
                    return True
            else:
                subst.restore(checkpoint)

        node.success = False
        node.failure_reason = "Tidak ada fakta atau aturan yang cocok"
        if self.enable_tabling:
            self.memo_table[(goal_str, repr(subst))] = False
        return False

    # ------------------------------------------------------------
    # Breadth-First Search (BFS) – implementasi benar dengan goal‑stack
    # ------------------------------------------------------------
    def _prove_bfs(self, query: Compound, initial_subst: Substitution, root_node: ProofNode) -> bool:
        """
        BFS menggunakan state (goal_stack, substitution, path).
        goal_stack adalah daftar goal yang harus dibuktikan (top = goal pertama).
        path merekam langkah untuk merekonstruksi pohon bukti.
        """
        from collections import deque
        queue = deque()
        initial_stack = [query]
        initial_path: List[Tuple[Optional[Rule], Optional[Compound], Substitution]] = []  # (rule, fact, subst)
        queue.append((initial_stack, initial_subst.copy(), initial_path))
        visited = set()

        while queue:
            goal_stack, subst, path = queue.popleft()

            # Tidak ada goal = semua terbukti
            if not goal_stack:
                self._build_proof_from_path(root_node, query, path)
                return True

            current_goal = goal_stack[0]
            remaining = goal_stack[1:]
            goal_ground = current_goal.substitute(subst)
            pred = goal_ground.predicate if goal_ground.is_compound() else goal_ground.name

            state_id = (str(goal_ground), repr(subst))
            if state_id in visited:
                continue
            visited.add(state_id)

            # Fakta
            for fact in self.facts.get(pred, []):
                new_subst = subst.copy()
                if Unification.unify(goal_ground, fact, new_subst):
                    new_stack = [g.substitute(new_subst) for g in remaining]
                    new_path = path + [(None, fact, new_subst.copy())]
                    queue.append((new_stack, new_subst, new_path))

            # Aturan
            for rule in self.rules_by_pred.get(pred, []):
                rule_std = self._standardize_rule(rule)
                new_subst = subst.copy()
                if Unification.unify(goal_ground, rule_std.conclusion, new_subst):
                    # Ganti goal pertama dengan premis‑premis aturan
                    extended_stack = [p.substitute(new_subst) for p in rule_std.premises] + remaining
                    new_path = path + [(rule_std, None, new_subst.copy())]
                    queue.append((extended_stack, new_subst, new_path))

        root_node.success = False
        root_node.failure_reason = "Tidak ditemukan solusi (BFS)"
        return False

    def _build_proof_from_path(self, root_node: ProofNode, original_query: Compound,
                               path: List[Tuple[Optional[Rule], Optional[Compound], Substitution]]):
        """
        Rekonstruksi pohon ProofNode dari jejak solusi BFS.
        path adalah urutan aplikasi aturan/fakta yang diperlukan.
        """
        if not path:
            root_node.success = True
            return

        # Rekonstruksi dengan memproses path secara rekursif (urutan kebalikan untuk DFS post‑order)
        # Untuk sederhana, kita tahu path pertama adalah langkah untuk original_query.
        # Kita bangun node saat ini sebagai root, lalu anak‑anaknya sesuai path.
        # Karena path hanya mencatat aturan/fakta yang diterapkan pada goal pertama tiap state,
        # kita bisa memakai struktur stack lokal.
        idx = 0
        # Kita akan membuat pemetaan goal -> node, mulai dari root.
        # Gunakan rekursi internal.
        def build(idx: int, goal: Compound, subst: Substitution, parent_node: ProofNode) -> int:
            if idx >= len(path):
                return idx
            rule, fact, step_subst = path[idx]
            if fact is not None:
                # Sukses dengan fakta
                parent_node.success = True
                parent_node.subst = step_subst
                return idx + 1
            elif rule is not None:
                # Aplikasi aturan
                parent_node.rule = rule
                parent_node.subst = step_subst
                next_idx = idx + 1
                prem_nodes = []
                # Bangun node untuk setiap premis (urutan sesuai path selanjutnya)
                for prem in rule.premises:
                    prem_ground = prem.substitute(step_subst)
                    child_node = ProofNode(prem_ground, rule, step_subst)
                    prem_nodes.append(child_node)
                    next_idx = build(next_idx, prem_ground, step_subst, child_node)
                parent_node.children = prem_nodes
                parent_node.success = all(c.success for c in prem_nodes)
                return next_idx
            else:
                raise ValueError("Path item tidak valid (rule dan fact kosong)")

        root_node.success = True
        try:
            build(0, original_query, Substitution(), root_node)
        except Exception:
            # Jika rekonstruksi gagal, cukup tandai sukses tanpa pohon detail
            root_node.success = True

    # ------------------------------------------------------------
    # Iterative Deepening Search
    # ------------------------------------------------------------
    def _prove_ids(self, goal: Compound, subst: Substitution, root_node: ProofNode, max_depth: int = 100) -> bool:
        for depth in range(1, max_depth + 1):
            self.memo_table.clear()
            if self._prove_dfs_depth_limit(goal, subst, 0, root_node, depth):
                return True
        return False

    def _prove_dfs_depth_limit(self, goal: Compound, subst: Substitution, depth: int,
                               node: ProofNode, limit: int) -> bool:
        if depth > limit:
            node.failure_reason = "Batas kedalaman tercapai"
            return False
        # Implementasi sama dengan DFS namun memakai limit
        goal_ground = goal.substitute(subst)
        pred = goal_ground.predicate if goal_ground.is_compound() else goal_ground.name

        for fact in self.facts.get(pred, []):
            checkpoint = subst.snapshot()
            if Unification.unify(goal_ground, fact, subst):
                node.success = True
                subst.restore(checkpoint)
                return True
            subst.restore(checkpoint)

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
                    if not self._prove_dfs_depth_limit(prem_ground, local_subst, depth + 1, prem_node, limit):
                        all_proven = False
                        break
                if all_proven:
                    node.success = True
                    node.rule = rule_std
                    node.children = premise_nodes
                    node.subst = local_subst
                    return True
            else:
                subst.restore(checkpoint)
        node.success = False
        return False

    def prove_negation(self, query_str: str) -> bool:
        return not self.reason(query_str)

    def get_chain(self) -> List[Dict[str, Any]]:
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
            self._flatten_proof_tree(child, chain, depth + 1)

    def print_proof_tree(self):
        if self.proof_root:
            print(self.proof_root.to_string())
        else:
            print("Belum ada hasil penalaran.")