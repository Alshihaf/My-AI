# core/imagination.py
"""
╔══════════════════════════════════════════════════════════════╗
║          IMAGINATION — Layered Simulation Engine             ║
║  Simulasi ribuan kemungkinan berlapis, efisien di mobile     ║
║  Kompatibel: Termux, Android, perangkat low-RAM              ║
╚══════════════════════════════════════════════════════════════╝

Arsitektur:
  ResourceMonitor   → Deteksi kapabilitas perangkat, adaptif
  SimulationOutcome → Unit data satu cabang simulasi
  SimulationEngine  → Ekspansi berlapis berbasis probabilitas
  OutcomeAnalyzer   → Statistik, risiko, rekomendasi
  SimulationStorage → Penyimpanan terkompresi (gzip JSONL)
  Imagination       → Antarmuka utama (simulate, batch, compare, stream)
"""

import logging
import os
import sys
import json
import gzip
import hashlib
import threading
import random
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from heapq import nlargest

logger = logging.getLogger(__name__)

__all__ = ["Imagination", "SimulationResult", "SimulationOutcome", "ResourceMonitor"]


# ══════════════════════════════════════════════════════════════
# Resource Monitor — Singleton, Mobile-Aware
# ══════════════════════════════════════════════════════════════

class ResourceMonitor:
    """
    Deteksi kapabilitas perangkat secara otomatis dan adaptif.
    Singleton — instansiasi hanya satu kali per proses.
    """

    _instance: Optional["ResourceMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ResourceMonitor":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._cpu_count: int = self._detect_cpu()
        self._memory_mb: int = self._detect_memory()
        self._is_mobile: bool = self._detect_mobile()
        logger.info(
            "ResourceMonitor: CPU=%d, RAM≈%dMB, Mobile=%s",
            self._cpu_count, self._memory_mb, self._is_mobile,
        )

    # ── Deteksi ──────────────────────────────────────────────

    @staticmethod
    def _detect_cpu() -> int:
        try:
            return max(1, (os.cpu_count() or 2) - 1)  # sisakan 1 core untuk OS
        except Exception:
            return 1

    @staticmethod
    def _detect_memory() -> int:
        """Estimasi RAM tersedia dalam MB."""
        try:
            if sys.platform.startswith("linux"):
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemAvailable" in line:
                            return int(line.split()[1]) // 1024
        except Exception:
            pass
        return 512  # fallback konservatif

    def _detect_mobile(self) -> bool:
        """Heuristik: Termux atau RAM < 1GB → perlakukan sebagai mobile."""
        indicators = [
            bool(os.environ.get("TERMUX_VERSION")),
            os.environ.get("PREFIX", "").startswith("/data"),
            os.path.exists("/data/data/com.termux"),
            self._memory_mb < 1024,
        ]
        return any(indicators)

    # ── Profil ───────────────────────────────────────────────

    @property
    def max_workers(self) -> int:
        return 2 if self._is_mobile else min(4, self._cpu_count)

    @property
    def max_outcomes_per_layer(self) -> int:
        """Batas cabang aktif per lapisan berdasarkan RAM tersedia."""
        if self._memory_mb < 512:
            return 50
        if self._memory_mb < 1024:
            return 200
        if self._memory_mb < 2048:
            return 500
        return 1000

    @property
    def recommended_max_layers(self) -> int:
        return 5 if self._is_mobile else 8

    def get_profile(self) -> Dict[str, Any]:
        return {
            "cpu_count": self._cpu_count,
            "memory_mb": self._memory_mb,
            "is_mobile": self._is_mobile,
            "max_workers": self.max_workers,
            "max_outcomes_per_layer": self.max_outcomes_per_layer,
            "recommended_max_layers": self.recommended_max_layers,
        }


# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════

@dataclass
class SimulationOutcome:
    """
    Satu cabang/node dalam pohon simulasi.

    weighted_score = probability × impact_score × confidence
    Digunakan sebagai metrik utama pruning dan ranking.
    """

    outcome_id: str
    description: str
    probability: float       # 0.0 – 1.0
    impact_score: float      # -1.0 (buruk) – 1.0 (baik)
    confidence: float        # keyakinan engine, 0.0 – 1.0
    layer: int               # kedalaman dari root
    parent_id: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)  # ID saja, hemat memori
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return self.probability * self.impact_score * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["weighted_score"] = round(self.weighted_score, 6)
        return d

    def __repr__(self) -> str:
        return (
            f"<Outcome layer={self.layer} prob={self.probability:.2f} "
            f"impact={self.impact_score:+.2f} score={self.weighted_score:+.4f}>"
        )


@dataclass
class SimulationResult:
    """Hasil lengkap satu skenario simulasi."""

    scenario_id: str
    scenario: str
    parameters: Dict[str, Any]
    timestamp: str
    total_outcomes: int
    layers_explored: int
    top_outcomes: List[Dict]
    risk_assessment: Dict[str, float]
    recommended_actions: List[str]
    simulation_time_ms: float
    resource_usage: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Ringkasan teks hasil simulasi."""
        lines = [
            f"📊 Skenario   : {self.scenario}",
            f"🔢 Outcomes   : {self.total_outcomes:,}",
            f"📚 Lapisan    : {self.layers_explored}",
            f"⚡ Waktu      : {self.simulation_time_ms:.1f} ms",
            "",
            "🏆 Top Outcomes:",
        ]
        for i, o in enumerate(self.top_outcomes[:5], 1):
            prob = o.get("probability", 0)
            desc = o.get("description", "")[:70]
            lines.append(f"  {i}. [{prob:.0%}] {desc}")

        ra = self.risk_assessment
        lines += [
            "",
            f"⚠️  Risiko     : rata={ra.get('rata_rata', 0):+.3f}, "
            f"variansi={ra.get('variansi', 0):.3f}, "
            f"rasio_p/n={ra.get('rasio_p_n', 0):.1f}",
            "",
            "✅ Rekomendasi:",
        ]
        for action in self.recommended_actions:
            lines.append(f"  • {action}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Simulation Engine — Generator-based, Memory Efficient
# ══════════════════════════════════════════════════════════════

class SimulationEngine:
    """
    Mesin simulasi probabilistik berlapis.

    Strategi ekspansi:
    - Setiap outcome induk melahirkan `branch_factor` anak
    - Branch_factor menyusut adaptif per lapisan (cegah ledakan eksponensial)
    - Hanya top-N outcome terbaik yang diteruskan ke lapisan berikutnya (pruning)
    - Semua ekspansi via generator → tidak memuat semua data sekaligus ke RAM
    """

    _PREFIXES = [
        "Jika kondisi terpenuhi",
        "Dalam skenario alternatif",
        "Konsekuensi langsung",
        "Efek berantai",
        "Dampak tidak langsung",
        "Kemungkinan paradoks",
        "Jalur optimal ditemukan",
        "Titik kritis teridentifikasi",
        "Pola emergent muncul",
        "Interaksi tak terduga",
    ]
    _SUFFIXES = [
        "membentuk pola baru",
        "mengubah kondisi awal",
        "membuka jalur alternatif",
        "menutup skenario lain",
        "memperkuat tren utama",
        "membalikkan asumsi dasar",
        "menciptakan bifurkasi baru",
        "menstabilkan sistem",
    ]

    def __init__(self, resource_monitor: ResourceMonitor) -> None:
        self.rm = resource_monitor
        self._registry: Dict[str, SimulationOutcome] = {}
        self._lock = threading.Lock()

    # ── Utilities ────────────────────────────────────────────

    @staticmethod
    def _make_id(scenario: str, layer: int, index: int, parent_id: str = "") -> str:
        raw = f"{scenario}|{layer}|{index}|{parent_id}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _generate_description(
        self,
        scenario: str,
        layer: int,
        index: int,
        parameters: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        prefix = self._PREFIXES[index % len(self._PREFIXES)]
        suffix = rng.choice(self._SUFFIXES)
        param_hint = ""
        if parameters:
            key = rng.choice(list(parameters.keys()))
            val = parameters[key]
            param_hint = f" [{key}={str(val)[:20]}]"
        scenario_snippet = scenario[:40].rstrip()
        return f"{prefix}{param_hint}: «{scenario_snippet}» → {suffix} (L{layer})"

    @staticmethod
    def _sample_conditions(
        parameters: Dict[str, Any], rng: random.Random
    ) -> Dict[str, Any]:
        if not parameters:
            return {}
        items = list(parameters.items())
        n = max(1, len(items) // 2)
        return dict(rng.sample(items, min(n, len(items))))

    # ── Core Expansion ───────────────────────────────────────

    def _expand_layer(
        self,
        scenario: str,
        layer: int,
        parents: List[SimulationOutcome],
        branch_factor: int,
        parameters: Dict[str, Any],
        rng: random.Random,
    ) -> Generator[SimulationOutcome, None, None]:
        """
        Generator: ekspansi satu lapisan dari semua parent.
        Distribusi probabilitas: multinomial, total <= parent.probability.
        """
        for parent in parents:
            pool = parent.probability
            for i in range(branch_factor):
                is_last = i == branch_factor - 1
                if is_last:
                    child_prob = max(0.01, pool)
                else:
                    child_prob = rng.uniform(0.05, pool * 0.55)
                    pool -= child_prob

                child_prob = min(max(child_prob, 0.001), 1.0)

                # Impact mengikuti distribusi Gaussian terpusat di parent
                raw_impact = rng.gauss(parent.impact_score * 0.65, 0.25)
                impact = max(-1.0, min(1.0, raw_impact))

                # Confidence menyusut per lapisan
                confidence = parent.confidence * rng.uniform(0.72, 0.96)

                child_id = self._make_id(scenario, layer, i, parent.outcome_id)

                child = SimulationOutcome(
                    outcome_id=child_id,
                    description=self._generate_description(
                        scenario, layer, i, parameters, rng
                    ),
                    probability=round(child_prob, 6),
                    impact_score=round(impact, 6),
                    confidence=round(confidence, 6),
                    layer=layer,
                    parent_id=parent.outcome_id,
                    conditions=self._sample_conditions(parameters, rng),
                    metadata={
                        "branch_index": i,
                        "depth_decay": round(1.0 / (layer + 1), 4),
                    },
                )

                with self._lock:
                    parent.children.append(child_id)
                    self._registry[child_id] = child

                yield child

    def simulate_layered(
        self,
        scenario: str,
        parameters: Dict[str, Any],
        n_layers: int,
        branch_factor: int,
        seed: Optional[int] = None,
    ) -> Tuple[List[SimulationOutcome], Dict[str, Any]]:
        """
        Jalankan simulasi berlapis penuh.

        Returns:
            (list_semua_outcomes, statistik_per_lapisan)
        """
        rng = random.Random(seed if seed is not None else time.time_ns())
        max_per_layer = self.rm.max_outcomes_per_layer

        # Root
        root_id = self._make_id(scenario, 0, 0)
        root = SimulationOutcome(
            outcome_id=root_id,
            description=f"[ROOT] {scenario[:80]}",
            probability=1.0,
            impact_score=0.0,
            confidence=1.0,
            layer=0,
            conditions=parameters,
        )
        with self._lock:
            self._registry[root_id] = root

        all_outcomes: List[SimulationOutcome] = [root]
        current_layer = [root]
        stats: Dict[str, Any] = {"per_layer": {0: 1}}

        for layer in range(1, n_layers + 1):
            # Branch factor adaptif: kurangi per lapisan agar tidak meledak
            adaptive_bf = max(2, branch_factor - (layer - 1))

            layer_outcomes: List[SimulationOutcome] = list(
                self._expand_layer(
                    scenario, layer, current_layer, adaptive_bf, parameters, rng
                )
            )

            # Pruning: hanya teruskan top-N berdasarkan |weighted_score|
            if len(layer_outcomes) > max_per_layer:
                layer_outcomes = nlargest(
                    max_per_layer,
                    layer_outcomes,
                    key=lambda o: abs(o.weighted_score),
                )

            stats["per_layer"][layer] = len(layer_outcomes)
            all_outcomes.extend(layer_outcomes)
            current_layer = layer_outcomes

            if not current_layer:
                logger.debug("Ekspansi berhenti di lapisan %d (tidak ada cabang)", layer)
                break

        stats["total"] = len(all_outcomes)
        stats["layers_actual"] = max(stats["per_layer"].keys())
        return all_outcomes, stats

    def clear_registry(self) -> None:
        """Bebaskan memori registry setelah simulasi selesai."""
        with self._lock:
            self._registry.clear()


# ══════════════════════════════════════════════════════════════
# Outcome Analyzer
# ══════════════════════════════════════════════════════════════

class OutcomeAnalyzer:
    """Analisis statistik, penilaian risiko, dan rekomendasi aksi."""

    def risk_assessment(
        self, outcomes: List[SimulationOutcome]
    ) -> Dict[str, float]:
        if not outcomes:
            return {}

        scores = [o.weighted_score for o in outcomes]
        n = len(scores)
        avg = sum(scores) / n
        variance = sum((s - avg) ** 2 for s in scores) / n
        negatives = [s for s in scores if s < 0]
        positives = [s for s in scores if s >= 0]

        return {
            "rata_rata": round(avg, 4),
            "variansi": round(variance, 4),
            "std_deviasi": round(variance ** 0.5, 4),
            "risiko_negatif": round(abs(sum(negatives)) / n, 4),
            "potensi_positif": round(sum(positives) / n, 4),
            "rasio_p_n": round(len(positives) / max(len(negatives), 1), 2),
            "total_sampel": n,
        }

    def top_outcomes(
        self, outcomes: List[SimulationOutcome], n: int = 10
    ) -> List[SimulationOutcome]:
        return nlargest(n, outcomes, key=lambda o: o.weighted_score)

    def worst_outcomes(
        self, outcomes: List[SimulationOutcome], n: int = 5
    ) -> List[SimulationOutcome]:
        """Outcome paling berisiko — berguna untuk analisis skenario terburuk."""
        return nlargest(n, outcomes, key=lambda o: -o.weighted_score)

    def recommend(
        self,
        outcomes: List[SimulationOutcome],
        risk: Dict[str, float],
    ) -> List[str]:
        recommendations: List[str] = []
        avg = risk.get("rata_rata", 0.0)
        variance = risk.get("variansi", 0.0)
        ratio = risk.get("rasio_p_n", 1.0)
        std = risk.get("std_deviasi", 0.0)

        # Arah utama
        if avg > 0.35:
            recommendations.append(
                "Lanjutkan dengan strategi agresif — mayoritas outcome menunjukkan dampak positif"
            )
        elif avg < -0.25:
            recommendations.append(
                "Tunda eksekusi: dominasi outcome negatif terdeteksi, cari kondisi penghambat"
            )
        else:
            recommendations.append(
                "Strategi moderat direkomendasikan — distribusi outcome mixed"
            )

        # Volatilitas
        if std > 0.4:
            recommendations.append(
                "Volatilitas sangat tinggi — siapkan contingency plan berlapis dan monitor ketat"
            )
        elif variance > 0.15:
            recommendations.append(
                "Variasi outcome signifikan — pertimbangkan hedging atau diversifikasi aksi"
            )
        else:
            recommendations.append(
                "Hasil relatif stabil — prediktabilitas tinggi, risiko kejutan rendah"
            )

        # Rasio positif/negatif
        if ratio > 4.0:
            recommendations.append(
                "Rasio positif sangat baik (>4:1) — momentum kuat, eksekusi prioritas"
            )
        elif ratio < 0.5:
            recommendations.append(
                "Rasio negatif dominan (<1:2) — tunggu perubahan kondisi eksternal"
            )

        # Skenario terburuk
        worst = self.worst_outcomes(outcomes, n=3)
        if worst:
            worst_desc = worst[0].description[:60]
            recommendations.append(
                f"Antisipasi skenario terburuk: «{worst_desc}...»"
            )

        return recommendations or ["Tidak cukup data untuk rekomendasi konkrit"]


# ══════════════════════════════════════════════════════════════
# Storage — Compressed JSONL (Gzip), Thread-Safe
# ══════════════════════════════════════════════════════════════

class SimulationStorage:
    """
    Penyimpanan hasil simulasi dalam format JSONL terkompresi gzip.

    Keunggulan vs plain text:
    - Ukuran file 5-10x lebih kecil
    - Mudah di-append (mode 'at')
    - Mudah dibaca line-per-line
    """

    _DEFAULT_FILENAME = "simulations.jsonl.gz"

    def __init__(self, base_path: str = "ai_core/memory/dreams") -> None:
        self.base_path = base_path
        self._write_lock = threading.Lock()
        os.makedirs(base_path, exist_ok=True)

    @property
    def _storage_path(self) -> str:
        return os.path.join(self.base_path, self._DEFAULT_FILENAME)

    def save(self, result: SimulationResult) -> None:
        entry = json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
        with self._write_lock:
            try:
                with gzip.open(self._storage_path, "at", encoding="utf-8") as f:
                    f.write(entry)
            except Exception as exc:
                logger.error("Gagal menyimpan simulasi: %s", exc)

    def load_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        results: List[Dict] = []
        try:
            with gzip.open(self._storage_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.error("Gagal membaca simulasi: %s", exc)
        return results[-n:]

    def get_scenario_hash(self, scenario: str, params: Dict[str, Any]) -> str:
        raw = json.dumps({"s": scenario, "p": params}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def storage_size_kb(self) -> float:
        try:
            return os.path.getsize(self._storage_path) / 1024
        except FileNotFoundError:
            return 0.0


# ══════════════════════════════════════════════════════════════
# Imagination — Antarmuka Utama
# ══════════════════════════════════════════════════════════════

class Imagination:
    """
    Modul Imajinasi: simulasi ribuan kemungkinan berlapis.

    Dirancang untuk:
    ✓ Mobile / Termux (resource-adaptive, generator-based)
    ✓ Simulasi tunggal, batch paralel, dan komparatif
    ✓ Streaming outcome satu per satu (ultra low-RAM)
    ✓ Cache in-memory untuk skenario berulang
    ✓ Penyimpanan terkompresi gzip

    Contoh penggunaan:
    ------------------
    imagination = Imagination()

    # Simulasi tunggal
    result = imagination.simulate(
        "Investasi di sektor energi terbarukan",
        parameters={"modal": "100jt", "horizon": "5thn"}
    )
    print(result.summary())

    # Batch paralel
    results = imagination.simulate_batch(
        ["Skenario A", "Skenario B", "Skenario C"]
    )

    # Perbandingan komparatif
    comparison = imagination.compare(["Strategi X", "Strategi Y"])

    # Streaming hemat memori (Termux-friendly)
    for outcome in imagination.stream_outcomes(
        "Ekspansi pasar baru",
        filter_fn=lambda o: o.impact_score > 0.3
    ):
        print(outcome)
    """

    def __init__(
        self,
        dream_module: Any = None,
        storage_path: str = "ai_core/memory/dreams",
        max_layers: Optional[int] = None,
        branch_factor: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self.dream = dream_module
        self.rm = ResourceMonitor()
        self.engine = SimulationEngine(self.rm)
        self.analyzer = OutcomeAnalyzer()
        self.storage = SimulationStorage(storage_path)
        self.seed = seed
        self.branch_factor = branch_factor
        self.max_layers = max_layers or self.rm.recommended_max_layers
        self._cache: Dict[str, SimulationResult] = {}
        self._cache_lock = threading.Lock()
        logger.info("Imagination siap — profil: %s", self.rm.get_profile())

    def __repr__(self) -> str:
        return (
            f"<Imagination layers={self.max_layers} "
            f"branch={self.branch_factor} "
            f"mobile={self.rm._is_mobile}>"
        )

    # ── Simulasi Tunggal ─────────────────────────────────────

    def simulate(
        self,
        scenario: str,
        parameters: Dict[str, Any] = None,
        layers: Optional[int] = None,
        branch_factor: Optional[int] = None,
        use_cache: bool = True,
        persist: bool = True,
    ) -> SimulationResult:
        """
        Simulasi satu skenario secara berlapis.

        Args:
            scenario     : Deskripsi skenario (teks bebas)
            parameters   : Konteks / variabel tambahan
            layers       : Jumlah lapisan (default: auto dari profil perangkat)
            branch_factor: Jumlah cabang per node (default: 5)
            use_cache    : Gunakan cache untuk skenario identik
            persist      : Simpan hasil ke disk (gzip)

        Returns:
            SimulationResult — panggil .summary() untuk ringkasan teks
        """
        params = parameters or {}
        n_layers = min(
            layers or self.max_layers,
            self.rm.recommended_max_layers,
        )
        bf = branch_factor or self.branch_factor

        cache_key = self.storage.get_scenario_hash(scenario, params)
        if use_cache:
            with self._cache_lock:
                if cache_key in self._cache:
                    logger.debug("Cache hit: %s", cache_key)
                    return self._cache[cache_key]

        t0 = time.perf_counter()
        outcomes, stats = self.engine.simulate_layered(
            scenario, params, n_layers, bf, seed=self.seed
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        risk = self.analyzer.risk_assessment(outcomes)
        top = self.analyzer.top_outcomes(outcomes, n=10)
        recs = self.analyzer.recommend(outcomes, risk)

        result = SimulationResult(
            scenario_id=cache_key,
            scenario=scenario,
            parameters=params,
            timestamp=datetime.now().isoformat(),
            total_outcomes=stats["total"],
            layers_explored=stats.get("layers_actual", n_layers),
            top_outcomes=[o.to_dict() for o in top],
            risk_assessment=risk,
            recommended_actions=recs,
            simulation_time_ms=round(elapsed_ms, 2),
            resource_usage=self.rm.get_profile(),
        )

        if use_cache:
            with self._cache_lock:
                self._cache[cache_key] = result

        if persist:
            self.storage.save(result)

        # Bebaskan memori registry segera setelah selesai
        self.engine.clear_registry()

        return result

    # ── Simulasi Batch (Paralel Ringan) ──────────────────────

    def simulate_batch(
        self,
        scenarios: List[str],
        shared_parameters: Dict[str, Any] = None,
        per_scenario_params: Optional[List[Dict[str, Any]]] = None,
        max_workers: Optional[int] = None,
    ) -> List[SimulationResult]:
        """
        Simulasi banyak skenario secara paralel dengan ThreadPoolExecutor.

        Thread-based (bukan multiprocessing) agar ringan di mobile.
        Urutan hasil dijamin sesuai urutan input.

        Args:
            scenarios           : List skenario yang akan disimulasikan
            shared_parameters   : Parameter yang berlaku untuk semua skenario
            per_scenario_params : Override per skenario (list sejajar scenarios)
            max_workers         : Jumlah worker (default: auto dari profil)
        """
        workers = max_workers or self.rm.max_workers
        shared = shared_parameters or {}
        params_list = per_scenario_params or [{} for _ in scenarios]

        results: List[Optional[SimulationResult]] = [None] * len(scenarios)

        def _worker(idx: int, scenario: str, params: Dict) -> Tuple[int, SimulationResult]:
            merged = {**shared, **params}
            return idx, self.simulate(scenario, merged, persist=False)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, i, s, p): i
                for i, (s, p) in enumerate(zip(scenarios, params_list))
            }
            for future in as_completed(futures):
                try:
                    idx, res = future.result()
                    results[idx] = res
                except Exception as exc:
                    logger.error(
                        "Batch error pada skenario ke-%d: %s",
                        futures[future], exc
                    )

        return [r for r in results if r is not None]

    # ── Perbandingan Komparatif ───────────────────────────────

    def compare(
        self,
        scenarios: List[str],
        parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Simulasikan dan bandingkan beberapa skenario, kembalikan ranking.

        Returns:
            Dict berisi ranking, skenario terbaik/terburuk, dan analisis
        """
        results = self.simulate_batch(scenarios, parameters)
        ranked = sorted(
            results,
            key=lambda r: r.risk_assessment.get("rata_rata", 0),
            reverse=True,
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(results),
            "ranking": [
                {
                    "rank": i + 1,
                    "scenario": r.scenario,
                    "avg_score": round(r.risk_assessment.get("rata_rata", 0), 4),
                    "total_outcomes": r.total_outcomes,
                    "volatility": round(r.risk_assessment.get("std_deviasi", 0), 4),
                    "p_n_ratio": r.risk_assessment.get("rasio_p_n", 1.0),
                    "top_recommendation": (
                        r.recommended_actions[0] if r.recommended_actions else "-"
                    ),
                }
                for i, r in enumerate(ranked)
            ],
            "best_scenario": ranked[0].scenario if ranked else None,
            "worst_scenario": ranked[-1].scenario if ranked else None,
            "winner_recommendation": (
                ranked[0].recommended_actions[0]
                if ranked and ranked[0].recommended_actions
                else "-"
            ),
        }

    # ── Streaming (Ultra Low-RAM) ────────────────────────────

    def stream_outcomes(
        self,
        scenario: str,
        parameters: Dict[str, Any] = None,
        layers: Optional[int] = None,
        filter_fn: Optional[Callable[[SimulationOutcome], bool]] = None,
    ) -> Generator[SimulationOutcome, None, None]:
        """
        Stream outcomes satu per satu tanpa menyimpan semuanya di RAM.
        Sangat cocok untuk Termux / perangkat dengan RAM sangat terbatas.

        Args:
            filter_fn: Fungsi filter opsional, contoh:
                       ``lambda o: o.impact_score > 0.5``
                       ``lambda o: o.layer >= 3 and o.probability > 0.1``
        """
        params = parameters or {}
        n_layers = min(layers or self.max_layers, self.rm.recommended_max_layers)
        rng = random.Random(self.seed if self.seed is not None else time.time_ns())

        root = SimulationOutcome(
            outcome_id="root",
            description=f"[ROOT] {scenario[:80]}",
            probability=1.0,
            impact_score=0.0,
            confidence=1.0,
            layer=0,
            conditions=params,
        )
        if filter_fn is None or filter_fn(root):
            yield root

        current_layer = [root]
        max_keep = self.rm.max_outcomes_per_layer

        for layer in range(1, n_layers + 1):
            adaptive_bf = max(2, self.branch_factor - (layer - 1))
            next_layer: List[SimulationOutcome] = []

            for outcome in self.engine._expand_layer(
                scenario, layer, current_layer, adaptive_bf, params, rng
            ):
                if filter_fn is None or filter_fn(outcome):
                    yield outcome
                next_layer.append(outcome)

            # Pruning untuk lapisan berikutnya
            current_layer = (
                nlargest(max_keep, next_layer, key=lambda o: abs(o.weighted_score))
                if len(next_layer) > max_keep
                else next_layer
            )

            if not current_layer:
                break

        self.engine.clear_registry()

    # ── Utilitas ─────────────────────────────────────────────

    def get_recent_simulations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Ambil n hasil simulasi terakhir dari penyimpanan."""
        return self.storage.load_recent(n)

    def get_resource_profile(self) -> Dict[str, Any]:
        """Profil kapabilitas perangkat yang terdeteksi."""
        return self.rm.get_profile()

    def clear_cache(self) -> None:
        """Hapus cache in-memory."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cache imagination dibersihkan")

    def storage_info(self) -> Dict[str, Any]:
        """Info ukuran file penyimpanan."""
        return {
            "path": self.storage.base_path,
            "size_kb": round(self.storage.storage_size_kb(), 2),
        }
