# tools/geology_actuator.py
import random
from typing import TYPE_CHECKING, List, Tuple, Optional

from tools.geology_api import MacrostratAPI

if TYPE_CHECKING:
    from core.flock_of_thought import FlockOfThought

class GeologyActuator:
    """
    Aktuator untuk aksi geologi: GEOLOGY_EXPLORE dan GEOLOGY_LEARN.
    Menggunakan MacrostratAPI untuk mengambil data dunia nyata.
    """
    def __init__(self, flock: 'FlockOfThought', 
                 default_lat: float = -6.2088,   # Jakarta sebagai default
                 default_lng: float = 106.8456,
                 default_label: str = "Default Location"):
        self.api = MacrostratAPI()
        self.flock = flock
        self.current_lat = default_lat
        self.current_lng = default_lng
        self.current_label = default_label

        # Daftar seed lokasi untuk eksplorasi (bisa diperluas)
        self.exploration_targets = [
            (-6.1754, 106.8272, "Monas"),
            (-7.2504, 112.7688, "Surabaya"),
            (3.5952, 98.6722, "Medan"),
            (-5.1477, 119.4327, "Makassar"),
            (-0.9471, 100.4172, "Padang"),
            # Tambahkan koordinat menarik lainnya
        ]

    def set_target_location(self, lat: float, lng: float, label: str):
        """
        Mengatur lokasi target untuk pembelajaran berikutnya.
        Dipanggil oleh Planner atau strategi tingkat tinggi.
        """
        self.current_lat = lat
        self.current_lng = lng
        self.current_label = label
        print(f"📍 GeologyActuator lokasi diatur ke: {label} ({lat}, {lng})")

    def execute(self, action: str) -> bool:
        """
        Menjalankan aksi geologi.
        
        Args:
            action: "GEOLOGY_EXPLORE" atau "GEOLOGY_LEARN"
            
        Returns:
            True jika berhasil, False jika gagal.
        """
        if action == "GEOLOGY_EXPLORE":
            return self._explore()
        elif action == "GEOLOGY_LEARN":
            return self._learn()
        else:
            print(f"⚠️ GeologyActuator: Aksi tidak dikenal '{action}'")
            return False

    def _explore(self) -> bool:
        """
        Mensimulasikan eksplorasi: memilih lokasi baru secara acak dari daftar,
        kemudian langsung mempelajarinya.
        """
        print("🗺️ GEOLOGY_EXPLORE: Mencari region geologi baru...")
        if not self.exploration_targets:
            print("    ⚠️ Tidak ada target eksplorasi tersedia.")
            return False

        lat, lng, label = random.choice(self.exploration_targets)
        self.set_target_location(lat, lng, label)
        print(f"    → Menjelajahi: {label} ({lat}, {lng})")
        # Setelah memilih lokasi, langsung belajar
        return self._learn()

    def _learn(self) -> bool:
        """
        Mengambil data geologi untuk lokasi saat ini dan memasukkannya ke
        Samantic Garden.
        """
        success, _ = self.learn_from_location(
            self.current_lat,
            self.current_lng,
            self.current_label
        )
        return success

    def learn_from_location(
        self, lat: float, lng: float, label: str, return_keywords: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Mengunduh data geologi dari API dan memprosesnya melalui flock.
        (Metode yang sudah ada, tanpa perubahan besar)
        """
        print(f"⛰️ GEOLOGY_LEARN: Mengambil data untuk {label} ({lat}, {lng})...")
        try:
            units = self.api.search_units(lat=lat, lng=lng, limit=5)
            if not units:
                print("    ⚠️ Tidak ada unit geologi ditemukan di lokasi ini.")
                return False, []

            all_keywords = []
            for unit in units:
                text = (
                    f"Geological Unit at {label}. "
                    f"Name: {unit.get('unit_name', 'N/A')}. "
                    f"Lithology: {unit.get('lith', 'N/A')}. "
                    f"Description: {unit.get('descrip', 'N/A')}. "
                    f"Age: {unit.get('age', 'N/A')}."
                )
                source_id = f"geo_unit_{unit.get('unit_id', 'unknown')}"
                keywords = self.flock.process_and_store(text, source_id)
                all_keywords.extend(keywords)

            unique_keywords = list(set(all_keywords))
            print(f"    ✅ Diproses {len(units)} unit. Kata kunci: {unique_keywords}")
            if return_keywords:
                return True, unique_keywords
            return True, []

        except Exception as e:
            print(f"    ❌ Gagal mengambil data geologi: {e}")
            return False, []