# tools/geology_api.py
import requests
from typing import List, Dict, Optional

class MacrostratAPI:
    """Jembatan antara Samre dan data geologi dunia nyata via Macrostrat API v2."""
    
    BASE_URL = "https://macrostrat.org/api/v2"
    
    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """Fungsi internal untuk melakukan GET request."""
        try:
            response = requests.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Macrostrat API Error: {e}")
            return None

    def search_units(self, lithology: str = None, lat: float = None, lng: float = None, interval_name: str = None, limit: int = 10) -> List[Dict]:
        """
        Mencari unit geologi berdasarkan litologi dan koordinat.
        Ini akan jadi 'makanan' bagi SamanticGarden.
        """
        params = {"limit": limit}
        if lithology:
            params["lith"] = lithology
        if lat and lng:
            params["lat"] = lat
            params["lng"] = lng
            params["adjacents"] = "true"
        if interval_name:
            params["interval_name"] = interval_name
            
        data = self._get("units", params)
        return data.get("success", {}).get("data", []) if data else []

    def get_column_data(self, col_id: int) -> Dict:
        """
        Mengambil seluruh kolom stratigrafi berdasarkan ID.
        Berguna sebagai data pelatihan untuk prediksi wellbore.
        """
        params = {"col_id": col_id, "response": "long", "format": "json"}
        data = self._get("columns", params)
        return data.get("success", {}).get("data", {}) if data else {}

    def query_geologic_map(self, lat: float, lng: float, radius_km: float = 10.0) -> Dict:
        """
        Mengambil data geologi permukaan di sekitar titik pengeboran.
        Bisa digunakan sebagai fitur tambahan dalam model prediktif.
        """
        # Endpoint map menggunakan /geologic_units/map
        params = {"lat": lat, "lng": lng, "radius": radius_km}
        data = self._get("geologic_units/map", params)
        return data.get("success", {}).get("data", {}) if data else {}
