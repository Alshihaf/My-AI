# tools/geology_actuator.py
from tools.geology_api import MacrostratAPI
from typing import TYPE_CHECKING, Tuple, List

if TYPE_CHECKING:
    from core.flock_of_thought import FlockOfThought

class GeologyActuator:
    def __init__(self, flock: 'FlockOfThought'):
        self.api = MacrostratAPI()
        self.flock = flock

    def learn_from_location(self, lat: float, lng: float, label: str, return_keywords: bool = False) -> Tuple[bool, List[str]]:
        """
        Fetches geological data for a location, processes it into the Samantic Garden,
        and optionally returns the keywords from the learned data.
        """
        print(f"⛰️ GEOLOGY_LEARN: Learning from location ({lat}, {lng}).")
        try:
            units = self.api.search_units(lat=lat, lng=lng, limit=5)
            if not units:
                print("    ⚠️ No geological units found at this location.")
                return False, []

            all_keywords = []
            for unit in units:
                # Create a descriptive text string from the API data
                text = f"Geological Unit at {label}. Name: {unit.get('unit_name', 'N/A')}. " \
                       f"Lithology: {unit.get('lith', 'N/A')}. Description: {unit.get('descrip', 'N/A')}. " \
                       f"Age: {unit.get('age', 'N/A')}."
                
                # Use the flock's central processing method to ingest the knowledge
                # and get the associated keywords.
                source_id = f"geo_unit_{unit.get('unit_id', 'unknown')}"
                keywords = self.flock.process_and_store(text, source_id)
                all_keywords.extend(keywords)

            print(f"    ✅ Successfully processed {len(units)} units. Total keywords: {len(all_keywords)}.")
            if return_keywords:
                return True, list(set(all_keywords)) # Return unique keywords
            return True, []

        except Exception as e:
            print(f"    ❌ CRITICAL an error during geology learning: {e}")
            return False, []
