"""
Samre - Autonomous AI Agent (v2.2)

This version adds a global exception handler to the main loop to prevent crashes
from unhandled exceptions anywhere in the agent's lifecycle.
"""

import sys
import os
import time
import argparse
import traceback # For detailed error logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.flock_of_thought import FlockOfThought

class AutonomousAgent:
    """The main class that runs Samre's autonomous lifecycle."""

    def __init__(self):
        self.flock = None
        self.running = True

    def start(self, cycle_delay: float = 5.0):
        """Starts the main autonomous loop with robust error handling."""
        print("🚀 Starting Samre's autonomous loop. Press Ctrl+C to stop.")
        try:
            # Initialize the core component within the try block
            self.flock = FlockOfThought()

            while self.running:
                self.flock.run_cycle()
                print(f"Waiting for {cycle_delay} seconds...")
                time.sleep(cycle_delay)

        except KeyboardInterrupt:
            print("\n🛑 Loop interrupted by user.")
        except Exception as e:
            # INTEGRATION: Global exception handler for the entire application
            print("🔥🔥 A FATAL UNHANDLED EXCEPTION OCCURRED 🔥🔥")
            print(f"Error: {e}")
            print("--- Stack Trace ---")
            traceback.print_exc()
            print("---------------------")
            print("The agent cannot continue and will now shut down.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Handles the graceful shutdown process, including saving state."""
        print("👋 Shutting down...")
        self.running = False
        if self.flock:
            # Only try to save state if the flock was successfully initialized
            self.flock.save_state()
        print("✅ Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Run the Samre Autonomous Agent.")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between cycles.")
    args = parser.parse_args()

    agent = AutonomousAgent()
    agent.start(cycle_delay=args.delay)

if __name__ == "__main__":
    main()
