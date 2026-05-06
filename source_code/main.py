"""
Samre - Autonomous AI Agent

This is the main entry point for running the autonomous agent loop.
"""

import sys
import os
import time
import argparse

# Add the project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.flock_of_thought import FlockOfThought

class AutonomousAgent:
    """The main class that runs Samre's autonomous lifecycle."""

    def __init__(self):
        self.flock = FlockOfThought()
        self.running = True

    def start(self, cycle_delay: float = 5.0):
        """Starts the main autonomous loop."""
        print("🚀 Starting Samre's autonomous loop. Press Ctrl+C to stop.")
        try:
            while self.running:
                self.flock.run_cycle()
                print(f"Waiting for {cycle_delay} seconds...")
                time.sleep(cycle_delay)
        except KeyboardInterrupt:
            print("\n🛑 Loop interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Handles the shutdown process."""
        print("👋 Shutting down...")
        self.running = False
        print("✅ Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Run the Samre Autonomous Agent.")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between cycles.")
    args = parser.parse_args()

    agent = AutonomousAgent()
    agent.start(cycle_delay=args.delay)

if __name__ == "__main__":
    main()
