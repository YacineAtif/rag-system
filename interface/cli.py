"""
Interactive command-line interface for the modular RAG system.
Supports basic commands for status, health checks, and processing mode control.
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from backend.config import Config
    from backend.health_check import HealthChecker
    from processing.hybrid_pipeline import HybridPipeline, ProcessingMode
except Exception as e:
    print(f"Failed to import backend modules: {e}")
    raise


class RAGCLI:
    """Simple interactive CLI for the RAG system."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.pipeline = HybridPipeline(self.config)
        self.health_checker = HealthChecker(self.config)
        self.current_mode = ProcessingMode.HYBRID_AUTO
        self.pipeline.initialize()

    def run(self) -> None:
        """Run the interactive command loop."""
        print("\n🤖 RAG CLI - type /help for commands")
        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue

                if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
                    print("Bye!")
                    break
                elif user_input.lower() == "/help":
                    self.show_help()
                elif user_input.lower() == "/status":
                    self.show_status()
                elif user_input.lower() == "/health":
                    self.show_health()
                elif user_input.lower().startswith("/mode"):
                    self.set_mode(user_input)
                else:
                    self.answer_query(user_input)
            except KeyboardInterrupt:
                print("\nInterrupted")
                break

    def show_help(self) -> None:
        print("\nAvailable commands:")
        print("  /status           Show pipeline status")
        print("  /health           Run health checks")
        print("  /mode <name>      Set processing mode")
        print("  /help             Show this help message")
        print("  /quit             Exit the CLI")

    def show_status(self) -> None:
        status = self.pipeline.get_status()
        print("\nPipeline Status:")
        for key, val in status.items():
            print(f"  {key}: {val}")
        print(f"Current mode: {self.current_mode.value}")

    def show_health(self) -> None:
        results = self.health_checker.full_system_check()
        print("\nHealth Check:")
        for name, result in results.items():
            if name == "overall":
                continue
            print(f"  {name}: {result['status']} - {result['message']}")
        if "overall" in results:
            overall = results["overall"]
            print(f"Overall: {overall['status']} - {overall['message']}")

    def set_mode(self, cmd: str) -> None:
        parts = cmd.split()
        if len(parts) < 2:
            print("Usage: /mode <" + "|".join(m.value for m in ProcessingMode) + ">")
            return
        mode_name = parts[1].lower()
        for mode in ProcessingMode:
            if mode.value == mode_name:
                self.current_mode = mode
                print(f"Mode set to {mode.value}")
                return
        print(f"Unknown mode: {mode_name}")

    def answer_query(self, query: str) -> None:
        # Placeholder contexts for demonstration
        contexts = ["This is a sample context for the query."]
        result = self.pipeline.process_query(query, contexts, self.current_mode)
        print(f"\nAnswer: {result.answer}")
        print(f"Confidence: {result.confidence:.3f}")


def main() -> None:
    cli = RAGCLI()
    cli.run()


if __name__ == "__main__":
    main()
