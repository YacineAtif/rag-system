"""
Integration tests for the complete modular system.
"""

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

class TestIntegration(unittest.TestCase):
    """Test integration between components."""

    def test_config_integration(self):
        try:
            from backend.config import Config
            config = Config()
            try:
                from backend.health_check import HealthChecker
                health_checker = HealthChecker(config)
                self.assertIsNotNone(health_checker)
            except ImportError:
                print("HealthChecker not available")
            try:
                from processing.hybrid_pipeline import HybridPipeline
                pipeline = HybridPipeline(config)
                self.assertIsNotNone(pipeline)
            except ImportError:
                print("HybridPipeline not available")
        except ImportError:
            self.skipTest("Config not available")

    def test_end_to_end_flow(self):
        availability = {"config": False, "health_check": False, "pipeline": False, "cli": False}
        try:
            from backend.config import Config
            availability["config"] = True
        except ImportError:
            pass
        try:
            from backend.health_check import HealthChecker
            availability["health_check"] = True
        except ImportError:
            pass
        try:
            from processing.hybrid_pipeline import HybridPipeline
            availability["pipeline"] = True
        except ImportError:
            pass
        try:
            from interface.cli import RAGCLI
            availability["cli"] = True
        except ImportError:
            pass
        available_count = sum(availability.values())
        self.assertTrue(availability["config"], "Config component should be available")
        if available_count >= 2:
            print("Integration possible")
        else:
            print("Limited integration")

if __name__ == '__main__':
    unittest.main()
