"""
Tests for the health checking system.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from backend.health_check import HealthChecker
    from backend.config import Config
    HEALTH_CHECK_AVAILABLE = True
except ImportError:
    HEALTH_CHECK_AVAILABLE = False

@unittest.skipUnless(HEALTH_CHECK_AVAILABLE, "HealthChecker not available")
class TestHealthChecker(unittest.TestCase):
    """Test health checking functionality."""

    def setUp(self):
        self.config = Config()
        self.health_checker = HealthChecker(self.config)

    def test_health_checker_initialization(self):
        self.assertIsNotNone(self.health_checker)
        self.assertIsNotNone(self.health_checker.config)

    def test_configuration_check(self):
        result = self.health_checker.check_configuration()
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('message', result)
        self.assertIn(result['status'], ['healthy', 'warning', 'error'])

    def test_documents_folder_check(self):
        result = self.health_checker.check_documents_folder()
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('message', result)
        self.assertIn('details', result)

    def test_dependencies_check(self):
        result = self.health_checker.check_dependencies()
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('details', result)
        details = result['details']
        self.assertIn('by_category', details)

    def test_full_system_check(self):
        results = self.health_checker.full_system_check()
        self.assertIsInstance(results, dict)
        expected_checks = ['configuration', 'documents_folder', 'dependencies', 'legacy_scripts']
        for check in expected_checks:
            if check in results:
                self.assertIn('status', results[check])
        self.assertIn('overall', results)
        overall = results['overall']
        self.assertIn('status', overall)
        self.assertIn(overall['status'], ['healthy', 'warning', 'error'])

if __name__ == '__main__':
    unittest.main()
