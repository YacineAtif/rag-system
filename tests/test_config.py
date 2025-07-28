"""
Tests for the configuration system.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from backend.config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

@unittest.skipUnless(CONFIG_AVAILABLE, "Config module not available")
class TestConfig(unittest.TestCase):
    """Test configuration functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config("nonexistent.yaml")

        # Test basic attributes exist
        self.assertTrue(hasattr(config, 'documents_folder'))
        self.assertTrue(hasattr(config, 'chunk_size'))
        self.assertTrue(hasattr(config, 'development'))

        # Test reasonable defaults
        self.assertIsInstance(config.chunk_size, int)
        self.assertGreater(config.chunk_size, 0)

    def test_validation(self):
        """Test configuration validation."""
        config = Config("nonexistent.yaml")

        # Should return list of errors
        errors = config.validate()
        self.assertIsInstance(errors, list)

if __name__ == '__main__':
    unittest.main()
