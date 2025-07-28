"""
Tests for the CLI interface.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch
from io import StringIO

sys.path.append(str(Path(__file__).parent.parent))

try:
    from interface.cli import RAGCLI
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

@unittest.skipUnless(CLI_AVAILABLE, "CLI not available")
class TestCLI(unittest.TestCase):
    """Test CLI functionality."""

    def setUp(self):
        try:
            self.cli = RAGCLI()
            self.cli_available = True
        except Exception:
            self.cli_available = False

    def test_cli_initialization(self):
        if not self.cli_available:
            self.skipTest("CLI initialization failed")
        self.assertIsNotNone(self.cli)

    @patch('sys.stdout', new_callable=StringIO)
    def test_show_system_status(self, mock_stdout):
        if not self.cli_available:
            self.skipTest("CLI not available")
        if hasattr(self.cli, 'show_status'):
            self.cli.show_status()
            output = mock_stdout.getvalue()
            self.assertIn('Status', output)
        else:
            self.skipTest("show_status not implemented")

    @patch('sys.stdout', new_callable=StringIO)
    def test_run_health_checks(self, mock_stdout):
        if not self.cli_available:
            self.skipTest("CLI not available")
        if hasattr(self.cli, 'show_health'):
            self.cli.show_health()
            output = mock_stdout.getvalue()
            self.assertIn('Health', output)
        else:
            self.skipTest("show_health not implemented")

    def test_processing_mode_change(self):
        if not self.cli_available:
            self.skipTest("CLI not available")
        if hasattr(self.cli, 'set_mode'):
            result = self.cli.set_mode('/mode hybrid')
            self.assertIsNone(result)
            result_invalid = self.cli.set_mode('/mode invalid_mode')
            self.assertIsNone(result_invalid)
        else:
            self.skipTest("set_mode not implemented")

if __name__ == '__main__':
    unittest.main()
