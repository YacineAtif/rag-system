"""
Enhanced test runner with detailed reporting.
"""

import unittest
import sys
from pathlib import Path


def run_all_tests():
    """Run all available tests with comprehensive reporting."""
    print("\U0001F9EA RAG System Test Suite")
    print("=" * 50)

    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')

    test_count = suite.countTestCases()
    print(f"Discovered {test_count} tests")

    runner = unittest.TextTestRunner(verbosity=2, buffer=True)

    print("\nRunning tests...")
    print("-" * 50)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, tb in result.failures:
            print(f"  {test}: {tb.splitlines()[-1] if tb else ''}")

    if result.errors:
        print("\nErrors:")
        for test, tb in result.errors:
            print(f"  {test}: {tb.splitlines()[-1] if tb else ''}")

    if result.skipped:
        print("\nSkipped:")
        for test, reason in result.skipped:
            print(f"  {test}: {reason}")

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
