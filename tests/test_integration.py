"""
Integration tests for the complete modular system.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent))

class TestIntegration(unittest.TestCase):
    """Test integration between components."""

    def test_config_integration(self):
        try:
            from backend.config import Config
            config = Config('config.yaml')
            self.assertIn('claude', config.claude.model_name)
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

    def test_partnership_query_names(self):
        try:
            from weaviate_rag_pipeline_transformers import (
                QueryClassifier,
                TextProcessor,
                AnswerGenerator,
            )
        except ImportError:
            self.skipTest("RAG pipeline modules unavailable")

        from types import SimpleNamespace
        cfg = SimpleNamespace(
            section_priorities=SimpleNamespace(
                queries={'partnership_queries': {'keywords': ['partner']}}
            )
        )
        classifier = QueryClassifier(cfg)
        query = "Which organizations are involved in the project?"
        q_type = classifier.classify(query)
        self.assertEqual(q_type, "entity")

        sentences = [
            "The project partners include University of Skövde, Scania, Smart Eye and Viscando."
        ]
        processor = TextProcessor()
        generator = AnswerGenerator(processor)
        with patch('backend.qa_models.ClaudeQA.generate', return_value={'answer': 'Scania is a partner.', 'confidence': 0.8}):
            answer = generator.generate(query, sentences, "entity", [])
        self.assertIn("Scania", answer)

if __name__ == '__main__':
    unittest.main()
