import unittest

from domain_concept_registry import DomainConceptRegistry
from graph_semantic_analyzer import GraphSemanticAnalyzer


class TestGraphSemanticAnalyzer(unittest.TestCase):
    def test_handles_non_string_results(self):
        registry = DomainConceptRegistry(["alice", "bob"])
        analyzer = GraphSemanticAnalyzer(registry)
        query = "Alice knows Bob"
        graph_results = [
            {"source": "Alice", "relationship": "knows", "target": "Bob"},
            ("Bob", "knows", "Charlie"),
            "Charlie -> knows -> Dana",
        ]

        score, matched, neighborhood = analyzer.analyze(query, graph_results)

        self.assertIsInstance(score, float)
        self.assertIn("alice", matched)
        self.assertIn("bob", matched)
        self.assertIn("charlie", [n.lower() for n in neighborhood])
        self.assertIn("dana", [n.lower() for n in neighborhood])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
