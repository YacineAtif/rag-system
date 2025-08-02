import unittest
from unittest.mock import Mock, call, patch

from backend.knowledge_graph import (
    build_knowledge_graph,
    query_knowledge_graph,
    hybrid_retrieval,
)


class TestKnowledgeGraph(unittest.TestCase):
    def test_build_knowledge_graph_calls(self):
        mock_session = Mock()
        mock_context = Mock(__enter__=Mock(return_value=mock_session), __exit__=Mock(return_value=None))
        mock_driver = Mock()
        mock_driver.session.return_value = mock_context

        triples = [("Alice", "knows", "Bob")]
        build_knowledge_graph(triples, mock_driver)

        self.assertIn(
            call("MERGE (a:Entity {name: $name})", name="Alice"),
            mock_session.run.call_args_list,
        )
        self.assertIn(
            call("MERGE (b:Entity {name: $name})", name="Bob"),
            mock_session.run.call_args_list,
        )
        self.assertTrue(
            any("MERGE (a)-[:KNOWS]->(b)" in c.args[0] for c in mock_session.run.call_args_list)
        )

    def test_query_knowledge_graph(self):
        mock_session = Mock()
        mock_session.run.return_value = [
            {"name": "Bob"},
            {"name": "Carol"},
        ]
        mock_context = Mock(__enter__=Mock(return_value=mock_session), __exit__=Mock(return_value=None))
        mock_driver = Mock()
        mock_driver.session.return_value = mock_context

        results = query_knowledge_graph("Alice", mock_driver)

        self.assertEqual(results, ["Bob", "Carol"])
        mock_session.run.assert_called_with(
            "MATCH (a:Entity {name: $name})-->(b:Entity) RETURN b.name AS name",
            name="Alice",
        )

    def test_hybrid_retrieval_merge(self):
        mock_driver = Mock()
        with patch("backend.knowledge_graph.query_knowledge_graph", return_value=["X", "Y"]):
            results = hybrid_retrieval("query", ["Y", "Z"], mock_driver, top_k=10)
        self.assertEqual(results, ["X", "Y", "Z"])


if __name__ == "__main__":
    unittest.main()
