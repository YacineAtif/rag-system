import unittest
from unittest.mock import patch, MagicMock

from backend.config import Config
from backend.knowledge_graph import KnowledgeGraph


class TestKnowledgeGraph(unittest.TestCase):
    def test_ingest_entities_calls_neo4j(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        with patch('backend.knowledge_graph.GraphDatabase.driver', return_value=mock_driver):
            kg = KnowledgeGraph(Config())
            triples = [
                ('Concept', 'AI', 'DEFINED_AS', 'Definition', 'Artificial Intelligence'),
                ('Theory', 'ML', 'DEVELOPED_BY', 'Person', 'Turing'),
            ]
            kg.ingest_entities(triples)
            self.assertEqual(mock_session.run.call_count, len(triples))
            kg.close()
            mock_driver.close.assert_called_once()

    def test_query_returns_data(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {'n': 1}
        mock_session.run.return_value = [mock_record]
        mock_driver.session.return_value.__enter__.return_value = mock_session
        with patch('backend.knowledge_graph.GraphDatabase.driver', return_value=mock_driver):
            kg = KnowledgeGraph(Config())
            res = kg.query('MATCH (n) RETURN n')
            self.assertEqual(res, [{'n': 1}])

