from unittest.mock import Mock, patch

from backend.config import Config
from processing.knowledge_graph import query_knowledge_graph


def test_query_knowledge_graph_uses_extracted_entity():
    config = Config()
    mock_session = Mock()
    mock_session.run.return_value = []
    mock_context = Mock(__enter__=Mock(return_value=mock_session), __exit__=Mock(return_value=None))
    mock_driver = Mock()
    mock_driver.session.return_value = mock_context
    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        query_knowledge_graph('what is Scania role', config)
    cypher = (
        "MATCH (n)-[r*1..2]-(m) "
        "WHERE toLower(n.name) CONTAINS toLower($q) "
        "RETURN n, m LIMIT 20"
    )
    mock_session.run.assert_called_once_with(cypher, q='Scania')
