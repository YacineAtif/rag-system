from __future__ import annotations

"""Neo4j based knowledge graph utilities."""

from typing import Iterable, Tuple

from neo4j import GraphDatabase

from .config import Config


class KnowledgeGraph:
    """Simple wrapper around a Neo4j database for document entities."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.driver = GraphDatabase.driver(
            self.config.neo4j.uri,
            auth=(self.config.neo4j.user, self.config.neo4j.password),
        )

    def close(self) -> None:
        self.driver.close()

    def ingest_entities(self, triples: Iterable[Tuple[str, str, str, str, str]]) -> None:
        """Ingest (source_type, source_name, rel, target_type, target_name)."""
        with self.driver.session() as session:
            for s_type, s_name, rel, t_type, t_name in triples:
                query = (
                    f"MERGE (a:{s_type} {{name:$s_name}}) "
                    f"MERGE (b:{t_type} {{name:$t_name}}) "
                    f"MERGE (a)-[:{rel}]->(b)"
                )
                session.run(query, s_name=s_name, t_name=t_name)

    def query(self, cypher: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(cypher)
            return [record.data() for record in result]
