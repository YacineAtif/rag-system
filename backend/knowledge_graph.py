
from typing import List, Tuple
from neo4j import Driver


def build_knowledge_graph(triples: List[Tuple[str, str, str]], driver: Driver) -> None:
    """Create nodes and relationships in Neo4j from triples."""
    with driver.session() as session:
        for subj, rel, obj in triples:
            session.run("MERGE (a:Entity {name: $name})", name=subj)
            session.run("MERGE (b:Entity {name: $name})", name=obj)
            session.run(
                "MATCH (a:Entity {name: $subj}) "
                "MATCH (b:Entity {name: $obj}) "
                f"MERGE (a)-[:{rel.upper()}]->(b)",
                subj=subj,
                obj=obj,
            )


def query_knowledge_graph(entity: str, driver: Driver) -> List[str]:
    """Return connected entity names for the given entity."""
    with driver.session() as session:
        result = session.run(
            "MATCH (a:Entity {name: $name})-->(b:Entity) RETURN b.name AS name",
            name=entity,
        )
        return [record["name"] for record in result]


def hybrid_retrieval(query: str, vector_results: List[str], driver: Driver, top_k: int = 5) -> List[str]:
    """Combine graph query results with vector search results and deduplicate."""
    graph_results = query_knowledge_graph(query, driver)
    combined = []
    for item in graph_results + vector_results:
        if item not in combined:
            combined.append(item)
        if len(combined) >= top_k:
            break
    return combined

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

