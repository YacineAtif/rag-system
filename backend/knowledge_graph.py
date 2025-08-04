
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    from neo4j import Driver, GraphDatabase
except Exception:  # pragma: no cover
    Driver = Any  # type: ignore

    class GraphDatabase:  # type: ignore
        @staticmethod
        def driver(*args, **kwargs):  # pragma: no cover - missing dependency
            raise ImportError("neo4j package required")


def build_knowledge_graph(triples: List[Tuple[str, str, str]], driver: Driver) -> None:
    """Create nodes and relationships in Neo4j from triples.

    Entities are merged first to ensure comprehensive coverage and avoid
    duplicate creation when multiple triples reference the same name.
    """
    with driver.session() as session:
        entities = {subj for subj, _, obj in triples} | {obj for _, _, obj in triples}
        for name in entities:
            session.run("MERGE (a:Entity {name: $name})", name=name)

        for subj, rel, obj in triples:
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


def hybrid_retrieval(
    query: str,
    driver: Driver,
    vector_search: Callable[[str], List[str]],
    top_k: int = 5,
) -> List[str]:
    """Retrieve related nodes and vector search results for a query.

    Both the knowledge graph and the vector index are queried and the
    resulting contexts are merged while preserving order and removing
    duplicates.
    """

    graph_results = query_knowledge_graph(query, driver)
    vector_results = vector_search(query)

    combined: List[str] = []
    for item in graph_results + vector_results:
        if item not in combined:
            combined.append(item)
        if len(combined) >= top_k:
            break
    return combined

"""Neo4j based knowledge graph utilities."""

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

