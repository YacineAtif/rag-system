"""Utility functions for working with a Neo4j knowledge graph and combining
results with vector search.
"""

from typing import List, Dict, Any

try:
    from backend.config import Config
except Exception:  # pragma: no cover - optional import for tests
    Config = None  # type: ignore


def _classify_query(query: str) -> str:
    """Very small heuristic to decide retrieval strategy."""
    q = query.lower()
    graph_keywords = [
        "relationship",
        "connected",
        "connect",
        "who",
        "partner",
        "collaborator",
    ]
    return "graph" if any(k in q for k in graph_keywords) else "vector"


def query_knowledge_graph(query: str, config: Config) -> List[Dict[str, Any]]:
    """Query Neo4j using a simple Cypher traversal.

    Parameters
    ----------
    query:
        Free text describing the entity of interest.
    config:
        Application configuration with optional knowledge graph settings.

    Returns
    -------
    List of dictionaries with node properties.
    """
    try:
        from neo4j import GraphDatabase
    except Exception:  # pragma: no cover - neo4j optional
        raise ImportError("neo4j package required for knowledge graph queries")

    kg_cfg = getattr(config, "knowledge_graph", None) or {}
    uri = getattr(kg_cfg, "uri", "bolt://localhost:7687")
    user = getattr(kg_cfg, "user", "neo4j")
    password = getattr(kg_cfg, "password", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    cypher = (
        "MATCH (n)-[r*1..2]-(m) "
        "WHERE toLower(n.name) CONTAINS toLower($q) "
        "RETURN n, m LIMIT 20"
    )
    records: List[Dict[str, Any]] = []
    with driver.session() as session:
        result = session.run(cypher, q=query)
        for rec in result:
            item: Dict[str, Any] = {}
            for key, val in rec.items():
                if hasattr(val, "_properties"):
                    props = dict(val._properties)
                    props["id"] = getattr(val, "id", None)
                    item[key] = props
                else:
                    item[key] = val
            records.append(item)
    return records


def _vector_search(query: str, config: Config):
    """Retrieve documents from Weaviate using Haystack components."""
    try:
        from haystack import Pipeline
        from haystack.components.embedders import SentenceTransformersTextEmbedder
        from haystack_integrations.document_stores.weaviate import (
            WeaviateDocumentStore,
        )
        from haystack_integrations.components.retrievers.weaviate import (
            WeaviateEmbeddingRetriever,
        )
    except Exception:  # pragma: no cover - optional dependencies
        return []

    document_store = WeaviateDocumentStore(url=config.weaviate.url)
    embedder = SentenceTransformersTextEmbedder(model=config.embedding.model_name)
    retriever = WeaviateEmbeddingRetriever(document_store=document_store)

    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    result = pipeline.run(
        {
            "embedder": {"text": query},
            "retriever": {"top_k": config.retrieval.default_top_k},
        }
    )
    return result.get("retriever", {}).get("documents", [])


def hybrid_retrieval(query: str, config: Config) -> List[str]:
    """Combine graph traversal with vector search for retrieval."""
    strategy = _classify_query(query)
    contexts: List[str] = []

    if strategy == "graph":
        try:
            graph_results = query_knowledge_graph(query, config)
        except Exception:
            graph_results = []
        for item in graph_results:
            n = item.get("n", {}).get("name", "")
            m = item.get("m", {}).get("name", "")
            if n and m:
                contexts.append(f"{n} -> {m}")
    
    docs = _vector_search(query, config)
    for doc in docs:
        content = getattr(doc, "content", None)
        if content:
            contexts.append(str(content))

    return contexts
