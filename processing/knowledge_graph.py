
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

"""Knowledge graph construction utilities."""

import logging
import os
from typing import Dict, List, Optional

from backend.config import Config
from backend.qa_models import ClaudeQA

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None  # type: ignore

try:
    from domain_loader import load_documents_from_folder
except Exception:  # pragma: no cover - optional dependency
    load_documents_from_folder = None  # type: ignore

logger = logging.getLogger(__name__)


def _parse_triples(text: str) -> List[Dict[str, str]]:
    """Parse triples in `subject|predicate|object` format."""
    triples: List[Dict[str, str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        triples.append({"subject": parts[0], "predicate": parts[1], "object": parts[2]})
    return triples


def build_knowledge_graph(config: Config) -> bool:
    """Build a knowledge graph in Neo4j from documents specified in the config.

    Each document chunk is processed by Claude to extract knowledge graph triples.
    Nodes and relationships are batch inserted into Neo4j along with metadata.
    """
    logging.basicConfig(level=logging.INFO)

    if load_documents_from_folder is None:
        logger.error("domain_loader module not available")
        return False

    if GraphDatabase is None:
        logger.error("Neo4j driver not installed")
        return False

    logger.info("Loading documents from %s", config.documents_folder)
    documents = load_documents_from_folder(config.documents_folder)
    if not documents:
        logger.warning("No documents found in %s", config.documents_folder)
        return False

    qa = ClaudeQA(config)
    rows: List[Dict[str, Optional[str]]] = []

    for doc in documents:
        text = getattr(doc, "content", "")
        meta = getattr(doc, "meta", {}) or {}
        try:
            result = qa.generate(
                "List entities and relationships as 'subject | relation | object' lines.",
                [text],
            )
            triples = _parse_triples(result.get("answer", ""))
            for t in triples:
                t.update(
                    {
                        "file": meta.get("filename"),
                        "section": meta.get("section_name"),
                        "confidence": result.get("confidence", 0.0),
                    }
                )
                rows.append(t)
        except Exception as e:  # pragma: no cover - runtime errors
            logger.exception("Failed to process chunk: %s", e)

    if not rows:
        logger.warning("No triples extracted from documents")
        return False

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    logger.info("Connecting to Neo4j at %s", uri)
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:  # pragma: no cover - runtime errors
        logger.error("Failed to connect to Neo4j: %s", e)
        return False

    query = """
    UNWIND $rows as row
    MERGE (s:Entity {name: row.subject})
    MERGE (o:Entity {name: row.object})
    MERGE (s)-[r:RELATED {relation: row.predicate}]->(o)
    SET r.file = row.file,
        r.section = row.section,
        r.confidence = row.confidence
    """
    try:
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, rows=rows))
        logger.info("Inserted %d triples into Neo4j", len(rows))
        return True
    except Exception as e:  # pragma: no cover - runtime errors
        logger.exception("Failed to write to Neo4j: %s", e)
        return False
    finally:
        try:
            driver.close()
        except Exception:
            pass

