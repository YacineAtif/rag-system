
"""Utility functions for working with a Neo4j knowledge graph and combining
results with vector search.
"""

import re
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
        "role",
    ]
    return "graph" if any(k in q for k in graph_keywords) else "vector"


def _extract_entity_name(query: str) -> str:
    """Extract a probable entity name from a natural language query."""
    patterns = [
        r"what is(?: the)? (.+?)'s role",
        r"what is(?: the)? role of (.+)",
        r"what is(?: the)? (.+?) role",
        r"who are(?: the)? partners of (.+)",
        r"who are(?: the)? (.+?)'s partners",
        r"who are(?: the)? (.+?) partners",
        r"who does (.+?) partner with",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            entity = match.group(1).strip()
            entity = re.sub(r"[?.,]$", "", entity).strip()
            return entity

    question_words = {"who", "what", "where", "when", "why", "how"}
    connectors = {"of", "the", "and", "&", "for"}
    tokens = query.split()
    name_tokens: List[str] = []
    for token in tokens:
        cleaned = re.sub(r"[?.,]", "", token)
        if not name_tokens:
            if cleaned.lower() in question_words:
                continue
            if cleaned[:1].isupper():
                name_tokens.append(cleaned)
        else:
            if cleaned[:1].isupper() or cleaned.lower() in connectors:
                name_tokens.append(cleaned)
            else:
                break
    if name_tokens:
        return " ".join(name_tokens)
    return query


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

    neo_cfg = getattr(config, "neo4j", None)
    if neo_cfg is None:
        raise ValueError("Neo4j configuration missing")

    driver = GraphDatabase.driver(
        neo_cfg.uri, auth=(neo_cfg.user, neo_cfg.password)
    )
    entity = _extract_entity_name(query)
    cypher = (
        "MATCH (n)-[r*1..2]-(m) "
        "WHERE toLower(n.name) CONTAINS toLower($q) "
        "RETURN n, m LIMIT 20"
    )
    records: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            result = session.run(cypher, q=entity)
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
    finally:
        driver.close()
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

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class DocumentMeta:
    """Metadata stored for each processed document."""

    hash: str
    modified: float


class DocumentTracker:
    """Keep track of document fingerprints for incremental processing."""

    def __init__(self, state_path: str) -> None:
        self.state_path = state_path
        self.documents: Dict[str, DocumentMeta] = {}
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for name, info in raw.get("documents", {}).items():
                    self.documents[name] = DocumentMeta(
                        hash=info.get("hash", ""),
                        modified=info.get("modified", 0.0),
                    )
            except Exception:
                # Corrupt state, start fresh
                self.documents = {}

    def _fingerprint(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def filter_documents(self, documents: List[Any]) -> Tuple[List[Any], List[str]]:
        """Return documents requiring processing and list of deleted docs."""

        to_process: List[Any] = []
        seen: Dict[str, DocumentMeta] = {}
        for doc in documents:
            meta = getattr(doc, "meta", {}) or {}
            name = meta.get("filename") or meta.get("file_path") or str(id(doc))
            mtime = meta.get("modified_date")
            if isinstance(mtime, str):
                # convert to timestamp
                try:
                    mtime = time.mktime(time.strptime(mtime))
                except Exception:
                    mtime = 0.0
            content_hash = self._fingerprint(getattr(doc, "content", ""))
            prev = self.documents.get(name)
            if not prev or prev.hash != content_hash:
                to_process.append(doc)
            seen[name] = DocumentMeta(hash=content_hash, modified=float(mtime or 0.0))

        deleted = [name for name in self.documents.keys() if name not in seen]
        self.documents = seen
        return to_process, deleted

    def save(self) -> None:
        data = {
            "documents": {
                name: {"hash": meta.hash, "modified": meta.modified}
                for name, meta in self.documents.items()
            }
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


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


def build_knowledge_graph(
    config: Config,
    state_path: str = "kg_state.json",
    cleanup_threshold: float = 0.0,
    stale_days: int = 30,
) -> bool:
    """Incrementally build or update a knowledge graph in Neo4j.

    Only documents whose content has changed since the last run are processed.
    Existing entities and relationships are merged and their confidence scores
    accumulated. Provenance information is tracked for each source document.
    Entities or relationships originating solely from deleted documents are
    pruned based on a confidence threshold.
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

    tracker = DocumentTracker(state_path)
    docs_to_process, deleted_docs = tracker.filter_documents(documents)

    if not docs_to_process and not deleted_docs:
        logger.info("No document changes detected; skipping update")
        return True

    qa = ClaudeQA(config)
    rows: List[Dict[str, Optional[str]]] = []

    for idx, doc in enumerate(docs_to_process, start=1):
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
            logger.info("Processed %d/%d documents", idx, len(docs_to_process))
        except Exception as e:  # pragma: no cover - runtime errors
            logger.exception("Failed to process chunk: %s", e)

    logger.info("Connecting to Neo4j at %s", config.neo4j.uri)
    try:
        driver = GraphDatabase.driver(
            config.neo4j.uri, auth=(config.neo4j.user, config.neo4j.password)
        )
    except Exception as e:  # pragma: no cover - runtime errors
        logger.error("Failed to connect to Neo4j: %s", e)
        return False

    ts = time.time()
    ingest_query = """
    UNWIND $rows as row
    MERGE (s:Entity {name: row.subject})
      ON CREATE SET s.sources=[row.file], s.confidence=row.confidence, s.last_seen=$ts
      ON MATCH SET s.sources = CASE WHEN row.file IN s.sources THEN s.sources ELSE s.sources + row.file END,
                    s.confidence = coalesce(s.confidence,0)+row.confidence,
                    s.last_seen=$ts
    MERGE (o:Entity {name: row.object})
      ON CREATE SET o.sources=[row.file], o.confidence=row.confidence, o.last_seen=$ts
      ON MATCH SET o.sources = CASE WHEN row.file IN o.sources THEN o.sources ELSE o.sources + row.file END,
                    o.confidence = coalesce(o.confidence,0)+row.confidence,
                    o.last_seen=$ts
    MERGE (s)-[r:RELATED {relation: row.predicate}]->(o)
      ON CREATE SET r.sources=[row.file], r.confidence=row.confidence, r.last_seen=$ts
      ON MATCH SET r.sources = CASE WHEN row.file IN r.sources THEN r.sources ELSE r.sources + row.file END,
                    r.confidence = coalesce(r.confidence,0)+row.confidence,
                    r.last_seen=$ts
    """

    remove_query_node = """
    MATCH (n:Entity)
    WHERE $file IN n.sources
    SET n.sources = [src IN n.sources WHERE src <> $file],
        n.confidence = coalesce(n.confidence,0) - 1,
        n.last_seen = $ts
    WITH n
    WHERE size(n.sources)=0 OR n.confidence <= $threshold
    DETACH DELETE n
    """

    remove_query_rel = """
    MATCH ()-[r:RELATED]-()
    WHERE $file IN r.sources
    SET r.sources = [src IN r.sources WHERE src <> $file],
        r.confidence = coalesce(r.confidence,0) - 1,
        r.last_seen = $ts
    WITH r
    WHERE size(r.sources)=0 OR r.confidence <= $threshold
    DELETE r
    """

    cleanup_query_nodes = """
    MATCH (n:Entity)
    WHERE n.last_seen < $cutoff AND n.confidence <= $threshold
    DETACH DELETE n
    """

    cleanup_query_rels = """
    MATCH ()-[r:RELATED]-()
    WHERE r.last_seen < $cutoff AND r.confidence <= $threshold
    DELETE r
    """

    try:
        with driver.session() as session:
            if rows:
                session.run(ingest_query, rows=rows, ts=ts)
                logger.info("Inserted/updated %d triples", len(rows))
            for name in deleted_docs:
                session.run(remove_query_rel, file=name, ts=ts, threshold=cleanup_threshold)
                session.run(remove_query_node, file=name, ts=ts, threshold=cleanup_threshold)

            cutoff = ts - stale_days * 24 * 3600
            session.run(cleanup_query_rels, cutoff=cutoff, threshold=cleanup_threshold)
            session.run(cleanup_query_nodes, cutoff=cutoff, threshold=cleanup_threshold)

        tracker.save()
        return True
    except Exception as e:  # pragma: no cover - runtime errors
        logger.exception("Failed to update Neo4j: %s", e)
        return False
    finally:
        try:
            driver.close()
        except Exception:
            pass

