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

