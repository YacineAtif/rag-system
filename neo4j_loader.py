"""Utility to build a knowledge graph in Neo4j from documents."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple, Optional

from backend.config import Config
from backend.knowledge_graph import KnowledgeGraph
from domain_loader import load_text_file, load_pdf_file, load_docx_file


def extract_simple_entities(text: str) -> Iterable[str]:
    """Very naive entity extractor based on capitalised words."""
    pattern = re.compile(r"\b([A-Z][A-Za-z0-9_]+(?:\s+[A-Z][A-Za-z0-9_]+)*)")
    for match in pattern.finditer(text):
        yield match.group(1)


def load_documents(folder: Path) -> Iterable[Tuple[str, str]]:
    """Yield (filename, content) for supported documents."""
    loaders = {
        ".txt": load_text_file,
        ".md": load_text_file,
        ".pdf": load_pdf_file,
        ".docx": load_docx_file,
    }
    for path in folder.rglob("*"):
        if path.suffix.lower() in loaders and path.is_file():
            yield path.name, loaders[path.suffix.lower()](str(path))


def build_graph(config: Optional[Config] = None) -> None:
    cfg = config or Config()
    kg = KnowledgeGraph(cfg)
    folder = Path(cfg.documents_folder)
    triples = []
    for name, content in load_documents(folder):
        doc_node = ("Document", name, "CONTAINS", "Sentence", name)
        # each entity becomes Concept node
        for entity in extract_simple_entities(content):
            triples.append(("Document", name, "MENTIONS", "Concept", entity))
    if triples:
        kg.ingest_entities(triples)
    kg.close()


if __name__ == "__main__":
    build_graph()
