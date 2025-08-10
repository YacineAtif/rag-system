"""Shared backend logic for RAG system.

This module exposes a `RAGBackend` class that wraps the existing
`RAGPipeline` from `weaviate_rag_pipeline_transformers` so it can be
reused by both the command line interface and the new web application.

The implementation mirrors the setup performed in the CLI's `main`
function but packages it in a class that can be instantiated and queried
programmatically.
"""
from __future__ import annotations

import json
import time
import subprocess
from typing import Optional

import yaml

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import (
    WeaviateEmbeddingRetriever,
)

# Reuse the existing pipeline and utilities from the CLI module.
from weaviate_rag_pipeline_transformers import (
    RAGPipeline,
    check_docker_containers,
    setup_logging,
    wait_for_weaviate,
)


class RAGBackend:
    """Backend helper that wraps the heavy RAG pipeline."""

    def __init__(self, config_path: str = "config.yaml", force_rebuild: bool = False):
        # Load configuration and set up logging exactly as the CLI does.
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        setup_logging(config)

        self._ensure_infrastructure()

        if not wait_for_weaviate():
            raise RuntimeError("Weaviate not responding. Check docker-compose logs.")

        # Build Haystack components
        self.document_store = WeaviateDocumentStore(url="http://localhost:8080")
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            progress_bar=False,
            encode_kwargs={"convert_to_tensor": False},
        )
        self.text_embedder.warm_up()
        retriever = WeaviateEmbeddingRetriever(document_store=self.document_store)

        # Initialize pipeline from the original module
        self.pipeline = RAGPipeline(self.document_store, retriever, self.text_embedder)

        # Process documents and populate knowledge graph
        self.pipeline.process_documents_intelligently()
        if force_rebuild:
            self.pipeline.force_rebuild_graph()
        else:
            self.pipeline.populate_knowledge_graph()

    def _ensure_infrastructure(self) -> None:
        """Ensure required Docker services are running."""
        if check_docker_containers():
            return
        try:
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            # Give services a moment to start
            time.sleep(30)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime environment
            raise RuntimeError("Failed to start Docker containers") from exc

    def _format_answer(self, text: str) -> str:
        """Normalize whitespace and preserve paragraph breaks."""
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    def query(self, query: str) -> dict:
        """Execute a query through the RAG pipeline."""
        result = self.pipeline.query_with_graph(query)
        if "answer" in result:
            result["answer"] = self._format_answer(result["answer"])
        return result


__all__ = ["RAGBackend"]
