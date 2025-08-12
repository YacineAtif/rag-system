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
import os
from typing import Optional

import yaml

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import (
    WeaviateEmbeddingRetriever,
)

# Import the updated Config class
from backend.config import Config

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
        # Load configuration using the updated Config class
        self.config = Config(config_path)
        print(f"🔧 RAGBackend using environment: {self.config.environment}")
        
        # Set up logging using the raw YAML for setup_logging function compatibility
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        setup_logging(config_dict)

        self._ensure_infrastructure()

        # Get environment-aware Weaviate URL
        weaviate_url = self._get_current_weaviate_url()
        print(f"🔗 Connecting to Weaviate at: {weaviate_url}")

        if not wait_for_weaviate(url=weaviate_url):
            raise RuntimeError(f"Weaviate not responding at {weaviate_url}. Check configuration.")

        # Build Haystack components with environment-aware URLs
        self.document_store = WeaviateDocumentStore(url=weaviate_url)
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            progress_bar=False,
            encode_kwargs={"convert_to_tensor": False},
        )
        print("🔥 Warming up text embedder...")
        self.text_embedder.warm_up()
        retriever = WeaviateEmbeddingRetriever(document_store=self.document_store)

        # Initialize pipeline with the updated Config object
        # This will now use environment-based Neo4j configuration
        self.pipeline = RAGPipeline(self.document_store, retriever, self.text_embedder)
        
        # CRITICAL FIX: Make sure the pipeline uses our environment-aware config
        self.pipeline.graph_builder.config = self.config
        
        # Update the global CONFIG to match our environment
        global CONFIG
        from weaviate_rag_pipeline_transformers import CONFIG as GLOBAL_CONFIG
        GLOBAL_CONFIG.environment = self.config.environment
        GLOBAL_CONFIG.neo4j = self.config.neo4j
        
        print(f"🔗 Neo4j connection will use: {self._get_current_neo4j_uri()}")

        # Check if we should skip document processing
        skip_processing = os.getenv("SKIP_DOCUMENT_PROCESSING", "false").lower() == "true"
        if skip_processing:
            print("⏭️ Skipping document processing (SKIP_DOCUMENT_PROCESSING=true)")
        else:
            # Process documents and populate knowledge graph
            self.pipeline.process_documents_intelligently()
            if force_rebuild:
                self.pipeline.force_rebuild_graph()
            else:
                self.pipeline.populate_knowledge_graph()

    def _get_current_weaviate_url(self) -> str:
        """Get the Weaviate URL based on current environment."""
        env = self.config.environment
        
        # Check if environment-specific Weaviate URL is configured
        if hasattr(self.config, 'weaviate_environments') and env in self.config.weaviate_environments:
            return self.config.weaviate_environments[env]["url"]
        
        # Fallback to main config or environment variable
        return self.config.weaviate.url

    def _get_current_neo4j_uri(self) -> str:
        """Get the Neo4j URI that will be used based on current environment."""
        env = self.config.environment
        print(f"🔍 Debug: Environment is '{env}'")
        
        if env == 'railway' and hasattr(self.config.neo4j, 'railway') and self.config.neo4j.railway:
            uri = self.config.neo4j.railway.uri
            print(f"🚀 Using Railway Neo4j: {uri}")
            return uri
        elif env == 'local' and hasattr(self.config.neo4j, 'local') and self.config.neo4j.local:
            uri = self.config.neo4j.local.uri
            print(f"🏠 Using Local Neo4j: {uri}")
            return uri
        elif env == 'aura' and hasattr(self.config.neo4j, 'aura') and self.config.neo4j.aura:
            uri = self.config.neo4j.aura.uri
            print(f"☁️ Using Aura Neo4j: {uri}")
            return uri
        elif env == 'production' and hasattr(self.config.neo4j, 'production') and self.config.neo4j.production:
            uri = self.config.neo4j.production.uri
            print(f"🏭 Using Production Neo4j: {uri}")
            return uri
        else:
            uri = self.config.neo4j.uri
            print(f"⚠️ Fallback Neo4j: {uri} (environment: {env})")
            return uri

    def _ensure_infrastructure(self) -> None:
        """Ensure required infrastructure is available."""
        environment = self.config.environment
        
        # For cloud environments, assume infrastructure is managed externally
        if environment in ["railway", "aura", "production"]:
            print(f"🌐 Running in {environment} environment - infrastructure managed externally")
            return
        
        # For local development, check and start Docker containers if needed
        if environment == "local":
            if check_docker_containers():
                print("✅ Docker containers already running")
                return
            
            print("🚀 Starting Docker infrastructure...")
            try:
                subprocess.run(["docker-compose", "up", "-d"], check=True)
                print("✅ Docker containers started")
                # Give services a moment to start
                time.sleep(30)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Failed to start Docker containers") from exc
        else:
            # Unknown environment - assume infrastructure is available
            print(f"⚠️ Unknown environment '{environment}' - assuming infrastructure is available")

    def _format_answer(self, text: str) -> str:
        """Normalize whitespace and preserve paragraph breaks."""
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    def query(self, query: str) -> dict:
        """Execute a query through the RAG pipeline."""
        result = self.pipeline.query_with_graph(query)
        if "answer" in result:
            result["answer"] = self._format_answer(result["answer"])
        
        # Add environment info to the result
        result["environment"] = self.config.environment
        result["neo4j_uri"] = self._get_current_neo4j_uri()
        result["weaviate_url"] = self._get_current_weaviate_url()
        
        return result

    def get_stats(self) -> dict:
        """Get statistics from the knowledge graph."""
        try:
            with self.pipeline.graph_builder.driver.session() as session:
                entity_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                entity_count = entity_result.single()["count"]
                
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
            
            return {
                "entities": entity_count,
                "relationships": rel_count,
                "environment": self.config.environment,
                "neo4j_uri": self._get_current_neo4j_uri(),
                "weaviate_url": self._get_current_weaviate_url()
            }
        except Exception as e:
            return {
                "entities": 0,
                "relationships": 0,
                "environment": self.config.environment,
                "error": str(e)
            }

    def get_health(self) -> dict:
        """Get health status with environment information."""
        return {
            "status": "ok",
            "environment": self.config.environment,
            "neo4j_uri": self._get_current_neo4j_uri(),
            "weaviate_url": self._get_current_weaviate_url()
        }


__all__ = ["RAGBackend"]