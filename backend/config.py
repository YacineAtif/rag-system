"""Configuration management for the RAG system with local/railway modes."""

from dataclasses import dataclass, field
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

@dataclass
class WeaviateConfig:
    url: str = "http://localhost:8080"
    index_name: str = "Default"

@dataclass
class ClaudeConfig:
    model_name: str = "claude-3-5-haiku-20241022"
    api_key: str = ""
    max_tokens: int = 1000
    temperature: float = 0.1

@dataclass
class ChunkProcessingConfig:
    section_patterns: list = field(default_factory=lambda: [
        r"^#+\s*(.+)$",  # Markdown headers
        r"^([A-Z][^:]*):$",  # Title case sections ending with colon
        r"^\d+\.\s*([^:]+):?$"  # Numbered sections
    ])

@dataclass
class Config:
    documents_folder: str = "documents"
    chunk_size: int = 2000
    chunk_overlap: int = 200
    development: bool = True
    environment: str = "local"
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    chunk_processing: ChunkProcessingConfig = field(default_factory=ChunkProcessingConfig)
    config_path: str = "config.yaml"

    def __post_init__(self):
        # Load YAML configuration if available
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f) or {}

                # Core settings
                self.environment = data.get("environment", self.environment)
                self.documents_folder = data.get("documents_folder", self.documents_folder)
                self.chunk_size = data.get("chunk_size", self.chunk_size)
                self.chunk_overlap = data.get("chunk_overlap", self.chunk_overlap)
                self.development = data.get("development", self.development)

                # Environment-specific settings
                env_cfg = data.get("environments", {}).get(self.environment, {})
                weav_cfg = env_cfg.get("weaviate", {})
                neo_cfg = env_cfg.get("neo4j", {})

                self.weaviate.url = weav_cfg.get("url", self.weaviate.url)
                self.weaviate.index_name = weav_cfg.get("index_name", self.weaviate.index_name)
                self.neo4j.uri = neo_cfg.get("uri", self.neo4j.uri)
                self.neo4j.user = neo_cfg.get("user", self.neo4j.user)
                self.neo4j.password = neo_cfg.get("password", self.neo4j.password)
                self.neo4j.database = neo_cfg.get("database", self.neo4j.database)

                claude_cfg = data.get("claude", {})
                self.claude.model_name = claude_cfg.get("model_name", self.claude.model_name)
                self.claude.max_tokens = claude_cfg.get("max_tokens", self.claude.max_tokens)
                self.claude.temperature = claude_cfg.get("temperature", self.claude.temperature)

            except Exception as e:
                print(f"⚠️ Warning: Error loading config file {self.config_path}: {e}")

        # Environment variable overrides
        env_override = os.getenv("ENVIRONMENT")
        if env_override:
            self.environment = env_override

        weav_url = os.getenv("WEAVIATE_URL")
        if weav_url:
            self.weaviate.url = weav_url

        neo_uri = os.getenv("NEO4J_URI")
        if neo_uri:
            self.neo4j.uri = neo_uri
        neo_user = os.getenv("NEO4J_USER")
        if neo_user:
            self.neo4j.user = neo_user
        neo_pwd = os.getenv("NEO4J_PASSWORD")
        if neo_pwd:
            self.neo4j.password = neo_pwd
        neo_db = os.getenv("NEO4J_DATABASE")
        if neo_db:
            self.neo4j.database = neo_db

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.claude.api_key = api_key

        dev_env = os.getenv("DEVELOPMENT")
        if dev_env:
            self.development = dev_env.lower() == "true"

    def validate(self) -> None:
        """Basic configuration validation."""
        if not Path(self.documents_folder).exists():
            print(f"Warning: Documents folder not found: {self.documents_folder}")
        if self.chunk_size <= self.chunk_overlap:
            print("Warning: Chunk size must be larger than overlap")
        if self.chunk_size <= 0 or self.chunk_overlap < 0:
            print("Warning: Chunk size and overlap must be positive")

    def get_summary(self) -> dict:
        """Return a summary of key configuration settings."""
        return {
            "environment": self.environment,
            "documents_folder": self.documents_folder,
            "neo4j_uri": self.neo4j.uri,
            "weaviate_url": self.weaviate.url,
            "claude_model": self.claude.model_name
        }