"""Configuration management for the Claude-based RAG system."""

from dataclasses import dataclass, field
from typing import Optional, List
import os
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    YAML_AVAILABLE = False


@dataclass
class WeaviateConfig:
    url: str = "http://localhost:8080"
    index_name: str = "rag_docs"


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ClaudeConfig:
    model_name: str = "claude-3-5-haiku-20241022"
    api_key: str = ""
    max_tokens: int = 1000
    temperature: float = 0.1


@dataclass
class RetrievalConfig:
    default_top_k: int = 5
    partnership_top_k: int = 5
    factual_top_k: int = 5
    similarity_threshold: float = 0.3
    enable_section_reranking: bool = True
    section_name_boost: float = 1.0


@dataclass
class PromptingConfig:
    context_instructions: dict = field(default_factory=dict)
    system_prompt: str = ""


@dataclass
class HardwareConfig:
    use_gpu: bool = False
    gpu_memory_fraction: float = 0.0
    num_threads: int = 1
    max_model_memory: str = ""
    enable_model_offloading: bool = False
    enable_response_cache: bool = False
    cache_size: int = 0
    cache_ttl: int = 0


@dataclass
class ChunkProcessingConfig:
    add_section_markers: bool = True
    add_metadata_context: bool = True
    max_context_length: int = 100
    section_patterns: list = field(default_factory=list)


@dataclass
class QueryProcessingConfig:
    expand_synonyms: bool = False
    synonyms: dict = field(default_factory=dict)


@dataclass
class DomainDetectionConfig:
    enabled: bool = False
    domain_keywords: dict = field(default_factory=dict)


@dataclass
class LoggingConfig:
    log_chunk_retrieval: bool = False
    log_section_matching: bool = False
    log_query_classification: bool = False
    log_model_selection: bool = False
    log_confidence_scores: bool = False
    log_performance_metrics: bool = False
    level: str = "INFO"
    file: str = "logs/rag_system.log"
    track_model_performance: bool = False
    save_response_analytics: bool = False


class Config:
    """Central configuration management."""

    def __init__(self, config_path: str = "config.yaml"):
        self.documents_folder = "documents"
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.development = True

        self.weaviate = WeaviateConfig()
        self.embedding = EmbeddingConfig()
        self.claude = ClaudeConfig()
        self.retrieval = RetrievalConfig()
        self.prompting = PromptingConfig()
        self.hardware = HardwareConfig()
        self.chunk_processing = ChunkProcessingConfig()
        self.query_processing = QueryProcessingConfig()
        self.domain_detection = DomainDetectionConfig()
        self.logging = LoggingConfig()
        self.section_priorities = {}
        self.semantic_metadata = {}

        self.config_path = config_path
        if YAML_AVAILABLE and Path(config_path).exists():
            self._load_from_yaml(config_path)
        elif Path(config_path).exists():
            print(f"Warning: {config_path} found but PyYAML not available")

        self._apply_env_overrides()

    def _load_from_yaml(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            self.documents_folder = data.get("documents_folder", self.documents_folder)
            self.chunk_size = data.get("chunk_size", self.chunk_size)
            self.chunk_overlap = data.get("chunk_overlap", self.chunk_overlap)
            self.development = data.get("development", self.development)

            config_mappings = {
                "weaviate": self.weaviate,
                "embedding": self.embedding,
                "claude": self.claude,
                "retrieval": self.retrieval,
                "prompting": self.prompting,
                "hardware": self.hardware,
                "chunk_processing": self.chunk_processing,
                "query_processing": self.query_processing,
                "domain_detection": self.domain_detection,
                "logging": self.logging,
            }

            for name, obj in config_mappings.items():
                if name in data:
                    for key, value in data[name].items():
                        if hasattr(obj, key):
                            setattr(obj, key, value)

            if "section_priorities" in data:
                self.section_priorities = data.get("section_priorities", {})
            if "semantic_metadata" in data:
                self.semantic_metadata = data.get("semantic_metadata", {})
        except Exception as e:
            print(f"Warning: Could not load config from {path}: {e}")

    def _apply_env_overrides(self) -> None:
        self.development = os.getenv("DEVELOPMENT", str(self.development)).lower() == "true"
        self.documents_folder = os.getenv("DOCUMENTS_FOLDER", self.documents_folder)
        if os.getenv("WEAVIATE_URL"):
            self.weaviate.url = os.getenv("WEAVIATE_URL")
        if os.getenv("ANTHROPIC_API_KEY"):
            self.claude.api_key = os.getenv("ANTHROPIC_API_KEY")

    def validate(self) -> List[str]:
        errors = []
        if not Path(self.documents_folder).exists() and not self.development:
            errors.append(f"Documents folder not found: {self.documents_folder}")
        if self.chunk_size <= self.chunk_overlap:
            errors.append("Chunk size must be larger than overlap")
        if self.chunk_size <= 0 or self.chunk_overlap < 0:
            errors.append("Chunk size and overlap must be positive")
        return errors

    def setup_directories(self) -> None:
        Path(self.documents_folder).mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def get_summary(self) -> dict:
        return {
            "documents_folder": self.documents_folder,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "development_mode": self.development,
            "weaviate_url": self.weaviate.url,
            "embedding_model": self.embedding.model_name,
            "claude_model": self.claude.model_name,
            "retrieval_default_top_k": self.retrieval.default_top_k,
            "use_gpu": self.hardware.use_gpu,
        }
