"""Configuration management for the Claude-based RAG system."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
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
class Neo4jEnvironmentConfig:
    """Configuration for a single Neo4j environment."""
    uri: str = "bolt://localhost:7687"
    user: str = ""
    password: str = ""
    database: str = "neo4j"


@dataclass
class Neo4jConfig:
    """Configuration for the Neo4j knowledge graph with environment support."""
    # Legacy single config for backward compatibility
    uri: str = "bolt://localhost:7687"
    user: str = ""
    password: str = ""
    database: str = "neo4j"
    
    # New environment-based configs
    local: Optional[Neo4jEnvironmentConfig] = None
    aura: Optional[Neo4jEnvironmentConfig] = None
    railway: Optional[Neo4jEnvironmentConfig] = None  # Add railway support
    production: Optional[Neo4jEnvironmentConfig] = None

    def __post_init__(self):
        # Initialize environment configs if not provided
        if self.local is None:
            self.local = Neo4jEnvironmentConfig()
        if self.aura is None:
            self.aura = Neo4jEnvironmentConfig()
        if self.railway is None:
            self.railway = Neo4jEnvironmentConfig()
        if self.production is None:
            self.production = Neo4jEnvironmentConfig()


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


@dataclass
class OODConfig:
    """Out-of-domain verification settings."""
    enabled: bool = False
    similarity_threshold: float = 0.25
    similarity_check_enabled: bool = True
    min_neo4j_relations: int = 1
    domain_keywords: List[str] = field(default_factory=list)


class Config:
    """Central configuration management."""

    def __init__(self, config_path: str = "config.yaml"):
        self.documents_folder = "documents"
        self.chunk_size = 2000
        self.chunk_overlap = 200
        self.development = True
        self.environment = "local"  # Add environment setting

        self.weaviate = WeaviateConfig()
        self.embedding = EmbeddingConfig()
        self.claude = ClaudeConfig()
        self.neo4j = Neo4jConfig()
        self.retrieval = RetrievalConfig()
        self.prompting = PromptingConfig()
        self.hardware = HardwareConfig()
        self.chunk_processing = ChunkProcessingConfig()
        self.query_processing = QueryProcessingConfig()
        self.domain_detection = DomainDetectionConfig()
        self.logging = LoggingConfig()
        self.ood = OODConfig()
        self.section_priorities = {}
        self.semantic_metadata = {}
        self.weaviate_environments = {}  # Add support for environment-specific Weaviate URLs

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
            self.environment = data.get("environment", self.environment)  # Load environment

            # Load Weaviate environment configurations
            if "weaviate_environments" in data:
                self.weaviate_environments = data["weaviate_environments"]

            # Handle Neo4j configuration specially
            if "neo4j" in data:
                neo4j_data = data["neo4j"]
                
                # Load environment-specific configs
                if "local" in neo4j_data:
                    local_config = neo4j_data["local"]
                    self.neo4j.local = Neo4jEnvironmentConfig(
                        uri=local_config.get("uri", "bolt://localhost:7687"),
                        user=local_config.get("user", "neo4j"),
                        password=local_config.get("password", "password"),
                        database=local_config.get("database", "neo4j")
                    )
                
                if "aura" in neo4j_data:
                    aura_config = neo4j_data["aura"]
                    self.neo4j.aura = Neo4jEnvironmentConfig(
                        uri=aura_config.get("uri", ""),
                        user=aura_config.get("user", "neo4j"),
                        password=aura_config.get("password", ""),
                        database=aura_config.get("database", "neo4j")
                    )
                
                if "railway" in neo4j_data:
                    railway_config = neo4j_data["railway"]
                    self.neo4j.railway = Neo4jEnvironmentConfig(
                        uri=railway_config.get("uri", ""),
                        user=railway_config.get("user", "neo4j"),
                        password=railway_config.get("password", ""),
                        database=railway_config.get("database", "neo4j")
                    )
                
                if "production" in neo4j_data:
                    prod_config = neo4j_data["production"]
                    self.neo4j.production = Neo4jEnvironmentConfig(
                        uri=prod_config.get("uri", ""),
                        user=prod_config.get("user", ""),
                        password=prod_config.get("password", ""),
                        database=prod_config.get("database", "neo4j")
                    )

            # Handle other configs normally
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
                "ood": self.ood,
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
        
        # CRITICAL FIX: Apply environment override FIRST
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        
        # Apply environment-specific Weaviate URL AFTER environment is set
        if hasattr(self, 'weaviate_environments') and self.environment in self.weaviate_environments:
            self.weaviate.url = self.weaviate_environments[self.environment]["url"]
        
        # Environment variable overrides
        if os.getenv("WEAVIATE_URL"):
            self.weaviate.url = os.getenv("WEAVIATE_URL")
        if os.getenv("ANTHROPIC_API_KEY"):
            self.claude.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # IMPORTANT: Apply Railway-specific environment variables for Neo4j
        if self.environment == "railway":
            # Set Railway Neo4j config from environment variables or use hardcoded values
            if not self.neo4j.railway:
                self.neo4j.railway = Neo4jEnvironmentConfig()
            
            # Use environment variables if available, otherwise use Railway defaults
            self.neo4j.railway.uri = os.getenv("NEO4J_URI", "bolt://turntable.proxy.rlwy.net:43560")
            self.neo4j.railway.user = os.getenv("NEO4J_USER", "neo4j")
            self.neo4j.railway.password = os.getenv("NEO4J_PASSWORD", "mg89wxdi38d1xytau4yd40d7telcxtqo")
            self.neo4j.railway.database = os.getenv("NEO4J_DATABASE", "neo4j")
            
            print(f"🚀 Railway Neo4j configured: {self.neo4j.railway.uri}")
        
        # Handle environment variables for production Neo4j
        if os.getenv("NEO4J_URI"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USER"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.user = os.getenv("NEO4J_USER")
        if os.getenv("NEO4J_PASSWORD"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.password = os.getenv("NEO4J_PASSWORD")
        if os.getenv("NEO4J_DATABASE"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.database = os.getenv("NEO4J_DATABASE")

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
        # Get current environment Neo4j config
        neo4j_env_config = getattr(self.neo4j, self.environment, None)
        neo4j_uri = neo4j_env_config.uri if neo4j_env_config else self.neo4j.uri
        
        return {
            "environment": self.environment,
            "documents_folder": self.documents_folder,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "development_mode": self.development,
            "weaviate_url": self.weaviate.url,
            "embedding_model": self.embedding.model_name,
            "claude_model": self.claude.model_name,
            "retrieval_default_top_k": self.retrieval.default_top_k,
            "use_gpu": self.hardware.use_gpu,
            "neo4j_uri": neo4j_uri,
        }