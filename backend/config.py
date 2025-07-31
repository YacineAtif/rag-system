"""
Configuration management for the modular RAG system.
Centralizes settings and replaces hardcoded values from existing scripts.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os
from pathlib import Path

# Try to import yaml, handle gracefully if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. Add 'PyYAML>=6.0' to requirements.txt")

@dataclass
class WeaviateConfig:
    url: str = "http://localhost:8080"
    index_name: str = "rag_docs"

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class DeBERTaConfig:
    model_name: str = "microsoft/deberta-v3-base-squad2"
    max_length: int = 512
    confidence_threshold: float = 0.7
    max_answer_length: int = 256

@dataclass
class QwenConfig:
    model_name: str = "qwen-7b-chat"
    api_url: str = ""
    api_key: str = ""
    max_tokens: int = 1000
    temperature: float = 0.1

@dataclass
class HybridConfig:
    confidence_threshold: float = 0.75
    enable_qwen_fallback: bool = True


@dataclass
class OpenAIConfig:
    enabled: bool = False
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    fallback_only: bool = True

@dataclass
class RetrievalConfig:
    default_top_k: int = 5
    partnership_top_k: int = 5
    factual_top_k: int = 5
    similarity_threshold: float = 0.3
    enable_section_reranking: bool = True
    section_name_boost: float = 1.0
    deberta_max_context: int = 400
    qwen_max_context: int = 800

@dataclass
class PromptingConfig:
    context_instructions: dict = None
    qwen_system_prompt: str = ""
    deberta_instruction: str = ""

    def __post_init__(self):
        if self.context_instructions is None:
            self.context_instructions = {}

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
    section_patterns: list = None

    def __post_init__(self):
        if self.section_patterns is None:
            self.section_patterns = []

@dataclass
class QueryProcessingConfig:
    expand_synonyms: bool = False
    synonyms: dict = None

    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = {}

@dataclass
class DomainDetectionConfig:
    enabled: bool = False
    domain_keywords: dict = None

    def __post_init__(self):
        if self.domain_keywords is None:
            self.domain_keywords = {}

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
class MultiModelConfig:
    enabled: bool = True
    primary_llm: str = "qwen"
    qa_model: str = "deberta"
    model_selection: dict = None
    confidence_thresholds: dict = None
    ensemble_mode: str = ""
    combine_answers: bool = False
    timeout_seconds: int = 45
    max_retries: int = 2
    parallel_inference: bool = False
    offload_unused_models: bool = False
    max_memory_usage: str = ""

    def __post_init__(self):
        if self.model_selection is None:
            self.model_selection = {}
        if self.confidence_thresholds is None:
            self.confidence_thresholds = {}

@dataclass
class SectionPrioritiesConfig:
    queries: dict = None

    def __post_init__(self):
        if self.queries is None:
            self.queries = {}
=======

@dataclass
class MultiModelConfig:
    """Configuration for multi-model selection and thresholds."""

    enabled: bool = True
    model_selection: dict = field(default_factory=dict)
    confidence_thresholds: dict = field(default_factory=dict)
    primary_llm: str = "qwen"
    qa_model: str = "deberta"


class Config:
    """Central configuration management with YAML and environment support."""

    def __init__(self, config_path: str = "config.yaml"):
        # Default values (extracted from existing scripts)
        self.documents_folder = "documents"
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.development = True

        # Initialize sub-configs with dataclasses
        self.weaviate = WeaviateConfig()
        self.embedding = EmbeddingConfig()
        self.deberta = DeBERTaConfig()
        self.qwen = QwenConfig()
        self.hybrid = HybridConfig()

        self.openai = OpenAIConfig()
        self.retrieval = RetrievalConfig()
        self.prompting = PromptingConfig()
        self.hardware = HardwareConfig()
        self.chunk_processing = ChunkProcessingConfig()
        self.query_processing = QueryProcessingConfig()
        self.domain_detection = DomainDetectionConfig()
        self.logging = LoggingConfig()
        self.multi_model = MultiModelConfig()
        self.section_priorities = SectionPrioritiesConfig()
        self.semantic_metadata = {}

        self.multi_model = MultiModelConfig()


        # Store config path
        self.config_path = config_path

        # Load from YAML if available
        if YAML_AVAILABLE and Path(config_path).exists():
            self._load_from_yaml(config_path)
        elif Path(config_path).exists():
            print(f"Warning: {config_path} found but PyYAML not available")

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _load_from_yaml(self, path: str):
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}

            # Update main settings
            self.documents_folder = data.get('documents_folder', self.documents_folder)
            self.chunk_size = data.get('chunk_size', self.chunk_size)
            self.chunk_overlap = data.get('chunk_overlap', self.chunk_overlap)
            self.development = data.get('development', self.development)

            # Update nested configurations
            config_mappings = {
                'weaviate': self.weaviate,
                'embedding': self.embedding,
                'deberta': self.deberta,
                'qwen': self.qwen,
                'hybrid': self.hybrid,

                'openai': self.openai,
                'retrieval': self.retrieval,
                'prompting': self.prompting,
                'hardware': self.hardware,
                'chunk_processing': self.chunk_processing,
                'query_processing': self.query_processing,
                'domain_detection': self.domain_detection,
                'logging': self.logging,
                'multi_model': self.multi_model

                'multi_model': self.multi_model,

            }

            for config_name, config_obj in config_mappings.items():
                if config_name in data:
                    config_data = data[config_name]
                    for key, value in config_data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)

            # Non-dataclass mappings
            if 'section_priorities' in data:
                self.section_priorities.queries = data.get('section_priorities', {})
            if 'semantic_metadata' in data:
                self.semantic_metadata = data.get('semantic_metadata', {})

        except Exception as e:
            print(f"Warning: Could not load config from {path}: {e}")

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Main settings
        self.development = os.getenv('DEVELOPMENT', str(self.development)).lower() == 'true'
        self.documents_folder = os.getenv('DOCUMENTS_FOLDER', self.documents_folder)

        # Weaviate settings
        if os.getenv('WEAVIATE_URL'):
            self.weaviate.url = os.getenv('WEAVIATE_URL')

        # API keys and sensitive settings
        if os.getenv('QWEN_API_KEY'):
            self.qwen.api_key = os.getenv('QWEN_API_KEY')
        if os.getenv('QWEN_API_URL'):
            self.qwen.api_url = os.getenv('QWEN_API_URL')

        # Hardware overrides
        if os.getenv('USE_GPU'):
            self.hardware.use_gpu = os.getenv('USE_GPU').lower() == 'true'
        if os.getenv('GPU_MEMORY_FRACTION'):
            try:
                self.hardware.gpu_memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION'))
            except ValueError:
                pass
        if os.getenv('NUM_THREADS'):
            try:
                self.hardware.num_threads = int(os.getenv('NUM_THREADS'))
            except ValueError:
                pass

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors/warnings."""
        errors = []

        # Check documents folder in production
        if not Path(self.documents_folder).exists() and not self.development:
            errors.append(f"Documents folder not found: {self.documents_folder}")

        # Validate chunk settings
        if self.chunk_size <= self.chunk_overlap:
            errors.append("Chunk size must be larger than overlap")

        if self.chunk_size <= 0 or self.chunk_overlap < 0:
            errors.append("Chunk size and overlap must be positive")

        # Validate confidence thresholds
        if not (0 <= self.hybrid.confidence_threshold <= 1):
            errors.append("Hybrid confidence threshold must be between 0 and 1")

        if not (0 <= self.deberta.confidence_threshold <= 1):
            errors.append("DeBERTa confidence threshold must be between 0 and 1")

        # Check required settings in production
        if not self.development:
            if self.hybrid.enable_qwen_fallback and not self.qwen.api_key:
                errors.append("Qwen API key required when fallback enabled in production")

        return errors

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.documents_folder).mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def get_summary(self) -> dict:
        """Get configuration summary for display."""
        return {
            'documents_folder': self.documents_folder,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'development_mode': self.development,
            'weaviate_url': self.weaviate.url,
            'embedding_model': self.embedding.model_name,
            'deberta_model': self.deberta.model_name,
            'qwen_configured': bool(self.qwen.api_key),
            'retrieval_default_top_k': self.retrieval.default_top_k,
            'use_gpu': self.hardware.use_gpu
        }

# Test the configuration system
if __name__ == "__main__":
    print("Testing Configuration System...")
    print("=" * 50)

    try:
        # Test configuration loading
        config = Config()
        print("Configuration loaded successfully")

        # Show current settings
        print("\nCurrent Settings:")
        summary = config.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Test validation
        print("\nValidation Check:")
        errors = config.validate()
        if errors:
            print("  Issues found:")
            for error in errors:
                print(f"    - {error}")
        else:
            print("  All validations passed")

        # Test directory setup
        print("\nDirectory Setup:")
        config.setup_directories()
        docs_exists = Path(config.documents_folder).exists()
        logs_exists = Path("logs").exists()
        print(f"  Documents folder: {'OK' if docs_exists else 'MISSING'} {config.documents_folder}")
        print(f"  Logs folder: {'OK' if logs_exists else 'MISSING'} logs/")

        print("\nConfiguration system working correctly")

    except Exception as e:
        print(f"Configuration system test failed: {e}")
        import traceback
        traceback.print_exc()
