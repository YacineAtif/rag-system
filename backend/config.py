"""
Configuration management for the modular RAG system.
Centralizes settings and replaces hardcoded values from existing scripts.
"""

from dataclasses import dataclass
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
                'hybrid': self.hybrid
            }

            for config_name, config_obj in config_mappings.items():
                if config_name in data:
                    config_data = data[config_name]
                    for key, value in config_data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)

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
            'qwen_configured': bool(self.qwen.api_key)
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
