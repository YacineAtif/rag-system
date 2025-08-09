"""
Hybrid processing pipeline for the RAG system.
Foundation for combining extractive and generative approaches.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

# Try to import config, handle gracefully if not available
try:
    from backend.config import Config
    from backend.qa_models import ClaudeQA
except ImportError:
    print("Backend config not available, using defaults")
    Config = None

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Available processing modes for the pipeline."""
    LEGACY = "legacy"
    EXTRACTIVE_ONLY = "extractive"
    GENERATIVE_ONLY = "generative"
    HYBRID_AUTO = "hybrid_auto"
    ENHANCED = "enhanced"

@dataclass
class QueryResult:
    """Standardized query result format across all processing modes."""
    answer: str
    confidence: float
    processing_mode: str
    sources: List[str]
    metadata: Dict[str, Any]

class HybridPipeline:
    """Main processing pipeline - foundation for modular RAG system."""

    def __init__(self, config=None):
        """Initialize pipeline with optional configuration."""
        self.config = config or (Config() if Config else None)
        self.initialized = False

        # Component placeholder for future extensions
        self.claude_interface = None
        self.legacy_pipeline = None

        # Setup logging
        logging.basicConfig(level=logging.WARNING)
        logger.info("HybridPipeline instance created")

    def _is_factual(self, question: str) -> bool:
        q = question.lower()
        return q.startswith("what is") or q.startswith("define") or "definition" in q

    def _is_partnership(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "partner",
            "collaborator",
            "organization",
            "company",
            "team",
            "consortium",
            "stakeholder",
            "member",
            "involved",
            "working",
        ]
        return any(k in q for k in keywords)

    def _route_models(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        """Use Claude for all queries."""
        claude = ClaudeQA(self.config)
        res = claude.generate(question, contexts)
        res["model"] = "claude"
        return res




    def initialize(self) -> bool:
        """Initialize all available pipeline components."""
        try:
            logger.info("Initializing pipeline components...")

            # Check for legacy pipeline availability
            try:
                import weaviate_rag_pipeline_transformers
                self.legacy_pipeline = weaviate_rag_pipeline_transformers
                logger.info("Legacy pipeline module available")
            except ImportError:
                logger.warning("Legacy pipeline module not available")



            self.initialized = True
            logger.info("Pipeline initialization completed")
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    def process_query(
        self,
        question: str,
        contexts: List[str],
        mode: ProcessingMode = ProcessingMode.HYBRID_AUTO
    ) -> QueryResult:
        """Process a query through the specified processing mode."""
        if not self.initialized:
            logger.error("Pipeline not initialized")
            return QueryResult(
                answer="Error: Pipeline not initialized. Call initialize() first.",
                confidence=0.0,
                processing_mode="error",
                sources=[],
                metadata={"error": "not_initialized"}
            )

        logger.info(f"Processing query with mode: {mode.value}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Contexts provided: {len(contexts)}")

        if mode == ProcessingMode.LEGACY:
            return self._process_legacy_mode(question, contexts)
        elif mode == ProcessingMode.EXTRACTIVE_ONLY:
            return self._process_extractive_placeholder(question, contexts)
        elif mode == ProcessingMode.GENERATIVE_ONLY:
            return self._process_generative_placeholder(question, contexts)
        elif mode == ProcessingMode.ENHANCED:
            return self._process_enhanced_placeholder(question, contexts)
        else:
            return self._process_hybrid_auto_placeholder(question, contexts)

    def _process_legacy_mode(self, question: str, contexts: List[str]) -> QueryResult:
        """Process using legacy pipeline approach (integration placeholder)."""
        if self.legacy_pipeline:
            logger.info("Using legacy pipeline mode")
            return QueryResult(
                answer=f"[Legacy Mode] Processing question: '{question}'\n\nThis would integrate with the existing weaviate_rag_pipeline_transformers.py logic. The legacy system would process this query using the established embedding and retrieval pipeline.",
                confidence=0.8,
                processing_mode="legacy_integrated",
                sources=["legacy_pipeline"],
                metadata={
                    "method": "legacy_integration",
                    "legacy_module": "weaviate_rag_pipeline_transformers",
                    "context_count": len(contexts)
                }
            )
        else:
            logger.warning("Legacy pipeline not available")
            return QueryResult(
                answer="Legacy pipeline mode requested but legacy module not available. Please ensure weaviate_rag_pipeline_transformers.py is accessible.",
                confidence=0.0,
                processing_mode="legacy_error",
                sources=[],
                metadata={"error": "legacy_module_not_found"}
            )

    def _process_extractive_placeholder(self, question: str, contexts: List[str]) -> QueryResult:
        """Extractive mode using Claude."""
        logger.info("Using extractive placeholder mode")

        res = self._route_models(question, contexts)
        return QueryResult(
            answer=res.get("answer", ""),
            confidence=res.get("confidence", 0.0),
            processing_mode="extractive",
            sources=[f"context_{i}" for i in range(len(contexts))],
            metadata={"model": res.get("model"), "model_confidence": res.get("confidence")}
        )

    def _process_generative_placeholder(self, question: str, contexts: List[str]) -> QueryResult:
        """Generative mode answered by Claude."""
        logger.info("Using generative placeholder mode")

        res = self._route_models(question, contexts)
        return QueryResult(
            answer=res.get("answer", ""),
            confidence=res.get("confidence", 0.0),
            processing_mode="generative",
            sources=[f"context_{i}" for i in range(len(contexts))],
            metadata={"model": res.get("model"), "model_confidence": res.get("confidence")}
        )

    def _process_enhanced_placeholder(self, question: str, contexts: List[str]) -> QueryResult:
        """Enhanced processing combining extractive and generative."""
        logger.info("Using enhanced placeholder mode")

        res = self._route_models(question, contexts)

        return QueryResult(
            answer=res.get("answer", ""),
            confidence=res.get("confidence", 0.0),
            processing_mode="enhanced",
            sources=[f"context_{i}" for i in range(len(contexts))],
            metadata={"model": res.get("model"), "model_confidence": res.get("confidence")}
        )

    def _process_hybrid_auto_placeholder(self, question: str, contexts: List[str]) -> QueryResult:
        """Placeholder for hybrid auto routing."""
        logger.info("Using hybrid auto placeholder mode")

        if len(question.split()) <= 5:
            return self._process_extractive_placeholder(question, contexts)
        else:
            return self._process_generative_placeholder(question, contexts)

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "initialized": self.initialized,
            "claude_available": self.claude_interface is not None,
            "config_loaded": self.config is not None,
            "legacy_available": self.legacy_pipeline is not None,
        }

# Test the pipeline
if __name__ == "__main__":
    print("\U0001f9ea Testing HybridPipeline...")

    try:
        config = Config() if Config else None
        pipeline = HybridPipeline(config)

        print("Initializing pipeline...")
        success = pipeline.initialize()
        print(f"Initialization: {'Success' if success else 'Failed'}")

        status = pipeline.get_status()
        print(f"Status: {status}")

        question = "What is artificial intelligence?"
        contexts = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans.",
            "AI research focuses on developing algorithms that can perform tasks requiring human intelligence."
        ]

        print("\n\U0001f9ea Testing processing modes:")
        for mode in ProcessingMode:
            print(f"\n  {mode.value}:")
            result = pipeline.process_query(question, contexts, mode)
            print(f"    Answer: {result.answer[:100]}...")
            print(f"    Confidence: {result.confidence:.3f}")
            print(f"    Mode: {result.processing_mode}")

        print("\nHybridPipeline test complete")

    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
