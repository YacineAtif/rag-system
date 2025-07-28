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

        # Component placeholders (will be implemented in future iterations)
        self.deberta_reader = None
        self.qwen_interface = None
        self.legacy_pipeline = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger.info("HybridPipeline instance created")

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

            # Future: Initialize DeBERTa reader
            # self.deberta_reader = DeBERTaReader(self.config)

            # Future: Initialize Qwen interface
            # self.qwen_interface = QwenInterface(self.config)

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
        """Placeholder for future DeBERTa extractive processing."""
        logger.info("Using extractive placeholder mode")

        if contexts:
            question_words = set(question.lower().split())
            best_context = ""
            max_relevance = 0

            for context in contexts:
                context_words = set(context.lower().split())
                relevance = len(question_words & context_words)
                if relevance > max_relevance:
                    max_relevance = relevance
                    best_context = context

            snippet = best_context[:250] + "..." if len(best_context) > 250 else best_context
            confidence = min(0.8, max_relevance * 0.1)
        else:
            snippet = "No context provided for extraction"
            confidence = 0.1

        return QueryResult(
            answer=f"[Extractive Mode - Placeholder] Based on relevant context:\n\n{snippet}\n\nThis is a placeholder response. Full DeBERTa V3 extractive QA integration will provide precise span-based answers.",
            confidence=confidence,
            processing_mode="extractive_placeholder",
            sources=["context_analysis"] if contexts else [],
            metadata={
                "method": "keyword_matching_placeholder",
                "relevance_score": max_relevance if contexts else 0,
                "context_count": len(contexts),
                "future_model": "microsoft/deberta-v3-base-squad2"
            }
        )

    def _process_generative_placeholder(self, question: str, contexts: List[str]) -> QueryResult:
        """Placeholder for future Qwen generative processing."""
        logger.info("Using generative placeholder mode")

        if contexts:
            context_summary = " ".join(contexts)
            context_summary = context_summary[:400] + "..." if len(context_summary) > 400 else context_summary
        else:
            context_summary = "No context provided"

        return QueryResult(
            answer=f"[Generative Mode - Placeholder] Question: {question}\n\nContext Summary: {context_summary}\n\nThis is a placeholder response. Full Qwen LLM integration will provide comprehensive, contextually-aware generated answers that can synthesize information across multiple sources.",
            confidence=0.6,
            processing_mode="generative_placeholder",
            sources=[f"context_{i}" for i in range(len(contexts))],
            metadata={
                "method": "template_placeholder",
                "context_length": len(context_summary),
                "question_length": len(question),
                "future_model": "qwen-7b-chat"
            }
        )

    def _process_enhanced_placeholder(self, question: str, contexts: List[str]) -> QueryResult:
        """Placeholder for enhanced processing (extractive + generative refinement)."""
        logger.info("Using enhanced placeholder mode")

        extractive_result = self._process_extractive_placeholder(question, contexts)

        enhanced_answer = f"{extractive_result.answer}\n\n[Enhanced with Generative Refinement]\nThe extractive answer above would be enhanced and refined using Qwen's generative capabilities to provide more comprehensive, fluent, and contextually appropriate responses."

        return QueryResult(
            answer=enhanced_answer,
            confidence=extractive_result.confidence,
            processing_mode="enhanced_placeholder",
            sources=extractive_result.sources,
            metadata={
                "method": "enhanced_placeholder",
                "base_mode": "extractive",
                "future_model": "qwen-7b-chat"
            }
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
            "deberta_available": self.deberta_reader is not None,
            "qwen_available": self.qwen_interface is not None,
            "config_loaded": self.config is not None,
            "legacy_available": self.legacy_pipeline is not None
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
