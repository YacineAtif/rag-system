"""Multi-layered OOD detection and abstention system.

This module provides a skeleton implementation of the multi-layer
Out-Of-Domain detection pipeline described in the project requirements.
The design focuses on extensibility and configuration driven behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import math


@dataclass
class KeywordTiers:
    """Hierarchical keyword configuration."""
    tier_1_critical: Dict[str, List[str]] = field(default_factory=dict)
    tier_2_important: Dict[str, List[str]] = field(default_factory=dict)
    tier_3_contextual: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class QualityGates:
    min_context_relevance: float = 0.7
    min_passage_count: int = 2
    max_contradiction_score: float = 0.3
    min_information_density: float = 0.6


@dataclass
class AbstentionConfig:
    enable_informed_refusal: bool = True
    enable_partial_answers: bool = True
    enable_clarification_requests: bool = True
    confidence_explanation: bool = True


@dataclass
class OODDetectionConfig:
    enabled: bool = True
    similarity_threshold: float = 0.15
    graph_connectivity_threshold: float = 0.6
    context_quality_threshold: float = 0.7
    generation_confidence_threshold: float = 0.8
    keywords: KeywordTiers = field(default_factory=KeywordTiers)
    quality_gates: QualityGates = field(default_factory=QualityGates)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)


class QueryAnalyzer:
    """Perform basic query analysis and expansion.

    The implementation here is intentionally lightweight – production systems
    should replace heuristics with model powered reasoning components.
    """

    def analyze(self, query: str) -> Dict[str, str]:
        # Very small heuristic classifier
        q = query.lower()
        if any(word in q for word in ["why", "cause", "reason"]):
            qtype = "causal"
        elif any(word in q for word in ["compare", "versus", "difference"]):
            qtype = "comparative"
        elif any(word in q for word in ["how", "procedure"]):
            qtype = "procedural"
        else:
            qtype = "factual"
        return {"type": qtype, "expanded_query": query}


class DomainBoundaryDetector:
    """Combine multiple signals to decide whether the query is in-domain."""

    def __init__(self, config: OODDetectionConfig):
        self.cfg = config

    def _keyword_score(self, query: str) -> float:
        q = query.lower()
        tiers = [
            (self.cfg.keywords.tier_1_critical, 1.0),
            (self.cfg.keywords.tier_2_important, 0.8),
            (self.cfg.keywords.tier_3_contextual, 0.5),
        ]
        score = 0.0
        for tier, weight in tiers:
            for group in tier.values():
                if any(kw.lower() in q for kw in group):
                    score = max(score, weight)
        return score

    def embedding_similarity(self, similarity: float) -> bool:
        return similarity >= self.cfg.similarity_threshold

    def check(self, query: str, similarity: float = 1.0) -> Tuple[bool, Dict[str, float]]:
        kw = self._keyword_score(query)
        passed = kw >= 0.5 and self.embedding_similarity(similarity)
        return passed, {"keyword_score": kw, "similarity": similarity}


class RetrievalQualityGate:
    def __init__(self, config: OODDetectionConfig):
        self.cfg = config

    def evaluate(self, relevances: Sequence[float]) -> bool:
        if len(relevances) < self.cfg.quality_gates.min_passage_count:
            return False
        avg = sum(relevances) / len(relevances)
        return avg >= self.cfg.quality_gates.min_context_relevance


class GenerationGuard:
    def __init__(self, config: OODDetectionConfig):
        self.cfg = config

    def evaluate(self, probs: Sequence[float]) -> float:
        if not probs:
            return 0.0
        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        confidence = 1.0 / (1.0 + entropy)
        return confidence

    def should_generate(self, confidence: float) -> bool:
        return confidence >= self.cfg.generation_confidence_threshold


class MultiLayerOODDetector:
    """High level orchestrator for the multi-layer OOD system."""

    def __init__(self, config: OODDetectionConfig | None = None):
        self.cfg = config or OODDetectionConfig()
        self.query_analyzer = QueryAnalyzer()
        self.boundary = DomainBoundaryDetector(self.cfg)
        self.retrieval = RetrievalQualityGate(self.cfg)
        self.guard = GenerationGuard(self.cfg)

    def process(
        self,
        query: str,
        similarity: float,
        retrieved_relevances: Sequence[float],
        token_probs: Sequence[float],
    ) -> Dict[str, bool]:
        analysis = self.query_analyzer.analyze(query)
        in_domain, signals = self.boundary.check(query, similarity)
        retrieval_ok = self.retrieval.evaluate(retrieved_relevances)
        confidence = self.guard.evaluate(token_probs)
        allow_generation = (
            in_domain and retrieval_ok and self.guard.should_generate(confidence)
        )
        return {
            "in_domain": in_domain,
            "analysis": analysis,
            "signals": signals,
            "retrieval_ok": retrieval_ok,
            "confidence": confidence,
            "allow_generation": allow_generation,
        }
