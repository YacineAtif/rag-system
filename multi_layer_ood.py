"""Multi-layered OOD detection and abstention system.

This module provides a skeleton implementation of the multi-layer
Out-Of-Domain detection pipeline described in the project requirements.
The design focuses on extensibility and configuration driven behaviour.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional


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
    response_verification: "ResponseVerificationConfig" = field(
        default_factory=lambda: ResponseVerificationConfig()
    )


@dataclass
class ResponseVerificationConfig:
    enable_fact_checking: bool = True
    enable_consistency_validation: bool = True
    enable_citation_verification: bool = True
    hallucination_detection_threshold: float = 0.3
    high_confidence_threshold: float = 0.9
    relaxed_hallucination_threshold: float = 0.5


class QueryAnalyzer:
    """Perform basic query analysis and expansion.

    The implementation here is intentionally lightweight – production systems
    should replace heuristics with model powered reasoning components.
    """

    def analyze(self, query: str) -> Dict[str, str]:
        """Return query type and a naive complexity estimate."""

        q = query.lower().strip()
        if any(word in q for word in ["why", "cause", "reason"]):
            qtype = "causal"
        elif any(word in q for word in ["compare", "versus", "difference"]):
            qtype = "comparative"
        elif any(word in q for word in ["how", "procedure", "steps"]):
            qtype = "procedural"
        else:
            qtype = "factual"

        word_count = len(q.split())
        complexity = "complex" if word_count > 15 or (" and " in q or " or " in q) else "simple"
        return {"type": qtype, "expanded_query": query, "complexity": complexity}


class DomainBoundaryDetector:
    """Combine multiple signals to decide whether the query is in-domain."""

    def __init__(self, config: OODDetectionConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__ + ".DomainBoundaryDetector")

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

    def check(
        self, query: str, similarity: float = 1.0, graph_connectivity: float = 1.0
    ) -> Tuple[bool, Dict[str, float]]:
        """Return True if all boundary signals pass configured thresholds."""

        kw = self._keyword_score(query)
        sim_ok = self.embedding_similarity(similarity)
        graph_ok = graph_connectivity >= self.cfg.graph_connectivity_threshold
        passed = kw >= 0.5 and sim_ok and graph_ok
        signals = {
            "keyword_score": kw,
            "similarity": similarity,
            "graph_connectivity": graph_connectivity,
        }
        self.logger.debug("Domain boundary signals: %s", signals)
        return passed, signals


class RetrievalQualityGate:
    def __init__(self, config: OODDetectionConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__ + ".RetrievalQualityGate")

    def evaluate(
        self,
        relevances: Sequence[float],
        contradiction_score: float = 0.0,
        information_density: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """Check retrieval quality against configured gates."""

        signals: Dict[str, float] = {}
        if len(relevances) < self.cfg.quality_gates.min_passage_count:
            return False, {
                "avg_relevance": 0.0,
                "contradiction_score": contradiction_score,
                "information_density": information_density or 0.0,
            }

        avg = sum(relevances) / len(relevances)
        signals["avg_relevance"] = avg

        info_density = (
            information_density
            if information_density is not None
            else avg
        )
        signals["information_density"] = info_density
        signals["contradiction_score"] = contradiction_score

        passed = (
            avg >= self.cfg.quality_gates.min_context_relevance
            and contradiction_score <= self.cfg.quality_gates.max_contradiction_score
            and info_density >= self.cfg.quality_gates.min_information_density
        )
        self.logger.debug("Retrieval quality signals: %s", signals)
        return passed, signals


class GenerationGuard:
    def __init__(self, config: OODDetectionConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__ + ".GenerationGuard")

    def evaluate(self, probs: Sequence[float]) -> Tuple[float, bool]:
        """Return confidence score and generation permission."""

        if not probs:
            self.logger.debug("No probability data provided; returning zero confidence")
            return 0.0, False

        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        confidence = 1.0 / (1.0 + entropy)
        risky = any(p < 1e-4 for p in probs)
        allow = confidence >= self.cfg.generation_confidence_threshold and not risky
        self.logger.debug(
            "Generation guard: entropy=%.4f confidence=%.4f risky=%s allow=%s",
            entropy,
            confidence,
            risky,
            allow,
        )
        return confidence, allow


class ResponseVerifier:
    """Verify generated responses using simple heuristic checks."""

    def __init__(self, config: ResponseVerificationConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__ + ".ResponseVerifier")

    def verify(
        self,
        answer: str,
        sources: Sequence[str],
        query: str,
        confidence: float | None = None,
    ) -> Tuple[bool, Dict[str, float]]:
        signals: Dict[str, float] = {}

        answer_l = answer.lower()
        # Fact checking: ensure some source text appears in answer
        if self.cfg.enable_fact_checking:
            matches = sum(1 for s in sources if s and s.lower() in answer_l)
            fact_score = matches / max(len(sources), 1)
        else:
            fact_score = 1.0
        signals["fact_score"] = fact_score

        # Consistency: naive check that answer doesn't contradict query intent
        if self.cfg.enable_consistency_validation:
            inconsistent = "i don't know" in answer_l and "?" not in query
            consistency = 0.0 if inconsistent else 1.0
        else:
            consistency = 1.0
        signals["consistency"] = consistency

        # Citation verification: ensure citation count <= sources
        if self.cfg.enable_citation_verification:
            citation_count = answer.count("[")  # simplistic
            citation_score = 1.0 if citation_count <= len(sources) else 0.0
        else:
            citation_score = 1.0
        signals["citation_score"] = citation_score

        hallucination_risk = 1.0 - fact_score
        signals["hallucination_risk"] = hallucination_risk

        threshold = self.cfg.hallucination_detection_threshold
        if (
            confidence is not None
            and confidence >= self.cfg.high_confidence_threshold
        ):
            threshold = max(threshold, self.cfg.relaxed_hallucination_threshold)

        passed = (
            hallucination_risk <= threshold
            and consistency > 0.0
            and citation_score > 0.0
        )
        self.logger.debug("Response verification signals: %s", signals)
        return passed, signals


class MultiLayerOODDetector:
    """High level orchestrator for the multi-layer OOD system."""

    def __init__(self, config: OODDetectionConfig | None = None):
        self.cfg = config or OODDetectionConfig()
        self.query_analyzer = QueryAnalyzer()
        self.boundary = DomainBoundaryDetector(self.cfg)
        self.retrieval = RetrievalQualityGate(self.cfg)
        self.guard = GenerationGuard(self.cfg)
        self.verifier = ResponseVerifier(self.cfg.response_verification)

    def process(
        self,
        query: str,
        similarity: float,
        graph_connectivity: float,
        retrieved_relevances: Sequence[float],
        token_probs: Sequence[float],
        answer: Optional[str] = None,
        sources: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        """Run all OOD checks and optional response verification."""

        analysis = self.query_analyzer.analyze(query)
        in_domain, boundary_signals = self.boundary.check(
            query, similarity, graph_connectivity
        )
        retrieval_ok, retrieval_signals = self.retrieval.evaluate(retrieved_relevances)
        confidence, gen_allowed = self.guard.evaluate(token_probs)
        allow_generation = in_domain and retrieval_ok and gen_allowed

        result: Dict[str, object] = {
            "analysis": analysis,
            "boundary_signals": boundary_signals,
            "retrieval_signals": retrieval_signals,
            "confidence": confidence,
            "in_domain": in_domain,
            "retrieval_ok": retrieval_ok,
            "allow_generation": allow_generation,
        }

        if answer is not None and sources is not None:
            verified, verification = self.verifier.verify(
                answer, sources, query, confidence
            )
            result["response_verified"] = verified
            result["verification_signals"] = verification

        return result
