"""Multi-layered OOD detection and abstention system.

This module provides a skeleton implementation of the multi-layer
Out-Of-Domain detection pipeline described in the project requirements.
The design focuses on extensibility and configuration driven behaviour.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer

logger = logging.getLogger(__name__)

from domain_concept_registry import DomainConceptRegistry
from graph_semantic_analyzer import GraphSemanticAnalyzer
from query_normalizer import QueryNormalizer


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
    """Configuration for response verification and uncertainty estimation."""

    # Context relevance
    relevance_threshold: float = 0.5
    min_relevant_passages: int = 1

    # Uncertainty metrics
    lexical_similarity_threshold: float = 0.5
    max_entropy_threshold: float = 1.5
    mean_entropy_threshold: float = 0.5
    sar_threshold: float = 0.3


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

    def __init__(self, config: OODDetectionConfig, graph_analyzer: "GraphSemanticAnalyzer"):
        self.cfg = config
        self.graph_analyzer = graph_analyzer
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

    def assess_graph_coverage(
        self, query: str, graph_results: Sequence[object]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze graph results for semantic coverage of the query."""

        score, entities, neighborhood = self.graph_analyzer.analyze(query, graph_results)
        self.logger.debug(
            "Graph coverage score=%.3f entities=%s neighborhood=%s",
            score,
            entities,
            neighborhood,
        )
        return score, entities, neighborhood

    def check(
        self,
        query: str,
        similarity: float,
        retrieval_quality: float,
        graph_semantic_score: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """Return True if boundary signals indicate an in-domain query."""

        kw = self._keyword_score(query)
        sim_threshold = self.cfg.similarity_threshold
        if graph_semantic_score >= 0.7:
            sim_threshold *= 0.5  # strong graph evidence lowers similarity requirement
        sim_ok = similarity >= sim_threshold
        final_score = 0.4 * similarity + 0.3 * graph_semantic_score + 0.3 * retrieval_quality
        passed = graph_semantic_score >= 0.7 or (kw >= 0.5 and sim_ok and final_score >= 0.5)
        signals = {
            "keyword_score": kw,
            "similarity": similarity,
            "retrieval_quality": retrieval_quality,
            "graph_semantic_score": graph_semantic_score,
            "final_score": final_score,
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


class ContextRelevanceAssessor:
    """Score and filter passages based on query relevance."""

    def __init__(self, cfg: ResponseVerificationConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__ + ".ContextRelevanceAssessor")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            self.logger.debug("Loading SentenceTransformer model %s", model_name)
            self._embedder = SentenceTransformer(model_name, show_progress_bar=False)
            self.logger.debug(
                "Loaded model with embedding dimension %s",
                self._embedder.get_sentence_embedding_dimension(),
            )
            self._use_transformer = True
        except Exception as exc:  # pragma: no cover - network failure fallback
            self.logger.warning(
                "Falling back to HashingVectorizer embeddings due to: %s", exc
            )
            self._use_transformer = False
            self._vectorizer = HashingVectorizer(
                n_features=384, alternate_sign=False, norm="l2"
            )
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._score_cache: Dict[Tuple[str, str], float] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        if text not in self._embedding_cache:
            if self._use_transformer:
                emb = self._embedder.encode(text, normalize_embeddings=True, show_progress_bar=False)
                self.logger.debug("Embedding for '%s' shape=%s", text, emb.shape)
            else:  # pragma: no cover - simple hashing fallback
                emb = self._vectorizer.transform([text]).toarray()[0]
                self.logger.debug("Hashing embedding for '%s' shape=%s", text, emb.shape)
            self._embedding_cache[text] = emb
        return self._embedding_cache[text]

    def score_passages(
        self,
        query: str,
        passages: Sequence[str],
        passage_embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[Tuple[str, float]]:
        """Return passages with relevance scores using semantic similarity."""

        q_emb = self._get_embedding(query)
        self.logger.debug("Query embedding shape=%s", getattr(q_emb, "shape", None))
        scored: List[Tuple[str, float]] = []
        for idx, p in enumerate(passages):
            key = (query, p)
            if key in self._score_cache:
                score = self._score_cache[key]
            else:
                if passage_embeddings and idx < len(passage_embeddings):
                    p_emb = np.asarray(passage_embeddings[idx])
                else:
                    p_emb = self._get_embedding(p)
                norm = np.linalg.norm(q_emb) * np.linalg.norm(p_emb)
                score = float(np.dot(q_emb, p_emb) / norm) if norm else 0.0
                self.logger.debug(
                    "Passage %d embedding shape=%s norm=%.4f score=%.4f",
                    idx,
                    getattr(p_emb, "shape", None),
                    norm,
                    score,
                )
                self._score_cache[key] = score
            scored.append((p, score))
        self.logger.debug("Context relevance scores: %s", scored)
        return scored

    def filter_relevant(
        self,
        query: str,
        passages: Sequence[str],
        passage_embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> Tuple[List[str], List[float]]:
        scored = self.score_passages(query, passages, passage_embeddings)
        relevant = [p for p, s in scored if s >= self.cfg.relevance_threshold]
        scores = [s for _, s in scored]
        return relevant, scores


class ResponseVerifier:
    """Estimate uncertainty of a generated answer against context."""

    def __init__(self, config: ResponseVerificationConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__ + ".ResponseVerifier")

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    @classmethod
    def _overlap_ratio(cls, a: str, b: str) -> float:
        ta, tb = cls._tokenize(a), cls._tokenize(b)
        if not ta:
            return 0.0
        return len(ta & tb) / len(ta)

    def verify(
        self,
        answer: str,
        context: Sequence[str],
        token_probs: Sequence[float] | None = None,
        attention: Sequence[float] | None = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """Return pass/fail along with uncertainty signals."""

        signals: Dict[str, float] = {}
        answer_l = answer.lower()

        overlaps = [self._overlap_ratio(answer_l, c.lower()) for c in context]
        lexical_similarity = max(overlaps, default=0.0)
        signals["lexical_similarity"] = lexical_similarity

        if token_probs:
            entropies = [-math.log(p + 1e-9) for p in token_probs]
            max_entropy = max(entropies)
            mean_entropy = sum(entropies) / len(entropies)
        else:
            max_entropy = mean_entropy = float("inf")
        signals["max_entropy"] = max_entropy
        signals["mean_entropy"] = mean_entropy

        if attention:
            sar_score = sum(attention) / len(attention)
        else:
            sar_score = 0.0
        signals["sar_score"] = sar_score

        confidence = (
            lexical_similarity
            + max(0.0, 1 - mean_entropy)
            + sar_score
        ) / 3
        signals["confidence"] = confidence

        passed = (
            lexical_similarity >= self.cfg.lexical_similarity_threshold
            and max_entropy <= self.cfg.max_entropy_threshold
            and mean_entropy <= self.cfg.mean_entropy_threshold
            and (sar_score >= self.cfg.sar_threshold or not attention)
        )
        self.logger.debug("Response verification signals: %s", signals)
        return passed, signals


class MultiLayerOODDetector:
    """High level orchestrator for the multi-layer OOD system."""

    def __init__(self, config: OODDetectionConfig | None = None):
        self.cfg = config or OODDetectionConfig()
        self.query_analyzer = QueryAnalyzer()
        concepts = ["sample", "theory", "widget", "system"]
        self.registry = DomainConceptRegistry(concepts)
        self.normalizer = QueryNormalizer(self.registry)
        self.graph_analyzer = GraphSemanticAnalyzer(self.registry)
        self.boundary = DomainBoundaryDetector(self.cfg, self.graph_analyzer)
        self.retrieval = RetrievalQualityGate(self.cfg)
        self.guard = GenerationGuard(self.cfg)
        self.relevance = ContextRelevanceAssessor(self.cfg.response_verification)
        self.verifier = ResponseVerifier(self.cfg.response_verification)

    def process(
        self,
        query: str,
        similarity: float,
        retrieved_passages: Sequence[str],
        token_probs: Sequence[float],
        graph_results: Sequence[object] | None = None,
        answer: Optional[str] = None,
        attention: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """Run all OOD checks and optional response verification."""

        logger.info("🔍 Analyzing query...")
        normalized_query, matched = self.normalizer.normalize(query)
        analysis = self.query_analyzer.analyze(normalized_query)
        graph_results = graph_results or []
        graph_score, graph_entities, neighborhood = self.boundary.assess_graph_coverage(
            normalized_query, graph_results
        )
        logger.info("📊 Processing embeddings...")
        scores = [s for _, s in self.relevance.score_passages(normalized_query, retrieved_passages)]
        boosted_scores: List[float] = []
        for passage, score in zip(retrieved_passages, scores):
            if any(ent in passage.lower() for ent in neighborhood):
                boosted_scores.append(min(1.0, score + 0.1))
            else:
                boosted_scores.append(score)
        relevant_passages = [
            p
            for p, s in zip(retrieved_passages, boosted_scores)
            if s >= self.cfg.response_verification.relevance_threshold
        ]
        retrieval_ok, retrieval_signals = self.retrieval.evaluate(boosted_scores)
        avg_rel = retrieval_signals.get("avg_relevance", 0.0)
        in_domain, boundary_signals = self.boundary.check(
            normalized_query, similarity, avg_rel, graph_score
        )
        confidence, gen_allowed = self.guard.evaluate(token_probs)
        final_score = 0.4 * similarity + 0.3 * graph_score + 0.3 * avg_rel
        allow_generation = (in_domain and retrieval_ok and gen_allowed) or graph_score >= 0.7
        abstention_reason = ""
        if not allow_generation:
            if not in_domain and graph_score < 0.7:
                abstention_reason = "out-of-domain"
            elif not retrieval_ok:
                abstention_reason = "insufficient context"
            else:
                abstention_reason = "low confidence"

        result: Dict[str, object] = {
            "analysis": analysis,
            "boundary_signals": boundary_signals,
            "retrieval_signals": retrieval_signals,
            "confidence": confidence,
            "in_domain": in_domain,
            "retrieval_ok": retrieval_ok,
            "allow_generation": allow_generation,
            "abstain": not allow_generation,
            "abstention_reason": abstention_reason,
            "relevance_scores": boosted_scores,
            "relevant_passages": relevant_passages,
            "graph_semantic_score": graph_score,
            "graph_entities": graph_entities,
            "graph_neighborhood": neighborhood,
            "final_score": final_score,
        }

        if answer is not None:
            verified, verification = self.verifier.verify(
                answer, relevant_passages, token_probs, attention
            )
            result["response_verified"] = verified
            result["verification_signals"] = verification

        logger.info("✅ Analysis complete")
        return result
