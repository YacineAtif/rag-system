import logging
import re
from typing import List, Sequence, Tuple

import numpy as np

from domain_concept_registry import DomainConceptRegistry

logger = logging.getLogger(__name__)


class GraphSemanticAnalyzer:
    """Analyze graph relationships for semantic coherence."""

    def __init__(self, registry: DomainConceptRegistry):
        self.registry = registry

    def analyze(self, query: str, graph_results: Sequence[str]) -> Tuple[float, List[str], List[str]]:
        tokens = re.findall(r"\w+", query.lower())
        matched: List[str] = []
        match_scores: List[float] = []
        for tok in tokens:
            concept, score = self.registry.match(tok)
            if score >= 0.8:
                matched.append(concept)
                match_scores.append(score)
        entity_counts = {e: 0 for e in matched}
        neighborhood: set[str] = set()
        for rel in graph_results:
            parts = [p.strip().lower() for p in re.split(r"->|--|,", rel) if p.strip()]
            if len(parts) < 2:
                continue
            for node in parts:
                if node in entity_counts:
                    entity_counts[node] += 1
                else:
                    neighborhood.add(node)
        centrality = (sum(entity_counts.values()) / (len(graph_results) or 1))
        density = len(graph_results) / (len(matched) + 1)
        semantic = float(np.mean(match_scores)) if match_scores else 0.0
        graph_score = (centrality + density + semantic) / 3
        logger.debug(
            "Graph semantics: score=%.3f centrality=%.3f density=%.3f semantic=%.3f", 
            graph_score, centrality, density, semantic,
        )
        return graph_score, matched, list(neighborhood)
