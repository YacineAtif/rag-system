import logging
import re
from typing import Any, List, Sequence, Tuple

import numpy as np

from domain_concept_registry import DomainConceptRegistry

logger = logging.getLogger(__name__)


class GraphSemanticAnalyzer:
    """Analyze graph relationships for semantic coherence."""

    def __init__(self, registry: DomainConceptRegistry):
        self.registry = registry

    def analyze(
        self, query: str, graph_results: Sequence[Any]
    ) -> Tuple[float, List[str], List[str]]:
        logger.debug("Graph results type: %s", type(graph_results))
        if graph_results:
            logger.debug("First result sample: %s", graph_results[0])
            logger.debug("First result type: %s", type(graph_results[0]))

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

        processed_results: List[str] = []
        for result in graph_results:
            if isinstance(result, dict):
                rel_str = (
                    f"{result.get('source', '')} -> {result.get('relationship', '')} -> {result.get('target', '')}"
                )
                processed_results.append(rel_str)
            elif isinstance(result, (tuple, list)):
                rel_str = " -> ".join(str(item) for item in result)
                processed_results.append(rel_str)
            elif isinstance(result, str):
                processed_results.append(result)
            else:
                processed_results.append(str(result))

        for rel in processed_results:
            if not isinstance(rel, str):
                continue
            parts = [p.strip().lower() for p in re.split(r"->|--|,", rel) if p.strip()]
            if len(parts) < 2:
                continue
            for node in parts:
                if node in entity_counts:
                    entity_counts[node] += 1
                else:
                    neighborhood.add(node)

        centrality = (sum(entity_counts.values()) / (len(processed_results) or 1))
        density = len(processed_results) / (len(matched) + 1)
        semantic = float(np.mean(match_scores)) if match_scores else 0.0
        graph_score = (centrality + density + semantic) / 3
        logger.debug(
            "Graph semantics: score=%.3f centrality=%.3f density=%.3f semantic=%.3f",
            graph_score,
            centrality,
            density,
            semantic,
        )
        return graph_score, matched, list(neighborhood)
