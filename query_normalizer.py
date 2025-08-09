import logging
from typing import List, Tuple

from domain_concept_registry import DomainConceptRegistry

logger = logging.getLogger(__name__)


class QueryNormalizer:
    """Normalize queries using domain vocabulary."""

    def __init__(self, registry: DomainConceptRegistry):
        self.registry = registry

    def normalize(self, query: str) -> Tuple[str, List[str]]:
        tokens = query.split()
        normalized: List[str] = []
        matched: List[str] = []
        for tok in tokens:
            concept, score = self.registry.match(tok)
            if score >= 0.8:
                normalized.append(concept)
                matched.append(concept)
            else:
                normalized.append(tok)
        normalized_query = " ".join(normalized)
        logger.debug("Normalized query '%s' -> '%s'", query, normalized_query)
        return normalized_query, matched
