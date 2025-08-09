import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
try:
    from sentence_transformers import SentenceTransformer
    _HAS_TRANSFORMER = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TRANSFORMER = False

logger = logging.getLogger(__name__)


class DomainConceptRegistry:
    """Registry of known domain concepts and their embeddings."""

    def __init__(self, concepts: Sequence[str], synonyms: Optional[Dict[str, str]] = None):
        self.concepts = [c.lower() for c in concepts]
        self.synonyms = {k.lower(): v.lower() for k, v in (synonyms or {}).items()}
        if _HAS_TRANSFORMER:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                self._embedder = SentenceTransformer(model_name, )
                self._use_transformer = True
            except Exception as exc:  # pragma: no cover - network failure fallback
                logger.warning("Falling back to HashingVectorizer embeddings due to: %s", exc)
                self._use_transformer = False
                self._vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")
        else:  # pragma: no cover - sklearn fallback
            self._use_transformer = False
            self._vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")
        self.embeddings = {c: self._embed(c) for c in self.concepts}

    def _embed(self, text: str) -> np.ndarray:
        if self._use_transformer:
            logger.debug("Embedding text '%s'", text)
            return np.asarray(
                self._embedder.encode(
                    text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                ),
                dtype=np.float32,
            )

        return self._vectorizer.transform([text]).toarray()[0]

    def resolve(self, term: str) -> str:
        term_l = term.lower()
        return self.synonyms.get(term_l, term_l)

    def match(self, term: str) -> Tuple[str, float]:
        """Return the closest concept and similarity score."""
        term = self.resolve(term)
        emb = self._embed(term)
        best, best_score = "", 0.0
        for concept, c_emb in self.embeddings.items():
            denom = (np.linalg.norm(emb) * np.linalg.norm(c_emb)) or 1.0
            score = float(np.dot(emb, c_emb) / denom)
            if score > best_score:
                best, best_score = concept, score
        return best, best_score

    def extract_from_query(self, query: str) -> List[str]:
        tokens = re.findall(r"\w+", query.lower())
        matched: List[str] = []
        for tok in tokens:
            concept, score = self.match(tok)
            if score >= 0.8:
                matched.append(concept)
        return matched
