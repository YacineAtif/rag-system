from typing import List, Dict, Any
from .llm_generator import LLMGenerator

class DeBERTaQA:
    """Lightweight placeholder for a DeBERTa-based QA model."""

    def __init__(self, config=None):
        self.config = config

    def answer(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        """Return an answer and confidence score."""
        if not contexts:
            return {"answer": "", "confidence": 0.0}

        q_words = set(question.lower().split())
        best_context = ""
        best_score = 0
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            score = len(q_words & ctx_words)
            if score > best_score:
                best_score = score
                best_context = ctx
        confidence = min(1.0, best_score / (len(q_words) or 1))
        return {"answer": best_context.strip(), "confidence": confidence}


class QwenGenerator:
    """Simple wrapper around an LLM generator representing Qwen."""

    def __init__(self, config=None):
        self.config = config

    def generate(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        try:
            llm = LLMGenerator()
            answer = llm.generate(query, contexts)
            confidence = 0.6
        except Exception:
            answer = ""
            confidence = 0.0
        return {"answer": answer, "confidence": confidence}
