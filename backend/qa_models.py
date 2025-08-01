"""Claude-based QA models for the RAG system."""

from typing import List, Dict, Any, Optional

from .llm_generator import LLMGenerator
from .config import Config


class ClaudeQA:
    """Simple wrapper that uses Claude for answer generation."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        model_name = getattr(self.config.claude, "model_name", "claude-3-5-haiku-20241022")
        self.generator = LLMGenerator(model=model_name)

    def generate(
        self,
        query: str,
        contexts: List[str],
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = f"{instruction}\n\n{query}" if instruction else query
        try:
            answer = self.generator.generate(prompt, contexts)
            return {"answer": answer, "confidence": 0.6}
        except Exception:
            return {"answer": "", "confidence": 0.0}

    def answer(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        return self.generate(question, contexts)
