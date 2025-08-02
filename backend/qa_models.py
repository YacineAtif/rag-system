from typing import List, Dict, Any, Optional
import logging

from .llm_generator import LLMGenerator
from .config import Config

logger = logging.getLogger(__name__)


class ClaudeQA:

    """Thin wrapper around LLMGenerator for question answering."""

    """Wrapper that uses LLMGenerator to communicate with Claude."""


    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        model_name = getattr(self.config.claude, "model_name", "claude-3-5-haiku-20241022")
        self.generator = LLMGenerator(
            model=model_name,
            api_key=self.config.claude.api_key,
            max_tokens=self.config.claude.max_tokens,
            temperature=self.config.claude.temperature,
        )
        self.system_prompt = self.config.prompting.system_prompt or (
            "You are a helpful assistant that summarizes partnership and collaborator information from provided context."
        )

    def generate(
        self,
        query: str,
        contexts: List[str],
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:

        """Generate an answer with Claude."""

        try:
            answer = self.generator.generate(
                query,
                contexts,
                instruction=instruction,
                system_prompt=self.system_prompt,
            )
            return {"answer": answer, "confidence": 0.6}
        except Exception as e:  # pragma: no cover - runtime errors
            logger.exception("Claude generation failed")
            return {"answer": f"Error: {e}", "confidence": 0.0}

    def answer(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        return self.generate(question, contexts)
