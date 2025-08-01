"""Claude-based QA models for the RAG system."""

from typing import List, Dict, Any, Optional

import logging
from .llm_generator import LLMGenerator
from .config import Config


logger = logging.getLogger(__name__)



class ClaudeQA:
    """Simple wrapper that uses Claude for answer generation."""

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

        self.generator = LLMGenerator(model=model_name)


    def generate(
        self,
        query: str,
        contexts: List[str],
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:

        prompt_instruction = instruction
        try:
            answer = self.generator.generate(
                query,
                contexts,
                instruction=prompt_instruction,
                system_prompt=self.system_prompt,
            )
            return {"answer": answer, "confidence": 0.6}
        except Exception as e:
            logger.exception("Claude generation failed")
            return {"answer": f"Error: {e}", "confidence": 0.0}

        prompt = f"{instruction}\n\n{query}" if instruction else query
        try:
            answer = self.generator.generate(prompt, contexts)
            return {"answer": answer, "confidence": 0.6}
        except Exception:
            return {"answer": "", "confidence": 0.0}


    def answer(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        return self.generate(question, contexts)
