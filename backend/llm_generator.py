import os
from typing import List

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - anthropic optional
    Anthropic = None  # type: ignore


class LLMGenerator:
    """Simple wrapper around the Claude API."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = model

    def generate(self, query: str, context_sentences: List[str]) -> str:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        if Anthropic is None:
            raise ImportError("anthropic package is required to use LLMGenerator")
        client = Anthropic(api_key=self.api_key)
        context = " ".join(context_sentences[:8])
        messages = [
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        try:  # pragma: no cover - runtime errors
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=messages,
            )
            return "".join(block.text for block in response.content).strip()
        except Exception as e:  # pragma: no cover - runtime errors
            raise RuntimeError(f"Anthropic API call failed: {e}") from e
