import os
from typing import List, Optional

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - anthropic optional
    Anthropic = None  # type: ignore


class LLMGenerator:
    """Minimal wrapper around the Anthropic client."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context_sentences: List[str],
        instruction: Optional[str] = None,
        system_prompt: str = "",
    ) -> str:
        """Call the Anthropic API and return the generated text."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        if Anthropic is None:
            raise ImportError("anthropic package is required to use LLMGenerator")

        client = Anthropic(api_key=self.api_key)
        context = " ".join(context_sentences[:8])
        user_content = f"Context:\n{context}\n\nQuestion:\n{query}"
        if instruction:
            user_content = f"{instruction}\n\n{user_content}"
        messages = [{"role": "user", "content": user_content}]

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            system=system_prompt,
        )
        return "".join(block.text for block in response.content).strip()
