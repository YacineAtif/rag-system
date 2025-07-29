import os
from typing import List

try:
    from openai import OpenAI
    OPENAI_V1 = True
except Exception:  # pragma: no cover - fallback for older openai or missing pkg
    try:
        import openai  # type: ignore
        OPENAI_V1 = False
    except Exception:
        OpenAI = None  # type: ignore
        openai = None  # type: ignore
        OPENAI_V1 = None


class LLMGenerator:
    """Simple wrapper around OpenAI ChatCompletion API."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(self, query: str, context_sentences: List[str]) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        context = " ".join(context_sentences[:4])
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using the provided context. "
                    "Respond in fluent, natural prose. Avoid bullet lists and repetition."
                ),
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}",
            },
        ]
        if OPENAI_V1 is True:
            client = OpenAI(api_key=self.api_key)
            try:
                response = client.chat.completions.create(model=self.model, messages=messages)
            except Exception as e:  # pragma: no cover - runtime errors
                raise RuntimeError(f"OpenAI API call failed: {e}") from e
            return response.choices[0].message.content.strip()
        elif OPENAI_V1 is False:
            openai.api_key = self.api_key
            try:
                response = openai.ChatCompletion.create(model=self.model, messages=messages)
            except Exception as e:
                raise RuntimeError(f"OpenAI API call failed: {e}") from e
            return response.choices[0].message["content"].strip()
        else:
            raise ImportError("openai package is required to use LLMGenerator")
