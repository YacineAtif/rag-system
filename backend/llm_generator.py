import os
from typing import List

import openai


class LLMGenerator:
    """Simple wrapper around OpenAI ChatCompletion API."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(self, query: str, context_sentences: List[str]) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai.api_key = self.api_key
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
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        return response.choices[0].message["content"].strip()
