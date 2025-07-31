
"""Model wrappers for extractive and generative QA."""

from typing import List, Dict, Any, Optional, Union

try:  # Optional torch dependency
    import torch
except Exception:  # pragma: no cover - allow missing torch
    torch = None
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from .llm_generator import LLMGenerator
from .config import Config

class DeBERTaQA:
    """Extractive QA using a DeBERTa model from HuggingFace."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        if torch is None:
            self.available = False
            self.device = "cpu"
            self.model = None
            self.tokenizer = None
            return

        self.device = self._select_device()
        model_name = self.config.deberta.model_name
        cache_dir = getattr(self.config.deberta, "cache_dir", None)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.to(self.device)
            self.available = True
        except Exception:  # pragma: no cover - runtime/installation issues
            self.available = False
            self.model = None
            self.tokenizer = None

    def _select_device(self) -> str:
        if torch is None:
            return "cpu"

        requested = getattr(self.config.deberta, "device", "auto")
        if requested == "cpu":
            return "cpu"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _fallback_answer(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        if not contexts:
            return {"answer": "", "confidence": 0.0}

        q_words = set(question.lower().split())
        best_ctx = max(
            contexts,
            key=lambda c: len(q_words & set(c.lower().split())),
            default="",
        )
        score = len(q_words & set(best_ctx.lower().split()))
        confidence = min(1.0, score / (len(q_words) or 1))
        return {"answer": best_ctx.strip(), "confidence": confidence}

    def answer(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        if not contexts:
            return {"answer": "", "confidence": 0.0}

        if not self.available:
            return self._fallback_answer(question, contexts)

        context = " ".join(contexts)
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.config.deberta, "max_length", 512),
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        max_len = getattr(self.config.deberta, "max_answer_length", 30)
        start_idx, end_idx = 0, 0
        max_score = float("-inf")
        for i in range(len(start_logits)):
            for j in range(i, min(i + max_len, len(end_logits))):
                score = start_logits[i] + end_logits[j]
                if score > max_score:
                    start_idx, end_idx, max_score = i, j, score

        answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)
        confidence = float((start_probs[start_idx] * end_probs[end_idx]).item())

        return {"answer": answer, "confidence": confidence}


class QwenGenerator:
    """Generative model wrapper for the Qwen LLM."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        if torch is None:
            self.available = False
            self.device = "cpu"
            self.model = None
            self.tokenizer = None
            return

        self.device = self._select_device()
        model_name = self.config.qwen.model_name
        cache_dir = getattr(self.config.qwen, "cache_dir", None)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=getattr(self.config.qwen, "trust_remote_code", False),
                cache_dir=cache_dir,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=getattr(self.config.qwen, "trust_remote_code", False),
                cache_dir=cache_dir,
            )
            self.model.to(self.device)
            self.available = True
        except Exception:  # pragma: no cover - runtime/installation issues
            self.available = False
            self.model = None
            self.tokenizer = None

    def _select_device(self) -> str:
        if torch is None:
            return "cpu"

        requested = getattr(self.config.qwen, "device", "auto")
        if requested == "cpu":
            return "cpu"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"



    def generate(
        self,
        query: str,
        contexts: List[str],
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an answer using the underlying LLM."""
        context = " ".join(contexts)

        if self.available:
            prompt_parts = []
            if instruction:
                prompt_parts.append(instruction)
            prompt_parts.append(query)
            prompt_parts.append(context)
            prompt = "\n\n".join(prompt_parts)
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=getattr(self.config.qwen, "max_new_tokens", 128),
                        do_sample=getattr(self.config.qwen, "do_sample", False),
                        temperature=getattr(self.config.qwen, "temperature", 0.7),
                    )
                answer = self.tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                return {"answer": answer, "confidence": 0.6}
            except Exception:
                pass

        try:
            llm = LLMGenerator()
            full_query = f"{instruction}\n\n{query}" if instruction else query
            answer = llm.generate(full_query, contexts)
            return {"answer": answer, "confidence": 0.6}
        except Exception:
            return {"answer": "", "confidence": 0.0}
