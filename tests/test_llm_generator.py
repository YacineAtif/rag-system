from unittest.mock import patch
from backend import qa_models


class DummyTokenizer:
    def __call__(self, *args, **kwargs):
        import torch
        return {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return "dummy"


class DummyDebertaModel:
    def to(self, device):
        pass

    def __call__(self, **inputs):
        import torch
        start = torch.tensor([[0.0, 4.0, 0.0]])
        end = torch.tensor([[0.0, 0.0, 4.0]])
        return type("O", (), {"start_logits": start, "end_logits": end})


class DummyQwenModel:
    def to(self, device):
        pass

    def generate(self, **kwargs):
        return kwargs["input_ids"]


def test_qwen_generator_fallback():
    with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=OSError), \
         patch("backend.qa_models.LLMGenerator.generate", return_value="hello") as mock_llm:
        gen = qa_models.QwenGenerator()
        result = gen.generate("q", ["ctx"])
    assert result == {"answer": "hello", "confidence": 0.6}
    mock_llm.assert_called_once_with("q", ["ctx"])


def test_deberta_answer_from_model():
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), \
         patch("transformers.AutoModelForQuestionAnswering.from_pretrained", return_value=DummyDebertaModel()):
        de = qa_models.DeBERTaQA()
        res = de.answer("q", ["ctx"])
    assert res["answer"] == "dummy"
    assert 0.0 <= res["confidence"] <= 1.0


def test_deberta_fallback_no_model():
    with patch("transformers.AutoModelForQuestionAnswering.from_pretrained", side_effect=OSError), \
         patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError):
        de = qa_models.DeBERTaQA()
        res = de.answer("who", ["Alice went home"])
    assert res["answer"] == "Alice went home"
    assert res["confidence"] > 0
