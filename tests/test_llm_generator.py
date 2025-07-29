import os
import types
import sys
import importlib
from unittest.mock import MagicMock
import pytest


def test_generate_with_openai_v1(monkeypatch):
    fake_openai_module = types.SimpleNamespace()
    class FakeResponse:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="Hi"))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = FakeResponse()
    fake_openai_module.OpenAI = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, 'openai', fake_openai_module)
    import backend.llm_generator as llm
    importlib.reload(llm)
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    generator = llm.LLMGenerator()
    result = generator.generate("q", ["c"])
    assert result == "Hi"
    fake_openai_module.OpenAI.assert_called_once_with(api_key="key")
    fake_client.chat.completions.create.assert_called_once()

def test_generate_no_api_key(monkeypatch):
    fake_openai_module = types.SimpleNamespace(OpenAI=MagicMock())
    monkeypatch.setitem(sys.modules, 'openai', fake_openai_module)
    import backend.llm_generator as llm
    importlib.reload(llm)
    generator = llm.LLMGenerator()
    with pytest.raises(ValueError):
        generator.generate("q", ["c"])
