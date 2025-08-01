from unittest.mock import patch, MagicMock
from backend import qa_models


def test_llm_generator_requires_api_key():
    with patch.dict('os.environ', {}, clear=True):
        gen = qa_models.LLMGenerator()
        try:
            gen.generate('q', ['ctx'])
        except ValueError as e:
            assert 'ANTHROPIC_API_KEY' in str(e)
        else:
            assert False, 'ValueError not raised'


def test_claude_generation_calls_api():
    from types import SimpleNamespace
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [SimpleNamespace(text="hi")]
    mock_client.messages.create.return_value = mock_resp
    with patch('backend.llm_generator.Anthropic', return_value=mock_client):
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'k'}):
            gen = qa_models.LLMGenerator()
            result = gen.generate('q', ['ctx'])
    assert result == 'hi'
    mock_client.messages.create.assert_called_once()


def test_claudeqa_generate():
    with patch('backend.qa_models.LLMGenerator.generate', return_value='answer') as mock_gen:
        qa = qa_models.ClaudeQA()
        res = qa.generate('q', ['c'])
    assert res == {'answer': 'answer', 'confidence': 0.6}
    mock_gen.assert_called_once()
