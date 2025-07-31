from unittest.mock import patch
from backend import qa_models


def test_qwen_generator_success():
    with patch('backend.qa_models.LLMGenerator.generate', return_value='hello') as mock_llm:
        gen = qa_models.QwenGenerator()
        result = gen.generate('q', ['ctx'])
    assert result == {'answer': 'hello', 'confidence': 0.6}
    mock_llm.assert_called_once_with('q', ['ctx'])


def test_qwen_generator_failure():
    with patch('backend.qa_models.LLMGenerator.generate', side_effect=RuntimeError):
        gen = qa_models.QwenGenerator()
        result = gen.generate('q', ['ctx'])
    assert result == {'answer': '', 'confidence': 0.0}


def test_deberta_answer_selects_best_context():
    de = qa_models.DeBERTaQA()
    result = de.answer('Who is Alice', ['Bob went home', 'Alice went home'])
    assert result['answer'] == 'Alice went home'
    # q_words = {"who","is","alice"}; best_score = 1; len(q_words)=3 => conf=1/3
    assert abs(result['confidence'] - (1/3)) < 0.0001


def test_deberta_no_contexts():
    de = qa_models.DeBERTaQA()
    result = de.answer('Any', [])
    assert result == {'answer': '', 'confidence': 0.0}
