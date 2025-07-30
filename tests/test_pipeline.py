"""
Tests for the hybrid processing pipeline.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from processing.hybrid_pipeline import HybridPipeline, ProcessingMode, QueryResult
    from backend.config import Config
    from weaviate_rag_pipeline_transformers import TextProcessor, AnswerGenerator
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

@unittest.skipUnless(PIPELINE_AVAILABLE, "HybridPipeline not available")
class TestHybridPipeline(unittest.TestCase):
    """Test hybrid processing pipeline functionality."""

    def setUp(self):
        self.config = Config()
        self.pipeline = HybridPipeline(self.config)

    def test_pipeline_initialization(self):
        self.assertIsNotNone(self.pipeline)
        self.assertFalse(self.pipeline.initialized)
        result = self.pipeline.initialize()
        self.assertIsInstance(result, bool)
        if result:
            self.assertTrue(self.pipeline.initialized)

    def test_processing_modes_enum(self):
        self.assertTrue(hasattr(ProcessingMode, 'EXTRACTIVE_ONLY'))
        self.assertTrue(hasattr(ProcessingMode, 'GENERATIVE_ONLY'))
        self.assertTrue(hasattr(ProcessingMode, 'HYBRID_AUTO'))
        self.assertIsInstance(ProcessingMode.EXTRACTIVE_ONLY.value, str)

    def test_query_processing_structure(self):
        self.pipeline.initialize()
        question = "What is artificial intelligence?"
        contexts = ["AI is intelligence demonstrated by machines."]
        result = self.pipeline.process_query(question, contexts)
        self.assertTrue(hasattr(result, 'answer'))
        self.assertTrue(hasattr(result, 'confidence'))
        self.assertTrue(hasattr(result, 'processing_mode'))

    def test_empty_contexts(self):
        self.pipeline.initialize()
        question = "What is AI?"
        contexts = []
        result = self.pipeline.process_query(question, contexts)
        self.assertTrue(hasattr(result, 'answer'))
        self.assertGreater(len(result.answer), 0)

    def test_pipeline_status(self):
        status = self.pipeline.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn('initialized', status)
        self.pipeline.initialize()
        status_after = self.pipeline.get_status()
        self.assertTrue(status_after['initialized'])

    def test_uninitialized_pipeline(self):
        question = "test"
        contexts = ["context"]
        result = self.pipeline.process_query(question, contexts)
        self.assertEqual(result.processing_mode, 'error')

    def test_answer_generator_with_llm(self):
        processor = TextProcessor()
        generator = AnswerGenerator(processor)
        sentences = ["AI is intelligence demonstrated by machines."]
        with patch('backend.qa_models.QwenGenerator.generate', return_value={'answer': 'A short answer.', 'confidence': 0.9}):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'dummy'}):
                answer = generator.generate('What is AI?', sentences, 'general', [])
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

    def test_definition_answer_uses_llm(self):
        processor = TextProcessor()
        generator = AnswerGenerator(processor)
        sentences = ["Evidence Theory is a mathematical framework for reasoning with uncertainty."]
        with patch('backend.qa_models.DeBERTaQA.answer', return_value={'answer': 'DE answer', 'confidence': 0.8}) as mock_deb:
            with patch('backend.qa_models.QwenGenerator.generate', return_value={'answer': 'fallback', 'confidence': 0.5}) as mock_qwen:
                answer = generator.generate('What is evidence theory?', sentences, 'definition', [])
        self.assertEqual(answer, 'DE answer')
        mock_deb.assert_called_once()
        mock_qwen.assert_not_called()

    def test_entity_relationship_query_uses_llm(self):
        processor = TextProcessor()
        generator = AnswerGenerator(processor)
        sentences = ["Alice and Bob collaborated on the project."]
        with patch('backend.qa_models.QwenGenerator.generate', return_value={'answer': 'Alice worked with Bob on the project.', 'confidence': 0.8}):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'dummy'}):
                answer = generator.generate('Who did Alice collaborate with?', sentences, 'entity', [])
        self.assertEqual(answer, 'Alice worked with Bob on the project.')

    def test_entity_list_fallback(self):
        processor = TextProcessor()
        generator = AnswerGenerator(processor)
        sentences = ["Charlie, Dana and Erin attended the meeting."]
        answer = generator.generate('Who was mentioned?', sentences, 'entity', [])
        self.assertIn('Entities mentioned:', answer)
        self.assertIn('Charlie', answer)

    def test_query_result_metadata_model(self):
        self.pipeline.initialize()
        with patch('backend.qa_models.DeBERTaQA.answer', return_value={'answer': 'fact', 'confidence': 0.9}):
            result = self.pipeline.process_query('What is AI?', ['AI context'], ProcessingMode.EXTRACTIVE_ONLY)
        self.assertEqual(result.metadata.get('model'), 'deberta')
        self.assertAlmostEqual(result.metadata.get('model_confidence'), 0.9)

if __name__ == '__main__':
    unittest.main()
