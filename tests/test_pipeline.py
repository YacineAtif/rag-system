"""
Tests for the hybrid processing pipeline.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch
import types

# Provide lightweight stubs so that importing the main module does not require
# heavy optional dependencies during tests.
haystack_stub = types.ModuleType("haystack")

class DummyDocument:
    def __init__(self, content="", meta=None, score=0.0):
        self.content = content
        self.meta = meta or {}
        self.score = score

class DummyPipeline:
    def __init__(self):
        pass
    def add_component(self, *a, **kw):
        pass
    def connect(self, *a, **kw):
        pass

haystack_stub.Document = DummyDocument
haystack_stub.Pipeline = DummyPipeline

sys.modules.setdefault("haystack", haystack_stub)
sys.modules.setdefault("haystack.components", types.ModuleType("haystack.components"))
writers_stub = types.ModuleType("haystack.components.writers")
writers_stub.DocumentWriter = object
sys.modules.setdefault("haystack.components.writers", writers_stub)
embedders_stub = types.ModuleType("haystack.components.embedders")
embedders_stub.SentenceTransformersTextEmbedder = object
embedders_stub.SentenceTransformersDocumentEmbedder = object
sys.modules.setdefault("haystack.components.embedders", embedders_stub)
store_stub = types.ModuleType("haystack_integrations.document_stores.weaviate")
store_stub.WeaviateDocumentStore = object
sys.modules.setdefault("haystack_integrations.document_stores.weaviate", store_stub)
retriever_stub = types.ModuleType("haystack_integrations.components.retrievers.weaviate")
retriever_stub.WeaviateEmbeddingRetriever = object
sys.modules.setdefault("haystack_integrations.components.retrievers.weaviate", retriever_stub)
requests_stub = types.ModuleType("requests")
def _dummy_get(*a, **kw):
    class R:
        status_code = 200
    return R()
requests_stub.get = _dummy_get
sys.modules.setdefault("requests", requests_stub)
yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda s: {}
sys.modules.setdefault("yaml", yaml_stub)

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
        self.assertEqual(result.answer, '')

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
                answer = generator.generate('Who was Alice\'s collaborator?', sentences, 'entity', [])
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

    def test_boost_documents(self):
        from types import SimpleNamespace
        from weaviate_rag_pipeline_transformers import boost_documents

        docs = [
            SimpleNamespace(content='a', meta={'section_name': 'partners'}, score=1.0),
            SimpleNamespace(content='b', meta={'section_name': 'intro'}, score=1.0),
        ]
        mock_config = {
            'retrieval': {'section_name_boost': 2.0},
            'section_priorities': {
                'partnership_queries': {
                    'priority_sections': ['partners'],
                    'boost_factor': 3.0,
                }
            },
        }
        with patch('weaviate_rag_pipeline_transformers.CONFIG', mock_config):
            boosted = boost_documents(docs, 'partnership')
        self.assertGreater(boosted[0].score, boosted[1].score)
        self.assertEqual(boosted[0].meta['section_name'], 'partners')

    def test_query_classifier_partnership(self):
        from weaviate_rag_pipeline_transformers import QueryClassifier

        cfg = {
            'section_priorities': {
                'partnership_queries': {
                    'keywords': ['partner'],
                }
            }
        }
        with patch('weaviate_rag_pipeline_transformers.CONFIG', cfg):
            cls = QueryClassifier(cfg)
            q_type = cls.classify('Who are the project partners?')
        self.assertEqual(q_type, 'partnership')

if __name__ == '__main__':
    unittest.main()
