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
# Provide minimal stubs for optional integrations
store_stub.WeaviateDocumentStore = object
sys.modules.setdefault("haystack_integrations.document_stores.weaviate", store_stub)
retriever_stub = types.ModuleType("haystack_integrations.components.retrievers.weaviate")
retriever_stub.WeaviateEmbeddingRetriever = object
sys.modules.setdefault("haystack_integrations.components.retrievers.weaviate", retriever_stub)

# Stub neo4j driver to avoid requiring the real package
neo4j_stub = types.ModuleType("neo4j")
class _DummyDriver:
    def session(self):
        class _Session:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                pass
            def run(self, *a, **kw):
                class _R:
                    def single(self):
                        return {"count": 0}
                return _R()
        return _Session()
neo4j_stub.GraphDatabase = types.SimpleNamespace(driver=lambda *a, **kw: _DummyDriver())
sys.modules.setdefault("neo4j", neo4j_stub)

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
        with patch('backend.qa_models.ClaudeQA.generate', return_value={'answer': '', 'confidence': 0.0}):
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
        with patch('backend.qa_models.ClaudeQA.generate', return_value={'answer': 'A short answer.', 'confidence': 0.9}):
            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'dummy'}):
                answer = generator.generate('What is AI?', sentences, 'general', [])
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

    def test_definition_answer_uses_llm(self):
        processor = TextProcessor()
        generator = AnswerGenerator(processor)
        sentences = ["Evidence Theory is a mathematical framework for reasoning with uncertainty."]
        with patch('backend.qa_models.ClaudeQA.generate', return_value={'answer': 'DE answer', 'confidence': 0.8}) as mock_claude:
            answer = generator.generate('What is evidence theory?', sentences, 'definition', [])
        self.assertEqual(answer, 'DE answer')
        mock_claude.assert_called()

    def test_entity_relationship_query_uses_llm(self):
        processor = TextProcessor()
        generator = AnswerGenerator(processor)
        sentences = ["Alice and Bob collaborated on the project."]
        with patch('backend.qa_models.ClaudeQA.generate', return_value={'answer': 'Alice worked with Bob on the project.', 'confidence': 0.8}):
            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'dummy'}):
                answer = generator.generate('Who was Alice\'s collaborator?', sentences, 'entity', [])
        self.assertEqual(answer, 'Alice worked with Bob on the project.')


    def test_query_result_metadata_model(self):
        self.pipeline.initialize()
        with patch('backend.qa_models.ClaudeQA.generate', return_value={'answer': 'fact', 'confidence': 0.9}):
            result = self.pipeline.process_query('What is AI?', ['AI context'], ProcessingMode.EXTRACTIVE_ONLY)
        self.assertEqual(result.metadata.get('model'), 'claude')
        self.assertAlmostEqual(result.metadata.get('model_confidence'), 0.9)

    def test_boost_documents(self):
        from types import SimpleNamespace
        from weaviate_rag_pipeline_transformers import boost_documents

        docs = [
            SimpleNamespace(content='a', meta={'section_name': 'partners'}, score=1.0),
            SimpleNamespace(content='b', meta={'section_name': 'intro'}, score=1.0),
        ]
        mock_config = SimpleNamespace(
            retrieval=SimpleNamespace(section_name_boost=2.0),
            section_priorities=SimpleNamespace(
                queries={
                    'partnership_queries': {
                        'priority_sections': ['partners'],
                        'boost_factor': 3.0,
                    }
                }
            ),
        )
        with patch('weaviate_rag_pipeline_transformers.CONFIG', mock_config):
            boosted = boost_documents(docs, 'partnership')
        self.assertGreater(boosted[0].score, boosted[1].score)
        self.assertEqual(boosted[0].meta['section_name'], 'partners')

    def test_query_classifier_partnership(self):
        from weaviate_rag_pipeline_transformers import QueryClassifier
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            section_priorities=SimpleNamespace(
                queries={'partnership_queries': {'keywords': ['partner']}}
            )
        )
        with patch('weaviate_rag_pipeline_transformers.CONFIG', cfg):
            cls = QueryClassifier(cfg)
            q_type = cls.classify('Who are the project partners?')
        self.assertEqual(q_type, 'partnership')

    def test_rag_pipeline_snippet_verification(self):
        from types import SimpleNamespace
        from weaviate_rag_pipeline_transformers import RAGPipeline

        class DummyNeo4jGraphBuilder:
            def __init__(self, *a, **kw):
                pass

        class DummyDoc:
            def __init__(self, content, score=1.0):
                self.content = content
                self.score = score
                self.meta = {}

        doc_content = (
            "Artificial intelligence enables machines to perform tasks. "
            "It is widely used in many applications."
        )

        with patch(
            'weaviate_rag_pipeline_transformers.Neo4jGraphBuilder',
            DummyNeo4jGraphBuilder,
        ):
            doc_store = SimpleNamespace(filter_documents=lambda: [])
            embedder = SimpleNamespace(run=lambda **kw: {"embedding": [0.1]})
            pipeline = RAGPipeline(doc_store, None, embedder)

        pipeline.hybrid_router = SimpleNamespace(
            hybrid_retrieve=lambda q: {
                "vector_results": [DummyDoc(doc_content)],
                "graph_results": [],
            },
            synthesize_answer=lambda q, res: "Artificial intelligence enables machines to perform tasks.",
        )

        with patch.object(
            pipeline.ood_detector, 'process', return_value={"allow_generation": True}
        ):
            with patch.object(
                pipeline.ood_detector.verifier, 'verify', return_value=(True, {})
            ) as mock_verify:
                result = pipeline.query_with_graph("What is AI?")

        self.assertEqual(
            result["answer"],
            "Artificial intelligence enables machines to perform tasks.",
        )
        called_sources = mock_verify.call_args[0][1]
        self.assertEqual(
            called_sources,
            ["Artificial intelligence enables machines to perform tasks."],
        )
        self.assertLess(len(called_sources[0]), len(doc_content))

if __name__ == '__main__':
    unittest.main()
