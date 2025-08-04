import unittest
from types import SimpleNamespace

from ood_verification import OODVerificationAgent

class DummyDoc:
    def __init__(self, embedding):
        self.embedding = embedding

class DummyStore:
    def filter_documents(self):
        return [DummyDoc([1.0, 0.0])]

class DummyTextEmbedder:
    def run(self, text):
        return {"embedding": [1.0, 0.0]}

class DummyTextProcessor:
    def extract_entities(self, text):
        return ["Entity"]

class DummyResult:
    def __init__(self, data):
        self._data = data
    def single(self):
        return self._data

class DummySession:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def run(self, query, **params):
        if "count(n)" in query:
            return DummyResult({"node_count": 1})
        return DummyResult({"has_relations": True})

class DummyDriver:
    def session(self):
        return DummySession()

class TestOODVerification(unittest.TestCase):
    def test_verify_disabled(self):
        cfg = SimpleNamespace(ood=SimpleNamespace(enabled=False))
        agent = OODVerificationAgent(cfg, DummyDriver(), DummyTextEmbedder(), DummyStore(), DummyTextProcessor())
        self.assertTrue(agent.verify("any"))

    def test_verify_enabled(self):
        cfg = SimpleNamespace(ood=SimpleNamespace(enabled=True, similarity_threshold=0.5, min_neo4j_relations=1))
        agent = OODVerificationAgent(cfg, DummyDriver(), DummyTextEmbedder(), DummyStore(), DummyTextProcessor())
        self.assertTrue(agent.verify("valid query"))

if __name__ == "__main__":
    unittest.main()
