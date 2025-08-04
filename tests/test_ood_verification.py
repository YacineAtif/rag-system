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


class OODTextEmbedder:
    def run(self, text):
        return {"embedding": [0.0, 1.0]}


class DummyTextProcessor:
    def extract_entities(self, text):
        return ["Entity"]


class NoEntityTextProcessor:
    def extract_entities(self, text):
        return []


class UnknownEntityTextProcessor:
    def extract_entities(self, text):
        return ["Pasta"]


class DummyResult:
    def __init__(self, data):
        self._data = data

    def single(self):
        return self._data


class DummySession:
    def __init__(self, has_relations=True):
        self.has_relations = has_relations

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def run(self, query, **params):
        if "count(n)" in query:
            return DummyResult({"node_count": 1})
        return DummyResult({"has_relations": self.has_relations})


class DummyDriver:
    def __init__(self, has_relations=True):
        self.has_relations = has_relations

    def session(self):
        return DummySession(self.has_relations)


class TestOODVerification(unittest.TestCase):
    def _agent(self, text_embedder=None, text_processor=None, driver=None):
        cfg = SimpleNamespace(
            ood=SimpleNamespace(enabled=True, similarity_threshold=0.35, min_neo4j_relations=1)
        )
        text_embedder = text_embedder or DummyTextEmbedder()
        text_processor = text_processor or DummyTextProcessor()
        driver = driver or DummyDriver()
        return OODVerificationAgent(cfg, driver, text_embedder, DummyStore(), text_processor)

    def test_verify_disabled(self):
        cfg = SimpleNamespace(ood=SimpleNamespace(enabled=False))
        agent = OODVerificationAgent(cfg, DummyDriver(), DummyTextEmbedder(), DummyStore(), DummyTextProcessor())
        self.assertTrue(agent.verify("any"))

    def test_domain_query_passes(self):
        agent = self._agent()
        self.assertTrue(agent.verify("what is evidence theory?"))

    def test_query_without_entities_allowed(self):
        agent = self._agent(text_processor=NoEntityTextProcessor())
        self.assertTrue(agent.verify("some generic query"))

    def test_reject_unknown_entity(self):
        agent = self._agent(
            text_processor=UnknownEntityTextProcessor(),
            driver=DummyDriver(has_relations=False),
        )
        self.assertFalse(agent.verify("how to cook pasta"))

    def test_reject_low_similarity(self):
        agent = self._agent(text_embedder=OODTextEmbedder())
        self.assertFalse(agent.verify("nonsense"))


if __name__ == "__main__":
    unittest.main()

