import time
import unittest
from multi_layer_ood import (
    MultiLayerOODDetector,
    OODDetectionConfig,
    KeywordTiers,
    QualityGates,
    ResponseVerificationConfig,
)


class TestMultiLayerOOD(unittest.TestCase):
    def setUp(self):
        keywords = KeywordTiers(
            tier_1_critical={"core": ["sample theory"]},
            tier_2_important={"tech": ["widget"]},
            tier_3_contextual={"general": ["system"]},
        )
        quality = QualityGates(
            min_context_relevance=0.5,
            min_passage_count=2,
            max_contradiction_score=0.2,
            min_information_density=0.4,
        )
        verify = ResponseVerificationConfig(
            relevance_threshold=0.5,
            lexical_similarity_threshold=0.3,
            max_entropy_threshold=2.0,
            mean_entropy_threshold=1.5,
            sar_threshold=0.0,
        )
        cfg = OODDetectionConfig(
            similarity_threshold=0.1,
            graph_connectivity_threshold=0.1,
            generation_confidence_threshold=0.5,
            keywords=keywords,
            quality_gates=quality,
            response_verification=verify,
        )
        self.detector = MultiLayerOODDetector(cfg)

    def test_allows_in_domain(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=[
                "Sample theory deals with uncertainty",
                "More details on sample theory",
            ],
            token_probs=[0.9, 0.9, 0.9],
            graph_results=["sample -> theory", "theory -> uncertainty"],
            answer="Sample theory deals with uncertainty [1]",
        )
        self.assertTrue(res["allow_generation"])
        self.assertTrue(res.get("response_verified"))

    def test_rejects_low_similarity(self):
        res = self.detector.process(
            query="Random question",
            similarity=0.01,
            retrieved_passages=[
                "Sample theory deals with uncertainty",
                "More details on sample theory",
            ],
            token_probs=[0.9],
            graph_results=[],
        )
        self.assertFalse(res["allow_generation"])

    def test_retrieval_quality_gate(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=["Irrelevant text"],  # not enough passages
            token_probs=[0.9, 0.9],
            graph_results=[],
        )
        self.assertFalse(res["retrieval_ok"])
        self.assertFalse(res["allow_generation"])

    def test_response_verification_hallucination(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=[
                "Sample theory deals with uncertainty",
                "More details on sample theory",
            ],
            token_probs=[0.9, 0.9],
            graph_results=["sample -> theory"],
            answer="Completely unrelated answer",
        )
        self.assertFalse(res.get("response_verified"))

    def test_response_verification_paraphrase(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=[
                "Sample theory deals with uncertainty",
                "More details on sample theory",
            ],
            token_probs=[0.9, 0.9],
            graph_results=["sample -> theory"],
            answer="Sample theory deals with uncertain information",
        )
        self.assertTrue(res.get("response_verified"))

    def test_response_verification_single_match_many_sources(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=[
                "Completely unrelated passage about other topics",
                "Sample theory deals with uncertainty",
            ],
            token_probs=[0.9, 0.9],
            graph_results=["sample -> theory"],
            answer="Sample theory deals with uncertain information",
        )
        self.assertTrue(res.get("response_verified"))

    def test_short_answer_verification(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=[
                "Sample theory also called Sample Alias theory",
                "More details on sample theory",
            ],
            token_probs=[0.9, 0.9],
            graph_results=["sample -> theory"],
            answer="Sample Alias",
        )
        self.assertTrue(res.get("response_verified"))

    def test_complex_query_analysis(self):
        analysis = self.detector.query_analyzer.analyze(
            "How does sample theory compare to probability and statistics in safety systems?"
        )
        self.assertEqual(analysis["type"], "comparative")
        self.assertEqual(analysis["complexity"], "complex")

    def test_generation_guard_low_confidence(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=[
                "Sample theory deals with uncertainty",
                "More details on sample theory",
            ],
            token_probs=[0.25, 0.25, 0.25, 0.25],
            graph_results=["sample -> theory"],
        )
        self.assertTrue(res["allow_generation"])

    def test_equal_vector_and_graph_results_pass(self):
        vector_relevances = [0.8] * 10
        res = self.detector.process(
            query="what is sample theory?",
            similarity=0.9,
            retrieved_passages=["sample theory"] * 10,
            token_probs=[0.9, 0.9],
            graph_results=[f"sample -> theory {i}" for i in range(10)],
        )
        self.assertTrue(res["allow_generation"])

    def test_performance_large_relevances(self):
        passages = ["sample theory"] * 10000
        start = time.perf_counter()
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            retrieved_passages=passages,
            token_probs=[0.9, 0.9],
            graph_results=["sample -> theory"],
        )
        elapsed = time.perf_counter() - start
        self.assertTrue(elapsed < 0.5, f"Processing took too long: {elapsed}")
        self.assertTrue(res["allow_generation"])


if __name__ == "__main__":
    unittest.main()
