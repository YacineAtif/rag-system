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
            enable_fact_checking=True,
            enable_consistency_validation=True,
            enable_citation_verification=True,
            hallucination_detection_threshold=0.3,
            source_match_threshold=0.5,
        )
        cfg = OODDetectionConfig(
            similarity_threshold=0.1,
            graph_connectivity_threshold=0.6,
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
            graph_connectivity=0.9,
            retrieved_relevances=[0.8, 0.9],
            token_probs=[0.9, 0.9, 0.9],
            answer="Sample theory deals with uncertainty [1]",
            sources=["Sample theory deals with uncertainty"],
        )
        self.assertTrue(res["allow_generation"])
        self.assertTrue(res.get("response_verified"))

    def test_rejects_low_similarity(self):
        res = self.detector.process(
            query="Random question",
            similarity=0.01,
            graph_connectivity=0.9,
            retrieved_relevances=[0.9, 0.8],
            token_probs=[0.9],
        )
        self.assertFalse(res["allow_generation"])

    def test_retrieval_quality_gate(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            graph_connectivity=0.9,
            retrieved_relevances=[0.2],  # not enough passages
            token_probs=[0.9, 0.9],
        )
        self.assertFalse(res["retrieval_ok"])
        self.assertFalse(res["allow_generation"])

    def test_response_verification_hallucination(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            graph_connectivity=0.9,
            retrieved_relevances=[0.9, 0.9],
            token_probs=[0.9, 0.9],
            answer="Completely unrelated answer",
            sources=["Sample theory deals with uncertainty"],
        )
        self.assertFalse(res.get("response_verified"))

    def test_response_verification_paraphrase(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            graph_connectivity=0.9,
            retrieved_relevances=[0.9, 0.9],
            token_probs=[0.9, 0.9],
            answer="Sample theory deals with uncertain information",
            sources=["Sample theory deals with uncertainty"],
        )
        self.assertTrue(res.get("response_verified"))

    def test_response_verification_single_match_many_sources(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            graph_connectivity=0.9,
            retrieved_relevances=[0.9, 0.9],
            token_probs=[0.9, 0.9],
            answer="Sample theory deals with uncertain information",
            sources=[
                "Completely unrelated passage about other topics",
                "Sample theory deals with uncertainty",
            ],
        )
        self.assertTrue(res.get("response_verified"))

    def test_short_answer_verification(self):
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            graph_connectivity=0.9,
            retrieved_relevances=[0.9, 0.9],
            token_probs=[0.9, 0.9],
            answer="Sample Alias",
            sources=["Sample theory also called Sample Alias theory"],
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
            graph_connectivity=0.9,
            retrieved_relevances=[0.9, 0.9],
            token_probs=[0.25, 0.25, 0.25, 0.25],
        )
        self.assertFalse(res["allow_generation"])

    def test_equal_vector_and_graph_results_pass(self):
        vector_relevances = [0.8] * 10
        graph_count = 10
        graph_connectivity = graph_count / max(len(vector_relevances), 1)
        res = self.detector.process(
            query="what is sample theory?",
            similarity=0.9,
            graph_connectivity=graph_connectivity,
            retrieved_relevances=vector_relevances,
            token_probs=[0.9, 0.9],
        )
        self.assertTrue(res["allow_generation"])

    def test_performance_large_relevances(self):
        relevances = [0.9] * 10000
        start = time.perf_counter()
        res = self.detector.process(
            query="Explain sample theory",
            similarity=0.9,
            graph_connectivity=0.9,
            retrieved_relevances=relevances,
            token_probs=[0.9, 0.9],
        )
        elapsed = time.perf_counter() - start
        self.assertTrue(elapsed < 0.5, f"Processing took too long: {elapsed}")
        self.assertTrue(res["allow_generation"])


if __name__ == "__main__":
    unittest.main()
