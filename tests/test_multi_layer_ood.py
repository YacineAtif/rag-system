import unittest
from multi_layer_ood import MultiLayerOODDetector, OODDetectionConfig, KeywordTiers


class TestMultiLayerOOD(unittest.TestCase):
    def setUp(self):
        keywords = KeywordTiers(tier_1_critical={"core": ["evidence theory"]})
        cfg = OODDetectionConfig(keywords=keywords)
        self.detector = MultiLayerOODDetector(cfg)

    def test_allows_in_domain(self):
        result = self.detector.process(
            query="What is evidence theory?",
            similarity=0.9,
            retrieved_relevances=[0.9, 0.8],
            token_probs=[0.99, 0.98, 0.97],
        )
        self.assertTrue(result["allow_generation"])

    def test_rejects_low_similarity(self):
        result = self.detector.process(
            query="Unrelated question?",
            similarity=0.05,
            retrieved_relevances=[0.9, 0.9],
            token_probs=[0.9, 0.9],
        )
        self.assertFalse(result["allow_generation"])


if __name__ == "__main__":
    unittest.main()
