import unittest
from multi_layer_ood import ContextRelevanceAssessor, ResponseVerificationConfig


class TestContextRelevanceAssessor(unittest.TestCase):
    def setUp(self):
        self.assessor = ContextRelevanceAssessor(ResponseVerificationConfig())

    def test_identical_text_high_similarity(self):
        scored = self.assessor.score_passages("apple", ["apple"])
        self.assertGreaterEqual(scored[0][1], 0.95)

    def test_different_text_low_similarity(self):
        scored = self.assessor.score_passages("apple", ["banana"])
        self.assertLess(scored[0][1], 0.95)


if __name__ == "__main__":
    unittest.main()
