import unittest
from types import SimpleNamespace
from typing import Optional

from processing.comprehensive_extractor import ComprehensiveExtractor


class DummyClaude:
    """Simple stub that returns predetermined responses."""

    def generate(
        self,
        query: str,
        contexts: list[str],
        instruction: Optional[str] = None,
    ):
        if "Analyze this document structure" in query:
            return {"answer": '{}'}
        if "CRITICAL: Extract ALL organizational entities" in query:
            # Only return a subset of organisations to simulate missing data
            return {
                "answer": (
                    '{"entities": ['
                    '{"name": "I2Connect", "type": "Project"},'
                    '{"name": "Scania", "type": "Organization"}'
                    '], "relationships": ['
                    '{"source": "Scania", "target": "I2Connect", "relationship": "INDUSTRIAL_PARTNER"}'
                    ']}'
                )
            }
        return {"answer": '{"entities": [], "relationships": []}'}


class TestComprehensiveExtractor(unittest.TestCase):
    def test_missing_organizations_are_added(self):
        doc_content = (
            "## Collaborators\n"
            "The I2Connect project partners with University of Skövde, "
            "Scania, Smart Eye, and Viscando Technologies."
        )
        doc = SimpleNamespace(content=doc_content, meta={"source": "test"})

        extractor = ComprehensiveExtractor(claude_extractor=DummyClaude())
        result = extractor.comprehensive_knowledge_extraction([doc])

        names = {e["name"] for e in result["entities"]}
        self.assertIn("University of Skövde", names)
        self.assertIn("Viscando Technologies", names)

        rels = result["relationships"]
        self.assertTrue(
            any(r["source"] == "University of Skövde" and r["target"] == "I2Connect" for r in rels)
        )
        self.assertTrue(
            any(r["source"] == "Viscando Technologies" and r["target"] == "I2Connect" for r in rels)
        )


if __name__ == "__main__":
    unittest.main()
