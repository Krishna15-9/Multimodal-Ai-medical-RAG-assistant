import unittest
from src.qa_system.qa_engine import QAEngine

class TestQAEngineResearchSummary(unittest.TestCase):
    def setUp(self):
        self.qa_engine = QAEngine()

    def test_research_summary_valid_topic(self):
        topic = "intermittent fasting"
        summary = self.qa_engine.get_research_summary(topic)
        self.assertIn("summary", summary)
        self.assertTrue(isinstance(summary["summary"], str))
        self.assertTrue(summary.get("document_count", 0) >= 0)

    def test_research_summary_no_documents(self):
        topic = "some very unlikely topic to have research articles 1234567890"
        summary = self.qa_engine.get_research_summary(topic)
        self.assertIn("summary", summary)
        self.assertTrue(summary.get("document_count", 0) >= 0)

    def test_research_summary_error_handling(self):
        # Simulate error by passing None (should raise or return error)
        summary = self.qa_engine.get_research_summary(None)
        self.assertIn("summary", summary)
        self.assertTrue("error" in summary or summary.get("document_count", 0) == 0)

if __name__ == "__main__":
    unittest.main()
