import unittest
from src.qa_system.qa_engine import QAEngine, QAEngineError

class TestQAEngineEdgeCases(unittest.TestCase):
    def setUp(self):
        self.qa_engine = QAEngine()

    def test_empty_question(self):
        response = self.qa_engine.ask_question("")
        self.assertIn("answer", response)
        self.assertTrue(response["confidence"] >= 0.0)

    def test_none_question(self):
        with self.assertRaises(TypeError):
            self.qa_engine.ask_question(None)

    def test_long_question(self):
        long_question = "What is the effect of " + "a" * 10000
        response = self.qa_engine.ask_question(long_question)
        self.assertIn("answer", response)
        self.assertTrue(response["confidence"] >= 0.0)

    def test_special_characters(self):
        special_question = "!@#$%^&*()_+-=[]{}|;':,.<>/?"
        response = self.qa_engine.ask_question(special_question)
        self.assertIn("answer", response)
        self.assertTrue(response["confidence"] >= 0.0)

    def test_non_string_question(self):
        with self.assertRaises(TypeError):
            self.qa_engine.ask_question(12345)

if __name__ == "__main__":
    unittest.main()
