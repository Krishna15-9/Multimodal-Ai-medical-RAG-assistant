import unittest
from src.vector_store.chroma_manager import ChromaManager

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.chroma_manager = ChromaManager()

    def test_collection_stats(self):
        stats = self.chroma_manager.get_collection_stats()
        self.assertIn('total_documents', stats)
        self.assertIsInstance(stats['total_documents'], int)
        self.assertGreaterEqual(stats['total_documents'], 0)

    def test_add_and_search_documents(self):
        # Add a test document
        chunks = ["This is a test document about healthcare."]
        embeddings = self.chroma_manager.embedding_function(chunks)
        metadatas = [{"title": "Test Document", "authors": "Test Author"}]

        add_result = self.chroma_manager.add_documents(chunks, embeddings, metadatas)
        self.assertTrue(add_result)

        # Search for the document
        results = self.chroma_manager.search_documents("healthcare", n_results=1)
        self.assertTrue(len(results) > 0)
        self.assertIn("document", results[0])
        self.assertIn("metadata", results[0])

if __name__ == "__main__":
    unittest.main()
