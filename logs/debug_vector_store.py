#!/usr/bin/env python3
"""
Debug script to check vector store and document retrieval.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.vector_store import ChromaManager
from src.qa_system import QAEngine

def debug_vector_store():
    """Debug the vector store and document retrieval."""
    print("🔍 Debugging Vector Store and Document Retrieval...")
    
    try:
        # Initialize managers
        chroma_manager = ChromaManager()
        qa_engine = QAEngine()
        
        print("✅ Managers initialized successfully")
        
        # Check collection stats
        print("\n📊 Checking Collection Statistics...")
        stats = chroma_manager.get_collection_stats()
        print(f"Total documents: {stats.get('total_documents', 0)}")
        print(f"Collection name: {stats.get('collection_name', 'Unknown')}")
        
        if stats.get('total_documents', 0) == 0:
            print("❌ No documents in collection! This explains why no results are found.")
            return False
        
        # Test basic document retrieval with very low threshold
        print("\n🔍 Testing Document Retrieval...")
        test_queries = [
            "diabetes",
            "intermittent fasting", 
            "obesity",
            "weight loss",
            "health"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            # Try with very low relevance threshold
            try:
                results = chroma_manager.search_documents(
                    query=query,
                    n_results=5
                )
                print(f"   Found {len(results)} results")
                
                if results:
                    for i, result in enumerate(results[:2]):
                        metadata = result.get('metadata', {})
                        title = metadata.get('title', 'No title')[:100]
                        print(f"   {i+1}. {title}...")
                else:
                    print("   ❌ No results found")
                    
            except Exception as e:
                print(f"   ❌ Error during search: {e}")
        
        # Test QA engine retrieval with different thresholds
        print("\n🤖 Testing QA Engine Retrieval...")
        test_question = "What are the benefits of intermittent fasting?"
        
        for min_relevance in [0.0, 0.1, 0.3, 0.5]:
            print(f"\n🔍 Testing with min_relevance={min_relevance}")
            try:
                docs = qa_engine.retrieve_relevant_documents(
                    query=test_question,
                    n_results=5,
                    min_relevance_score=min_relevance
                )
                print(f"   Found {len(docs)} documents")
                
                if docs:
                    for i, doc in enumerate(docs[:2]):
                        metadata = doc.get('metadata', {})
                        title = metadata.get('title', 'No title')[:80]
                        relevance = metadata.get('healthcare_relevance', 0)
                        print(f"   {i+1}. {title}... (relevance: {relevance})")
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_vector_store()
