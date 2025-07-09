#!/usr/bin/env python3
"""
Fix script to test and repair vector store issues.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.vector_store import ChromaManager
from src.qa_system import QAEngine

def test_and_fix_vector_store():
    """Test the vector store and fix common issues."""
    print("🔍 Testing and Fixing Vector Store Issues...")
    
    try:
        # Initialize managers
        chroma_manager = ChromaManager()
        qa_engine = QAEngine()
        
        print("✅ Managers initialized successfully")
        
        # Check collection stats
        print("\n📊 Checking Collection Statistics...")
        stats = chroma_manager.get_collection_stats()
        total_docs = stats.get('total_documents', 0)
        print(f"Total documents: {total_docs}")
        
        if total_docs == 0:
            print("❌ No documents in collection!")
            print("💡 Solution: Go to the Research & Ingest page and ingest some articles")
            return False
        
        # Test basic search without any filters
        print(f"\n🔍 Testing basic search on {total_docs} documents...")
        
        test_queries = ["diabetes", "fasting", "health", "weight"]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            try:
                # Test direct ChromaDB search (no filters)
                results = chroma_manager.search_documents(
                    query=query,
                    n_results=3
                )
                
                print(f"   ChromaDB search: {len(results)} results")
                
                if results:
                    # Check if metadata has healthcare_relevance field
                    first_result = results[0]
                    metadata = first_result.get('metadata', {})
                    has_relevance = 'healthcare_relevance' in metadata
                    
                    print(f"   Has healthcare_relevance field: {has_relevance}")
                    
                    if has_relevance:
                        relevance = metadata.get('healthcare_relevance', 0)
                        print(f"   Sample relevance score: {relevance}")
                    else:
                        print("   ⚠️  Documents missing healthcare_relevance field")
                        print("   💡 This explains why QA system finds no results")
                        print("   💡 Solution: Re-ingest documents to add missing metadata")
                
                # Test QA engine retrieval with very low threshold
                qa_results = qa_engine.retrieve_relevant_documents(
                    query=query,
                    n_results=3,
                    min_relevance_score=0.0  # No filtering
                )
                
                print(f"   QA engine search: {len(qa_results)} results")
                
                if qa_results:
                    print("   ✅ QA engine can retrieve documents!")
                    
                    # Test with higher threshold
                    qa_results_filtered = qa_engine.retrieve_relevant_documents(
                        query=query,
                        n_results=3,
                        min_relevance_score=0.3
                    )
                    print(f"   QA engine (relevance >= 0.3): {len(qa_results_filtered)} results")
                    
                else:
                    print("   ❌ QA engine cannot retrieve documents")
                
            except Exception as e:
                print(f"   ❌ Error during search: {e}")
        
        # Test a complete QA query
        print(f"\n🤖 Testing Complete Q&A Pipeline...")
        test_question = "What are the benefits of intermittent fasting?"
        
        try:
            response = qa_engine.ask_question(
                question=test_question,
                n_documents=3,
                min_relevance=0.0,  # No filtering
                include_sources=True
            )
            
            print(f"   Question: {test_question}")
            print(f"   Answer length: {len(response.get('answer', ''))} characters")
            print(f"   Sources found: {response.get('sources_count', 0)}")
            print(f"   Confidence: {response.get('confidence', 0):.2f}")
            
            if response.get('sources_count', 0) > 0:
                print("   ✅ Q&A system working!")
            else:
                print("   ❌ Q&A system not finding sources")
                
        except Exception as e:
            print(f"   ❌ Q&A pipeline error: {e}")
        
        # Provide recommendations
        print(f"\n💡 Recommendations:")
        print("1. Set minimum relevance to 0.0 in the Streamlit app")
        print("2. If documents are missing healthcare_relevance field, re-ingest them")
        print("3. Try broader search terms like 'health', 'medical', 'study'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_and_fix_vector_store()
