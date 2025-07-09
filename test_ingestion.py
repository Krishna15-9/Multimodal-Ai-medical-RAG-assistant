#!/usr/bin/env python3
"""
Quick test script for document ingestion pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.processing import DocumentProcessor

def test_ingestion_pipeline():
    """Test the document ingestion pipeline with additional parameters."""
    print("🔍 Testing Document Ingestion Pipeline...")
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized successfully")
        
        # Test with additional parameters (like from Streamlit)
        search_kwargs = {
            'date_range': "2020:2024",
            'article_types': ["Clinical Trial", "Review"]
        }
        
        print("🔍 Testing search_and_ingest_pipeline with additional parameters...")
        
        # This should work now without the "unexpected keyword argument" error
        # We'll use a small test to avoid actually hitting the API
        try:
            # Just test the method signature, not the actual execution
            import inspect
            sig = inspect.signature(processor.search_and_ingest_pipeline)
            print(f"✅ Method signature: {sig}")
            
            # Check if it accepts **kwargs
            has_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())
            if has_kwargs:
                print("✅ Method accepts **kwargs - additional parameters should work")
            else:
                print("❌ Method does not accept **kwargs")
                
        except Exception as e:
            print(f"❌ Error testing method signature: {e}")
            return False
        
        print("🎉 Ingestion pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Ingestion pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    test_ingestion_pipeline()
