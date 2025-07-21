#!/usr/bin/env python3
"""
Test script to verify the PM Knowledge Assistant is working
"""
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_document_processing():
    """Test if documents were processed correctly"""
    print("\n1. Testing Document Processing...")

    chunks_file = Path("data/chunks_metadata.json")
    if not chunks_file.exists():
        print("   ❌ No processed chunks found")
        print("   Run: python process_all_books.py")
        return False
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"   ✅ Found {len(chunks)} processed chunks")
    
    # Check a sample chunk
    if chunks:
        sample = chunks[0]
        print(f"   Sample chunk: {sample['book_title']} - {sample['chapter']}")
        print(f"   Author: {sample.get('author_name', 'Not found')}")
    
    return True

def test_embeddings():
    """Test if embeddings were generated"""
    print("\n2. Testing Embeddings...")
    
    index_file = Path("data/faiss_index.bin")
    metadata_file = Path("data/chunks_metadata.json")
    
    if not index_file.exists():
        print("   ❌ No FAISS index found")
        print("   Run: python generate_embeddings.py")
        return False
    
    if not metadata_file.exists():
        print("   ❌ No metadata found")
        return False
    
    print(f"   ✅ FAISS index exists: {index_file}")
    print(f"   ✅ Metadata exists: {metadata_file}")
    
    return True

def test_api_key():
    """Test if API key is configured"""
    print("\n3. Testing API Configuration...")
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        print("   ❌ OpenAI API key not configured")
        print("   Edit .env file and add your API key")
        return False
    
    print("   ✅ OpenAI API key configured")
    return True

def test_rag_pipeline():
    """Test if RAG pipeline works"""
    print("\n4. Testing RAG Pipeline...")
    
    try:
        from src.rag_pipeline import PMKnowledgeRAG
        
        # Initialize RAG
        rag = PMKnowledgeRAG()
        print("   ✅ RAG pipeline initialized")
        
        # Test a simple query
        test_query = "What is product management?"
        chunks = rag.retrieve_relevant_chunks(test_query, k=3)
        
        if chunks:
            print(f"   ✅ Retrieved {len(chunks)} relevant chunks")
            print(f"   Top result: {chunks[0]['book_title']} - {chunks[0]['chapter']}")
        else:
            print("   ❌ No chunks retrieved")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ RAG pipeline error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PM Knowledge Assistant - System Test")
    print("=" * 60)
    
    tests = [
        test_document_processing,
        test_api_key,
        test_embeddings,
        test_rag_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"   ❌ Test failed with error: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    if all(results):
        print("✅ All tests passed! System is ready to use.")
        print("\nRun 'python app.py' to launch the web interface.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        failed_count = len([r for r in results if not r])
        print(f"\nPassed: {len(results) - failed_count}/{len(results)} tests")

if __name__ == "__main__":
    main()