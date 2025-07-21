#!/usr/bin/env python3
"""
Test the RAG pipeline with some queries
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import PMKnowledgeRAG

def test_queries():
    """Test the RAG pipeline with sample questions"""
    print("Testing RAG Pipeline")
    print("=" * 60)
    
    try:
        # Initialize RAG
        rag = PMKnowledgeRAG()
        
        # Test questions
        test_questions = [
            "What are the key skills needed for a product manager?",
            "How should product managers prioritize features?",
            "What is the product manager role?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 40)
            
            result = rag.generate_answer(question, k=3)
            
            print(f"\nAnswer:\n{result['answer'][:300]}...")
            
            print(f"\nSources ({len(result['sources'])}):")
            for source in result['sources'][:2]:  # Show first 2 sources
                print(f"  • {source['book_title']} - {source['chapter']}")
                print(f"    by {source['author']} (relevance: {source['similarity_score']:.1%})")
        
        # Test comparison
        print(f"\n" + "=" * 60)
        print("Testing Perspective Comparison")
        print("=" * 60)
        
        topic = "product management skills"
        comparison = rag.compare_perspectives(topic, k=5)
        
        print(f"\nTopic: {topic}")
        print(f"Authors analyzed: {len(comparison['authors'])}")
        print(f"\nComparison:\n{comparison['comparison'][:400]}...")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_queries()