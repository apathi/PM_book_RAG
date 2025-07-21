#!/usr/bin/env python3
"""
Generate embeddings for already processed chunks
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.embeddings_generator import process_all_books

def main():
    print("=" * 60)
    print("PM Knowledge Assistant - Embeddings Generation")
    print("=" * 60)
    
    chunks_file = "data/chunks_metadata.json"

    if not Path(chunks_file).exists():
        print(f"❌ Error: Chunks file not found at {chunks_file}")
        print("Please run 'python process_all_books.py' first to process the documents.")
        return

    print(f"\n📚 Loading chunks from {chunks_file}...")
    
    try:
        results = process_all_books(chunks_file)
        
        print("\n✅ Embeddings generation complete!")
        print(f"Successfully processed: {results['successful']} chunks")
        print(f"Failed: {results['failed']} chunks")
        
        print("\n📁 Generated files:")
        print("- data/faiss_index.bin (vector index)")
        print("- data/chunks_metadata.json (chunk metadata)")
        print("- data/embeddings.pkl (embeddings backup)")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Set your OPENAI_API_KEY in the .env file")
        print("2. Have a stable internet connection")
        print("3. Have sufficient API credits")


if __name__ == "__main__":
    main()