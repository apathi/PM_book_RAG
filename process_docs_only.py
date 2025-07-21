#!/usr/bin/env python3
"""
Process documents only (no embeddings) for safer testing
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.document_processor import DocumentProcessor

def main():
    print("Processing documents only (no embeddings)...")
    print("=" * 60)
    
    # Process just a few books first
    processor = DocumentProcessor()
    
    # Get list of all book folders
    book_folders = [f for f in processor.base_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(book_folders)} books:")
    for i, folder in enumerate(book_folders, 1):
        print(f"{i}. {folder.name}")
    
    # Ask user which books to process
    response = input(f"\nProcess all {len(book_folders)} books? (y/n/number for just one book): ").strip()
    
    if response.lower() == 'n':
        print("Cancelled.")
        return
    elif response.isdigit():
        book_num = int(response) - 1
        if 0 <= book_num < len(book_folders):
            print(f"\nProcessing only: {book_folders[book_num].name}")
            result = processor.process_book(book_folders[book_num])
            
            print(f"\nResults:")
            print(f"Book: {result.book_title}")
            print(f"Author: {result.author_name}")
            print(f"Chapters: {result.total_chapters}")
            print(f"Chunks: {result.total_chunks}")
            print(f"Errors: {len(result.errors)}")
            
            # Save processed chunks
            processor.save_processed_chunks("test_chunks.json")
            print(f"\nSaved to data/test_chunks.json")
        else:
            print("Invalid book number")
            return
    else:
        # Process all books
        print(f"\nProcessing all {len(book_folders)} books...")
        results = processor.process_all_books()
        
        total_chunks = sum(r.total_chunks for r in results)
        total_errors = sum(len(r.errors) for r in results)
        
        print(f"\nSummary:")
        print(f"Total books: {len(results)}")
        print(f"Total chunks: {total_chunks}")
        print(f"Total errors: {total_errors}")
        
        # Save all processed chunks
        processor.save_processed_chunks("chunks_metadata.json")
        print(f"\nSaved to data/chunks_metadata.json")

if __name__ == "__main__":
    main()