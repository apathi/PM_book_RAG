#!/usr/bin/env python3
"""
Check which books have been processed so far
"""
import json
from pathlib import Path
from collections import Counter

def check_processed_books():
    """Check which books are in the current data files"""

    # Only check the current production data file
    data_files = [
        ("chunks_metadata.json", "Current FAISS Index Data")
    ]

    print("=" * 60)
    print("PROCESSED BOOKS ANALYSIS")
    print("=" * 60)

    # Track processed books for later comparison
    processed_book_titles = set()

    for filename, description in data_files:
        file_path = Path("data") / filename
        
        if not file_path.exists():
            print(f"\n{description}: FILE NOT FOUND")
            continue
            
        print(f"\n{description}:")
        print("-" * len(description))
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Count books and chapters
            book_counts = Counter()
            authors = {}
            
            for chunk in data:
                book_title = chunk.get('book_title', 'Unknown')
                author = chunk.get('author_name', 'Unknown')
                
                book_counts[book_title] += 1
                authors[book_title] = author
            
            print(f"Total chunks: {len(data)}")
            print(f"Books processed: {len(book_counts)}")
            
            for book, count in book_counts.items():
                author = authors.get(book, 'Unknown')
                print(f"  • {book} - {count} chapters (Author: {author})")
                # Store normalized book title for matching
                processed_book_titles.add(book.lower().strip())

        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
    
    # Check what's available to process
    print(f"\n{'AVAILABLE BOOKS TO PROCESS:'}")
    print("-" * 30)
    
    book_chapters_dir = Path("book_chapters")
    if book_chapters_dir.exists():
        available_books = [f for f in book_chapters_dir.iterdir() if f.is_dir()]
        print(f"Total available: {len(available_books)} books")
        
        for book_folder in sorted(available_books, key=lambda x: x.name):
            pdf_count = len(list(book_folder.rglob("*.pdf")))
            # Remove processing report files from count
            actual_pdfs = [f for f in book_folder.rglob("*.pdf") if "processing_report" not in f.name]
            actual_count = len(actual_pdfs)

            # Normalize folder name for matching
            folder_normalized = book_folder.name.replace("_chapters", "").replace("-", " ").replace("_", " ").lower().strip()

            # Check if this book has been processed
            is_processed = any(folder_normalized in processed_title or processed_title in folder_normalized
                             for processed_title in processed_book_titles)

            status = "✅ PROCESSED" if is_processed else "⏳ PENDING"

            print(f"  {status} - {book_folder.name} ({actual_count} chapters)")
    
    print(f"\n{'SUMMARY:'}")
    print("-" * 10)
    if Path("data/chunks_metadata.json").exists():
        print("✅ System ready with processed data")
        print("🌐 Web interface can be launched with: python app.py")
    else:
        print("❌ No processed data found")
        print("📚 Run: python process_all_books.py (all books) or python tests/test_one_book.py (single book)")

if __name__ == "__main__":
    check_processed_books()