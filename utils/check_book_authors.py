#!/usr/bin/env python3
"""Check author names from book title pages"""
import pdfplumber
from pathlib import Path

# Check first few pages of each book to find authors
books_to_check = [
    "books/cracking-the-pm-career.pdf",
    "books/the-pm-interview.pdf", 
    "books/decode-and-conquer.epub"
]

for book_path in books_to_check:
    book_file = Path(book_path)
    if book_file.exists() and book_file.suffix == '.pdf':
        print(f"\n{'='*60}")
        print(f"Checking: {book_file.name}")
        print('='*60)
        
        with pdfplumber.open(book_file) as pdf:
            # Check pages 0-6 (title, copyright, etc)
            for i in range(min(7, len(pdf.pages))):
                text = pdf.pages[i].extract_text()
                if text:
                    # Look for lines with author-like names
                    lines = text.split('\n')
                    for line in lines[:20]:  # First 20 lines of each page
                        # Check for lines that might contain author names
                        if any(word in line.lower() for word in ['by ', 'author', 'written']):
                            print(f"Page {i+1}: {line.strip()}")
                        # Also check for standalone names (all caps or title case)
                        elif line.strip() and len(line.split()) in [2, 3, 4]:
                            words = line.strip().split()
                            if all(word[0].isupper() for word in words if len(word) > 1):
                                # Likely a name
                                if not any(skip in line.lower() for skip in ['chapter', 'table', 'contents', 'copyright', 'isbn']):
                                    print(f"Page {i+1} (possible name): {line.strip()}")