#!/usr/bin/env python3
"""Validate chapter names for cracking-the-pm-career"""
import pdfplumber
from pathlib import Path
import re

# Check a few chapters from cracking-the-pm-career
chapters_to_check = [
    "book_chapters/cracking-the-pm-career_chapters/B._The_Product_Manager_Role/Chapter_1-Getting_Started.pdf",
    "book_chapters/cracking-the-pm-career_chapters/C._Product_Skills/Chapter_4-User_Insight.pdf",
    "book_chapters/cracking-the-pm-career_chapters/D._Execution_Skills/Chapter_10-Project_Management.pdf"
]

for chapter_path in chapters_to_check:
    chapter_file = Path(chapter_path)
    if chapter_file.exists():
        print(f"\n{'='*60}")
        print(f"File: {chapter_file.name}")
        
        # Extract from filename
        file_name = chapter_file.stem
        patterns = [
            r'Chapter[_\s]+(\d+)[-_](.+)',
            r'Chapter[_\s]+(\d+)',
        ]
        
        file_chapter = None
        for pattern in patterns:
            match = re.match(pattern, file_name, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    file_chapter = f"Chapter {groups[0]}: {groups[1].replace('_', ' ')}"
                else:
                    file_chapter = f"Chapter {groups[0]}"
                break
        
        print(f"From filename: {file_chapter}")
        
        # Check actual content
        with pdfplumber.open(chapter_file) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text()
                if text:
                    # Look for chapter title in first 500 characters
                    first_part = text[:500]
                    print(f"\nFirst 300 chars of content:")
                    print(first_part[:300])
                    print(f"\nChapter name extraction correct? Please verify.")