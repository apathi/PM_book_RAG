"""
Document processing pipeline for PM Knowledge Assistant
Handles PDF extraction, metadata generation, and chunk creation
"""
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber
from tqdm import tqdm
import logging
from datetime import datetime

from .schemas import ChunkMetadata, ProcessingResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process PM books from the book_chapters directory"""
    
    def __init__(self, base_path: str = "book_chapters"):
        self.base_path = Path(base_path)
        self.processed_chunks: List[ChunkMetadata] = []
        
    def extract_book_title(self, folder_name: str) -> str:
        """Extract clean book title from folder name"""
        # Remove _chapters suffix and clean up
        title = folder_name.replace("_chapters", "")
        title = title.replace("_", " ")
        title = title.replace("-", " ")
        return title.strip()
    
    def extract_chapter_info(self, file_path: Path) -> Tuple[str, Optional[str]]:
        """Extract chapter number/name from file path and content"""
        file_name = file_path.stem  # Remove .pdf extension
        
        # Try to extract chapter number and title from filename
        # Pattern: Chapter_XX-Title or similar variations
        patterns = [
            r'Chapter[_\s]+(\d+)[-_](.+)',
            r'Chapter[_\s]+(\d+)',
            r'Extra[_\s]+(\d+)[-_](.+)',
            r'(\d+)[._\s]+(.+)'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, file_name, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return f"Chapter {groups[0]}: {groups[1].replace('_', ' ')}", groups[0]
                else:
                    return f"Chapter {groups[0]}", groups[0]
        
        # If no pattern matches, clean up the filename
        clean_name = file_name.replace('_', ' ').replace('-', ' ')
        return clean_name, None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""
    
    def extract_author_from_content(self, content: str, book_title: str) -> Optional[str]:
        """Extract author name using production-ready patterns"""
        # Production-tested patterns focused on reliability
        patterns = [
            # Most reliable: Name + is + description (works for About the Author sections)
            r'(?:Dr\.\s+)?([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+is\s+(?:a|an|the)\s+(?!a\s+(?:book|chapter|section|introduction))',
            
            # After "About the Author" header
            r'About\s+the\s+Authors?\s*\n\s*(?:Dr\.\s+)?([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+is',
            
            # Name with academic credentials
            r'([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?),\s*(?:PhD|Ph\.D\.|Dr\.|MBA|MS|BS),?\s+is',
        ]
        
        # Search in first 1500 characters (About the Author sections are usually at the top)
        search_text = content[:1500] if len(content) > 1500 else content
        
        for pattern in patterns:
            matches = re.findall(pattern, search_text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                cleaned_author = self._clean_author_name(match)
                if cleaned_author:
                    return cleaned_author
        
        return None
    
    def get_folder_structure(self, file_path: Path, base_book_path: Path) -> str:
        """Get the folder structure relative to the book directory"""
        try:
            relative_path = file_path.relative_to(base_book_path)
            return str(relative_path.parent)
        except ValueError:
            return ""
    
    def process_chapter_file(self, file_path: Path, book_title: str, 
                           book_path: Path, author_name: Optional[str] = None) -> Optional[ChunkMetadata]:
        """Process a single chapter PDF file"""
        try:
            # Extract text content
            content = self.extract_text_from_pdf(file_path)
            if not content:
                logger.warning(f"No content extracted from {file_path}")
                return None
            
            # Extract chapter info
            chapter_name, chapter_num = self.extract_chapter_info(file_path)
            
            # Get section from parent folder
            parent_folder = file_path.parent.name
            section = None if parent_folder == book_path.name else parent_folder
            
            # Get folder structure
            folder_structure = self.get_folder_structure(file_path, book_path)
            
            # Create unique chunk ID
            chunk_id = f"{book_title}_{folder_structure}_{file_path.stem}".replace(" ", "_").replace("/", "_")
            
            # Count words
            word_count = len(content.split())
            
            # Create metadata
            chunk = ChunkMetadata(
                book_title=book_title,
                author_name=author_name,
                chapter=chapter_name,
                section=section,
                file_path=str(file_path),
                chunk_id=chunk_id,
                content=content,
                folder_structure=folder_structure,
                word_count=word_count
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _clean_author_name(self, raw_name: str) -> Optional[str]:
        """Clean and validate extracted author name with strict filtering"""
        if not raw_name:
            return None
            
        # Clean up the name
        cleaned = raw_name.strip()
        cleaned = re.sub(r'^(?:Dr\.\s+|Author\s+|Authors\s+)', '', cleaned)  # Remove prefixes
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.replace('\n', ' ').strip()
        
        # Strict validation
        name_parts = cleaned.split()
        
        # Must be 2-3 name parts
        if len(name_parts) < 2 or len(name_parts) > 3:
            return None
            
        # Each part must be a proper name (starts with capital, rest lowercase letters)
        for part in name_parts:
            if not (part[0].isupper() and part[1:].islower() and part.isalpha()):
                return None
        
        # Length checks
        if len(cleaned) < 5 or len(cleaned) > 40:
            return None
            
        # Reject common false positives
        false_positives = {
            'Introduction Teaching', 'Product Owner', 'Sir Alec', 'Product What', 
            'Design Metrics', 'Metrics There', 'Teaching Product', 'Owner Product',
            'Technical Product', 'Product Management', 'Management Product',
            'Business Product', 'Product Design', 'Product Strategy',
            'Desktop Application What'
        }
        
        if cleaned in false_positives:
            return None
            
        return cleaned
    
    def find_author_in_book(self, book_path: Path, book_title: str) -> Optional[str]:
        """Search for author name with improved priority-based approach"""
        # Priority file search terms (most reliable first)
        priority_files = [
            "about_the_author", "about_author", "about_the_authors", "authors",
            "contributors", "about_the_contributors",  # Often contains author info
            "preface", "foreword", "introduction"
            # NOTE: Excluding "reviewers" - reviewers are not authors!
        ]
        
        # Find priority files
        target_files = []
        all_pdfs = list(book_path.rglob("*.pdf"))
        
        # Sort by priority - About the Author files first
        for priority_term in priority_files:
            for pdf_file in all_pdfs:
                if (priority_term in pdf_file.name.lower() and 
                    pdf_file not in target_files and
                    "processing_report" not in pdf_file.name):
                    target_files.append(pdf_file)
        
        # Try top 2 most reliable files
        for pdf_file in target_files[:2]:
            try:
                # Extract only first page for About the Author files
                with pdfplumber.open(pdf_file) as pdf:
                    if pdf.pages:
                        content = pdf.pages[0].extract_text()
                        if content:
                            author = self.extract_author_from_content(content, book_title)
                            if author:
                                logger.info(f"Found author '{author}' in {pdf_file.name}")
                                return author
            except Exception as e:
                logger.warning(f"Error reading {pdf_file}: {str(e)}")
        
        # If not found in chapters, try the main book file as fallback
        books_dir = Path("books")
        if books_dir.exists():
            book_name_variants = [
                book_path.name.replace("_chapters", ""),
                book_path.name.replace("-", "_").replace("_chapters", ""),
                book_path.name.replace("_", "-").replace("-chapters", "")
            ]
            
            for book_variant in book_name_variants:
                book_file = books_dir / f"{book_variant}.pdf"
                if book_file.exists():
                    try:
                        with pdfplumber.open(book_file) as pdf:
                            # Check first 3 pages
                            content_parts = []
                            for i in range(min(3, len(pdf.pages))):
                                text = pdf.pages[i].extract_text()
                                if text:
                                    content_parts.append(text)
                            
                            content = "\n".join(content_parts)
                            author = self.extract_author_from_content(content, book_title)
                            if author:
                                logger.info(f"Found author '{author}' in main book file")
                                return author
                    except Exception as e:
                        logger.warning(f"Error reading main book {book_file}: {str(e)}")
        
        return None
    
    def process_book(self, book_folder: Path) -> ProcessingResult:
        """Process all chapters in a book folder"""
        book_title = self.extract_book_title(book_folder.name)
        logger.info(f"Processing book: {book_title}")
        
        # Find author name
        author_name = self.find_author_in_book(book_folder, book_title)
        if author_name:
            logger.info(f"Found author: {author_name}")
        
        chunks = []
        errors = []
        
        # Process all PDF files in the book folder
        pdf_files = list(book_folder.rglob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc=f"Processing {book_title}"):
            # Skip processing report files
            if "processing_report" in pdf_file.name:
                continue
                
            chunk = self.process_chapter_file(pdf_file, book_title, book_folder, author_name)
            if chunk:
                chunks.append(chunk)
            else:
                errors.append(f"Failed to process: {pdf_file}")
        
        # Store chunks
        self.processed_chunks.extend(chunks)
        
        result = ProcessingResult(
            book_title=book_title,
            author_name=author_name,
            total_chapters=len(pdf_files),
            total_chunks=len(chunks),
            errors=errors,
            metadata={
                "processed_at": datetime.now().isoformat(),
                "folder_path": str(book_folder)
            }
        )
        
        return result
    
    def process_all_books(self) -> List[ProcessingResult]:
        """Process all books in the book_chapters directory"""
        results = []
        
        # Find all book folders
        book_folders = [f for f in self.base_path.iterdir() if f.is_dir()]
        
        logger.info(f"Found {len(book_folders)} books to process")
        
        for book_folder in book_folders:
            try:
                result = self.process_book(book_folder)
                results.append(result)
                
                # Save intermediate results
                self.save_processed_chunks(f"chunks_{book_folder.name}.json")
                
            except Exception as e:
                logger.error(f"Error processing book {book_folder.name}: {str(e)}")
                results.append(ProcessingResult(
                    book_title=self.extract_book_title(book_folder.name),
                    author_name=None,
                    total_chapters=0,
                    total_chunks=0,
                    errors=[str(e)]
                ))
        
        return results
    
    def save_processed_chunks(self, filename: str = "chunks_metadata.json"):
        """Save processed chunks to JSON file"""
        output_path = Path("data") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        chunks_data = [chunk.model_dump() for chunk in self.processed_chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks_data)} chunks to {output_path}")
    
    def validate_chapter_names(self, book_folder: Path) -> Dict[str, str]:
        """Check if chapter names in files match actual content"""
        validation_results = {}
        
        pdf_files = list(book_folder.rglob("*.pdf"))[:5]  # Check first 5 files
        
        for pdf_file in pdf_files:
            content = self.extract_text_from_pdf(pdf_file)
            if content:
                # Look for chapter title in first 500 characters
                first_part = content[:500]
                file_chapter = self.extract_chapter_info(pdf_file)[0]
                
                # Try to find actual chapter title in content
                chapter_patterns = [
                    r'Chapter\s+\d+[:\s]+([^\n]+)',
                    r'CHAPTER\s+\d+[:\s]+([^\n]+)',
                    r'^\s*(\d+\.\s+[^\n]+)',
                ]
                
                actual_chapter = None
                for pattern in chapter_patterns:
                    match = re.search(pattern, first_part, re.MULTILINE)
                    if match:
                        actual_chapter = match.group(0).strip()
                        break
                
                validation_results[str(pdf_file)] = {
                    "file_derived": file_chapter,
                    "content_derived": actual_chapter or "Not found"
                }
        
        return validation_results


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    
    # Process a specific book for testing
    test_book = Path("book_chapters/cracking-the-pm-career_chapters")
    if test_book.exists():
        result = processor.process_book(test_book)
        print(f"Processed {result.book_title}:")
        print(f"  Author: {result.author_name}")
        print(f"  Chapters: {result.total_chapters}")
        print(f"  Chunks: {result.total_chunks}")
        
        # Validate chapter names
        validation = processor.validate_chapter_names(test_book)
        print("\nChapter name validation:")
        for file, names in validation.items():
            print(f"  {Path(file).name}:")
            print(f"    File: {names['file_derived']}")
            print(f"    Content: {names['content_derived']}")