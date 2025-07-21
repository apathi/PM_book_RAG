"""
Source Manager for UI - Clean book/chapter discovery and formatting
Integrates seamlessly with our robust StateManager
"""
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from ..state_manager import StateManager, ProcessingStatus

logger = logging.getLogger(__name__)


@dataclass
class ChapterInfo:
    """Information about a single chapter"""
    name: str
    status: str
    file_path: Optional[str] = None


@dataclass
class BookInfo:
    """Comprehensive information about a book"""
    name: str
    display_name: str
    document_status: str
    embedding_status: str
    total_chapters: int
    successful_chapters: int
    failed_chapters: int
    chapters: List[ChapterInfo]
    status_emoji: str
    status_text: str
    is_available: bool
    author_name: Optional[str] = None


class SourceManager:
    """
    Clean, elegant manager for book/chapter discovery and formatting
    
    Features:
    - Integrates with robust StateManager
    - Smart status formatting and display
    - Chapter-level granularity
    - Clean API for UI interactions
    """
    
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
    
    def _extract_author_from_chunks(self, book_name: str) -> Optional[str]:
        """
        Extract author name from processed chunks for this book
        
        Args:
            book_name: Name of the book to get author for
            
        Returns:
            Author name(s) or None if not found
        """
        try:
            book_files = self.state_manager.get_book_data_files(book_name)
            chunks_file = book_files["chunks"]
            
            if not chunks_file.exists():
                return None
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Look for author_name in chunks
            for chunk_data in chunks_data:
                author_name = chunk_data.get('author_name')
                if author_name and author_name.strip():
                    # Clean up the author name (remove newlines, etc.)
                    cleaned_author = author_name.replace('\n', ' ').strip()
                    
                    # Filter out obviously bad extractions
                    bad_patterns = ['good', 'best', 'new', 'the', 'this', 'that', 'copyright', 'all rights']
                    if (len(cleaned_author) > 2 and 
                        not any(bad_word in cleaned_author.lower() for bad_word in bad_patterns) and
                        not cleaned_author.isdigit()):
                        return cleaned_author
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not extract author for {book_name}: {e}")
            return None
    
    def get_available_books(self) -> List[BookInfo]:
        """
        Get all available books with comprehensive status information
        
        Returns:
            List of BookInfo objects with complete status details
        """
        books = []
        
        try:
            for book_name, book_status in self.state_manager.state.books.items():
                book_info = self._create_book_info(book_name, book_status)
                books.append(book_info)
            
            # Sort by completion status (completed first) then by name
            books.sort(key=lambda x: (
                0 if x.document_status == "completed" else 1,
                x.display_name.lower()
            ))
            
            return books
            
        except Exception as e:
            logger.error(f"Error getting available books: {e}")
            return []
    
    def _create_book_info(self, book_name: str, book_status) -> BookInfo:
        """Create comprehensive BookInfo from state data"""
        
        # Clean display name
        display_name = (book_name
                       .replace("_chapters", "")
                       .replace("-", " ")
                       .replace("_", " ")
                       .title())
        
        # Determine status indicators
        status_emoji, status_text, is_available = self._get_status_indicators(
            book_status.document_status, 
            book_status.embedding_status
        )
        
        # Get chapter details
        chapters = self._get_chapter_details(book_name, book_status)
        
        # Extract author information dynamically
        author_name = self._extract_author_from_chunks(book_name)
        
        return BookInfo(
            name=book_name,
            display_name=display_name,
            document_status=book_status.document_status.value,
            embedding_status=book_status.embedding_status.value,
            total_chapters=book_status.total_chapters,
            successful_chapters=book_status.successful_chapters,
            failed_chapters=book_status.failed_chapters,
            chapters=chapters,
            status_emoji=status_emoji,
            status_text=status_text,
            is_available=is_available,
            author_name=author_name
        )
    
    def _get_status_indicators(self, doc_status: ProcessingStatus, embed_status: ProcessingStatus) -> Tuple[str, str, bool]:
        """Get emoji, text, and availability for a book's status"""
        
        if doc_status == ProcessingStatus.COMPLETED and embed_status == ProcessingStatus.COMPLETED:
            return "✅", "Complete", True
        elif doc_status == ProcessingStatus.COMPLETED and embed_status == ProcessingStatus.IN_PROGRESS:
            return "🔄", "Generating Embeddings", True
        elif doc_status == ProcessingStatus.COMPLETED:
            return "📄", "Documents Ready", True
        elif doc_status == ProcessingStatus.IN_PROGRESS:
            return "🔄", "Processing", False
        elif doc_status == ProcessingStatus.FAILED:
            return "❌", "Failed", False
        else:
            return "⏸️", "Not Started", False
    
    def _get_chapter_details(self, book_name: str, book_status) -> List[ChapterInfo]:
        """Get detailed chapter information if available"""
        chapters = []
        
        if book_status.document_status != ProcessingStatus.COMPLETED:
            return chapters
        
        try:
            book_files = self.state_manager.get_book_data_files(book_name)
            chunks_file = book_files["chunks"]
            
            if not chunks_file.exists():
                return chapters
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Extract unique chapters with their status
            chapters_seen = set()
            for chunk_data in chunks_data:
                chapter_name = chunk_data.get('chapter', 'Unknown Chapter')
                if chapter_name not in chapters_seen:
                    chapters.append(ChapterInfo(
                        name=chapter_name,
                        status="completed"
                    ))
                    chapters_seen.add(chapter_name)
            
            # Sort chapters naturally
            chapters.sort(key=lambda x: x.name)
            
        except Exception as e:
            logger.warning(f"Could not load chapter details for {book_name}: {e}")
        
        return chapters
    
    def format_book_selection_choices(self, books: List[BookInfo]) -> List[Tuple[str, str]]:
        """
        Format books for Gradio CheckboxGroup choices
        
        Returns:
            List of (display_label, book_name) tuples
        """
        choices = []
        
        for book in books:
            if book.is_available:
                # Format: "✅ Cracking The Pm Career (57 chapters) Complete - Author Name"
                label = (f"{book.status_emoji} {book.display_name} "
                        f"({book.successful_chapters} chapters) {book.status_text}")
                
                # Add author name if available
                if book.author_name:
                    label += f" - {book.author_name}"
                
                choices.append((label, book.name))
        
        return choices
    
    def get_default_selections(self, books: List[BookInfo]) -> List[str]:
        """Get smart default selections (completed books)"""
        return [
            book.name for book in books 
            if book.is_available and book.document_status == "completed"
        ]
    
    def format_selection_summary(self, books: List[BookInfo], selected_books: List[str]) -> str:
        """Create a summary of current selection"""
        if not selected_books:
            return "📚 No books selected"
        
        total_chapters = 0
        selected_count = len(selected_books)
        
        for book in books:
            if book.name in selected_books:
                total_chapters += book.successful_chapters
        
        return f"📚 Selected: {total_chapters} chapters from {selected_count} books"
    
    def format_status_display(self, books: List[BookInfo]) -> str:
        """Format comprehensive status display"""
        if not books:
            return "No processed books available. Please run processing first."
        
        lines = []
        total_available_chapters = 0
        
        for book in books:
            # Status line
            chapters_info = f"({book.successful_chapters}"
            if book.total_chapters > book.successful_chapters:
                chapters_info += f"/{book.total_chapters}"
            chapters_info += " chapters)"
            
            line = f"{book.status_emoji} **{book.display_name}** {chapters_info} *{book.status_text}*"
            
            # Add author name if available
            if book.author_name:
                line += f" - {book.author_name}"
            
            lines.append(line)
            
            if book.is_available:
                total_available_chapters += book.successful_chapters
        
        lines.extend([
            "",
            f"**Total Available:** {total_available_chapters} chapters across {len([b for b in books if b.is_available])} books"
        ])
        
        return "\n\n".join(lines)
    
    def get_book_by_name(self, books: List[BookInfo], book_name: str) -> Optional[BookInfo]:
        """Get a specific book by name"""
        for book in books:
            if book.name == book_name:
                return book
        return None
    
    def filter_available_books(self, books: List[BookInfo]) -> List[BookInfo]:
        """Filter to only available books"""
        return [book for book in books if book.is_available]