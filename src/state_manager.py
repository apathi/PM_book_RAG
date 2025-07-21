"""
Robust state management for PM Knowledge Assistant
Tracks processing state with atomic operations and recovery capabilities
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status of processing for documents and embeddings"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BookStatus:
    """Status of a single book's processing"""
    book_name: str
    document_status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    embedding_status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    last_document_update: Optional[str] = None
    last_embedding_update: Optional[str] = None
    total_chapters: int = 0
    successful_chapters: int = 0
    failed_chapters: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


@dataclass 
class UnifiedIndexStatus:
    """Status of the unified index containing all books"""
    last_built: Optional[str] = None
    books_included: List[str] = None
    needs_rebuild: bool = True
    total_chunks: int = 0
    
    def __post_init__(self):
        if self.books_included is None:
            self.books_included = []


@dataclass
class ProcessingState:
    """Complete state of the processing system"""
    version: str = "1.0"
    books: Dict[str, BookStatus] = None
    unified_index: UnifiedIndexStatus = None
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.books is None:
            self.books = {}
        if self.unified_index is None:
            self.unified_index = UnifiedIndexStatus()


class StateManager:
    """
    Centralized state management for robust processing
    
    Features:
    - Atomic state updates
    - Backward compatibility with existing data
    - Recovery from any interruption point
    - Consistency validation
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.state_dir = self.data_dir / "state"
        self.books_dir = self.state_dir / "books"
        
        # State file paths
        self.state_file = self.state_dir / "processing_state.json"
        self.backup_state_file = self.state_dir / "processing_state_backup.json"
        
        # Ensure directories exist
        self._create_directories()
        
        # Load or initialize state
        self.state = self._load_state()
    
    def _create_directories(self):
        """Create necessary directory structure"""
        for dir_path in [self.data_dir, self.state_dir, self.books_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()
    
    def _load_state(self) -> ProcessingState:
        """Load processing state from disk with fallback to backup"""
        # Try main state file first
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._deserialize_state(data)
            except Exception as e:
                logger.warning(f"Failed to load main state file: {e}")
        
        # Try backup state file
        if self.backup_state_file.exists():
            try:
                with open(self.backup_state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info("Loaded state from backup file")
                return self._deserialize_state(data)
            except Exception as e:
                logger.warning(f"Failed to load backup state file: {e}")
        
        # Initialize new state
        logger.info("Initializing new processing state")
        return ProcessingState()
    
    def _deserialize_state(self, data: Dict) -> ProcessingState:
        """Convert JSON data back to ProcessingState object"""
        # Handle books
        books = {}
        for book_name, book_data in data.get('books', {}).items():
            # Convert status strings back to enums
            book_data['document_status'] = ProcessingStatus(book_data.get('document_status', 'not_started'))
            book_data['embedding_status'] = ProcessingStatus(book_data.get('embedding_status', 'not_started'))
            books[book_name] = BookStatus(**book_data)
        
        # Handle unified index
        unified_data = data.get('unified_index', {})
        unified_index = UnifiedIndexStatus(**unified_data)
        
        return ProcessingState(
            version=data.get('version', '1.0'),
            books=books,
            unified_index=unified_index,
            last_updated=data.get('last_updated')
        )
    
    def _serialize_state(self, state: ProcessingState) -> Dict:
        """Convert ProcessingState to JSON-serializable dict"""
        data = asdict(state)
        
        # Convert enums to strings
        for book_name, book_status in data['books'].items():
            book_status['document_status'] = book_status['document_status'].value
            book_status['embedding_status'] = book_status['embedding_status'].value
        
        return data
    
    def _save_state(self):
        """Atomically save state to disk with backup"""
        self.state.last_updated = self._get_timestamp()
        
        # Create backup of current state if it exists
        if self.state_file.exists():
            try:
                self.state_file.rename(self.backup_state_file)
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Save new state
        try:
            data = self._serialize_state(self.state)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            # Restore backup if save failed
            if self.backup_state_file.exists():
                self.backup_state_file.rename(self.state_file)
            raise
    
    def get_book_status(self, book_name: str) -> BookStatus:
        """Get status of a specific book"""
        if book_name not in self.state.books:
            self.state.books[book_name] = BookStatus(book_name=book_name)
        return self.state.books[book_name]
    
    def update_book_status(self, book_name: str, **updates):
        """Update book status with given fields"""
        book_status = self.get_book_status(book_name)
        
        # Update fields
        for key, value in updates.items():
            if hasattr(book_status, key):
                setattr(book_status, key, value)
        
        # Update timestamps
        if 'document_status' in updates:
            book_status.last_document_update = self._get_timestamp()
        if 'embedding_status' in updates:
            book_status.last_embedding_update = self._get_timestamp()
        
        # Save state
        self._save_state()
    
    def mark_book_document_started(self, book_name: str):
        """Mark book document processing as started"""
        self.update_book_status(book_name, document_status=ProcessingStatus.IN_PROGRESS)
    
    def mark_book_document_completed(self, book_name: str, total_chapters: int, successful: int, failed: int = 0):
        """Mark book document processing as completed"""
        self.update_book_status(
            book_name,
            document_status=ProcessingStatus.COMPLETED,
            total_chapters=total_chapters,
            successful_chapters=successful,
            failed_chapters=failed
        )
    
    def mark_book_document_failed(self, book_name: str, error_message: str):
        """Mark book document processing as failed"""
        book_status = self.get_book_status(book_name)
        book_status.error_messages.append(f"Document processing: {error_message}")
        self.update_book_status(book_name, document_status=ProcessingStatus.FAILED)
    
    def mark_book_embedding_started(self, book_name: str):
        """Mark book embedding generation as started"""
        self.update_book_status(book_name, embedding_status=ProcessingStatus.IN_PROGRESS)
    
    def mark_book_embedding_completed(self, book_name: str, successful: int, failed: int = 0):
        """Mark book embedding generation as completed"""
        self.update_book_status(
            book_name,
            embedding_status=ProcessingStatus.COMPLETED,
            successful_embeddings=successful,
            failed_embeddings=failed
        )
    
    def mark_book_embedding_failed(self, book_name: str, error_message: str):
        """Mark book embedding generation as failed"""
        book_status = self.get_book_status(book_name)
        book_status.error_messages.append(f"Embedding generation: {error_message}")
        self.update_book_status(book_name, embedding_status=ProcessingStatus.FAILED)
    
    def get_books_needing_document_processing(self) -> List[str]:
        """Get list of books that need document processing"""
        books = []
        
        # Check available books in book_chapters directory
        book_chapters_dir = Path("book_chapters")
        if book_chapters_dir.exists():
            for book_folder in book_chapters_dir.iterdir():
                if book_folder.is_dir():
                    book_name = book_folder.name
                    status = self.get_book_status(book_name)
                    
                    if status.document_status in [ProcessingStatus.NOT_STARTED, ProcessingStatus.FAILED]:
                        books.append(book_name)
        
        return books
    
    def get_books_needing_embedding_processing(self) -> List[str]:
        """Get list of books that need embedding processing"""
        books = []
        
        for book_name, book_status in self.state.books.items():
            if (book_status.document_status == ProcessingStatus.COMPLETED and 
                book_status.embedding_status in [ProcessingStatus.NOT_STARTED, ProcessingStatus.FAILED]):
                books.append(book_name)
        
        return books
    
    def get_completed_books(self) -> List[str]:
        """Get list of books that are fully processed (documents and embeddings)"""
        books = []
        
        for book_name, book_status in self.state.books.items():
            if (book_status.document_status == ProcessingStatus.COMPLETED and 
                book_status.embedding_status == ProcessingStatus.COMPLETED):
                books.append(book_name)
        
        return books
    
    def needs_unified_index_rebuild(self) -> bool:
        """Check if unified index needs to be rebuilt"""
        if self.state.unified_index.needs_rebuild:
            return True
        
        # Check if any completed books are not in the unified index
        completed_books = set(self.get_completed_books())
        included_books = set(self.state.unified_index.books_included)
        
        return completed_books != included_books
    
    def mark_unified_index_built(self, books_included: List[str], total_chunks: int):
        """Mark unified index as built"""
        self.state.unified_index.last_built = self._get_timestamp()
        self.state.unified_index.books_included = books_included.copy()
        self.state.unified_index.needs_rebuild = False
        self.state.unified_index.total_chunks = total_chunks
        self._save_state()
    
    def get_processing_summary(self) -> Dict:
        """Get summary of current processing state"""
        summary = {
            "total_books_available": len(list(Path("book_chapters").iterdir())) if Path("book_chapters").exists() else 0,
            "books_document_completed": len([b for b in self.state.books.values() if b.document_status == ProcessingStatus.COMPLETED]),
            "books_embedding_completed": len([b for b in self.state.books.values() if b.embedding_status == ProcessingStatus.COMPLETED]),
            "books_fully_completed": len(self.get_completed_books()),
            "books_needing_document_processing": len(self.get_books_needing_document_processing()),
            "books_needing_embedding_processing": len(self.get_books_needing_embedding_processing()),
            "unified_index_needs_rebuild": self.needs_unified_index_rebuild(),
            "unified_index_last_built": self.state.unified_index.last_built,
            "unified_index_total_chunks": self.state.unified_index.total_chunks
        }
        
        return summary
    
    def validate_consistency(self) -> Dict:
        """Validate consistency between state and actual data files"""
        issues = []
        
        # Check if stated completed books have their data files
        for book_name, book_status in self.state.books.items():
            if book_status.document_status == ProcessingStatus.COMPLETED:
                book_chunks_file = self.books_dir / f"{book_name}_chunks.json"
                if not book_chunks_file.exists():
                    issues.append(f"Book {book_name} marked as document-completed but chunks file missing")
            
            if book_status.embedding_status == ProcessingStatus.COMPLETED:
                book_embeddings_file = self.books_dir / f"{book_name}_embeddings.pkl"
                if not book_embeddings_file.exists():
                    issues.append(f"Book {book_name} marked as embedding-completed but embeddings file missing")
        
        # Check production data files consistency
        if not self.state.unified_index.needs_rebuild:
            production_chunks_file = self.data_dir / "chunks_metadata.json"
            production_faiss_file = self.data_dir / "faiss_index.bin"

            if not production_chunks_file.exists():
                issues.append("Production index marked as current but chunks_metadata.json missing")
            if not production_faiss_file.exists():
                issues.append("Production index marked as current but faiss_index.bin missing")
        
        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "checked_at": self._get_timestamp()
        }
    
    def reset_book_status(self, book_name: str):
        """Reset a book's status (for recovery scenarios)"""
        if book_name in self.state.books:
            del self.state.books[book_name]
            self._save_state()
    
    def get_book_data_files(self, book_name: str) -> Dict[str, Path]:
        """Get paths to all data files for a specific book"""
        return {
            "chunks": self.books_dir / f"{book_name}_chunks.json",
            "embeddings": self.books_dir / f"{book_name}_embeddings.pkl", 
            "faiss": self.books_dir / f"{book_name}_faiss.bin"
        }


# State rebuilding functions
def rebuild_state_from_data(state_manager: StateManager):
    """
    Rebuild state tracking from production data files
    Syncs processing_state.json with actual chunks_metadata.json and faiss_index.bin
    """
    logger.info("Checking for production data files...")

    # Check if production files exist
    chunks_file = Path("data/chunks_metadata.json")
    faiss_file = Path("data/faiss_index.bin")

    if chunks_file.exists() and faiss_file.exists():
        logger.info("Found production data files, rebuilding state...")

        # Load chunks to understand what books were processed
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            # Group chunks by book to understand what was processed
            books_found = {}
            for chunk in chunks_data:
                book_title = chunk.get('book_title', 'unknown')
                if book_title not in books_found:
                    books_found[book_title] = []
                books_found[book_title].append(chunk)

            # Update state to reflect that these books are completed
            for book_title, book_chunks in books_found.items():
                # Try to find the corresponding folder name
                book_chapters_dir = Path("book_chapters")
                folder_name = None

                if book_chapters_dir.exists():
                    for folder in book_chapters_dir.iterdir():
                        if folder.is_dir():
                            clean_folder_name = folder.name.replace("_chapters", "").replace("-", " ").replace("_", " ").lower()
                            clean_book_title = book_title.lower().strip()
                            if clean_book_title in clean_folder_name or clean_folder_name in clean_book_title:
                                folder_name = folder.name
                                break

                if folder_name:
                    # Mark as completed in state system
                    state_manager.mark_book_document_completed(
                        folder_name,
                        total_chapters=len(book_chunks),
                        successful=len(book_chunks)
                    )
                    state_manager.mark_book_embedding_completed(
                        folder_name,
                        successful=len(book_chunks)
                    )

                    logger.info(f"Detected processed book: {book_title} -> {folder_name} ({len(book_chunks)} chunks)")

            # Mark unified index as built
            state_manager.mark_unified_index_built(
                books_included=list(books_found.keys()),
                total_chunks=len(chunks_data)
            )

            logger.info(f"Successfully rebuilt state for {len(books_found)} books with {len(chunks_data)} total chunks")

        except Exception as e:
            logger.error(f"Failed to rebuild state from data: {e}")
            raise

    else:
        logger.info("No production data found, starting fresh")


if __name__ == "__main__":
    # Test the state manager
    logging.basicConfig(level=logging.INFO)
    
    state_manager = StateManager()

    # Rebuild state from production data
    rebuild_state_from_data(state_manager)
    
    # Print summary
    summary = state_manager.get_processing_summary()
    print("Processing State Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate consistency
    validation = state_manager.validate_consistency()
    print(f"\nConsistency Check: {'✅ PASSED' if validation['consistent'] else '❌ FAILED'}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"  - {issue}")