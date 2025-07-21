"""
Atomic book processor for robust document processing
Handles individual book processing with checkpointing and recovery
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .document_processor import DocumentProcessor
from .schemas import ChunkMetadata, ProcessingResult
from .state_manager import StateManager, ProcessingStatus

logger = logging.getLogger(__name__)


class BookProcessor:
    """
    Atomic book processing with checkpointing and recovery
    
    Features:
    - Process books individually with full isolation
    - Checkpoint progress during processing
    - Resume from interruptions
    - Validate book integrity
    - Compatible with existing DocumentProcessor
    """
    
    def __init__(self, state_manager: StateManager, base_path: str = "book_chapters"):
        self.state_manager = state_manager
        self.base_path = Path(base_path)
        self.document_processor = DocumentProcessor(base_path)
        
    def get_book_folder_path(self, book_name: str) -> Optional[Path]:
        """Get the folder path for a book by name"""
        book_folder = self.base_path / book_name
        if book_folder.exists() and book_folder.is_dir():
            return book_folder
        return None
    
    def validate_book_folder(self, book_folder: Path) -> Tuple[bool, List[str]]:
        """Validate that a book folder is ready for processing"""
        issues = []
        
        if not book_folder.exists():
            issues.append(f"Book folder does not exist: {book_folder}")
            return False, issues
        
        if not book_folder.is_dir():
            issues.append(f"Book path is not a directory: {book_folder}")
            return False, issues
        
        # Check for PDF files
        pdf_files = list(book_folder.rglob("*.pdf"))
        actual_pdfs = [f for f in pdf_files if "processing_report" not in f.name]
        
        if not actual_pdfs:
            issues.append(f"No PDF files found in book folder: {book_folder}")
            return False, issues
        
        logger.info(f"Book folder validated: {len(actual_pdfs)} PDF files found")
        return True, []
    
    def save_book_checkpoint(self, book_name: str, chunks: List[ChunkMetadata]) -> bool:
        """Save processed chunks for a book as checkpoint"""
        try:
            books_dir = self.state_manager.books_dir
            checkpoint_file = books_dir / f"{book_name}_chunks.json"
            
            # Convert chunks to serializable format
            chunks_data = [chunk.model_dump() for chunk in chunks]
            
            # Save with atomic write (write to temp file, then rename)
            temp_file = checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.rename(checkpoint_file)
            
            logger.info(f"Saved checkpoint for {book_name}: {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {book_name}: {e}")
            return False
    
    def load_book_checkpoint(self, book_name: str) -> Optional[List[ChunkMetadata]]:
        """Load processed chunks from checkpoint"""
        try:
            books_dir = self.state_manager.books_dir
            checkpoint_file = books_dir / f"{book_name}_chunks.json"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Convert back to ChunkMetadata objects
            chunks = [ChunkMetadata(**chunk_dict) for chunk_dict in chunks_data]
            
            logger.info(f"Loaded checkpoint for {book_name}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {book_name}: {e}")
            return None
    
    def validate_book_integrity(self, book_name: str) -> Dict[str, any]:
        """Validate integrity of processed book data"""
        validation_result = {
            "book_name": book_name,
            "valid": True,
            "issues": [],
            "stats": {}
        }
        
        try:
            # Check state consistency
            book_status = self.state_manager.get_book_status(book_name)
            
            # If marked as completed, check data files exist
            if book_status.document_status == ProcessingStatus.COMPLETED:
                chunks = self.load_book_checkpoint(book_name)
                
                if chunks is None:
                    validation_result["valid"] = False
                    validation_result["issues"].append("Book marked as completed but no checkpoint file found")
                else:
                    # Check chunk count consistency
                    if len(chunks) != book_status.successful_chapters:
                        validation_result["issues"].append(
                            f"Chunk count mismatch: {len(chunks)} chunks vs {book_status.successful_chapters} expected"
                        )
                    
                    # Check for required fields
                    required_fields = ['book_title', 'chapter', 'content', 'chunk_id']
                    for i, chunk in enumerate(chunks):
                        for field in required_fields:
                            if not getattr(chunk, field, None):
                                validation_result["issues"].append(f"Chunk {i} missing required field: {field}")
                    
                    validation_result["stats"] = {
                        "total_chunks": len(chunks),
                        "total_words": sum(chunk.word_count for chunk in chunks if chunk.word_count),
                        "chapters_with_content": len([c for c in chunks if c.content and len(c.content.strip()) > 0])
                    }
            
            # Final validation
            validation_result["valid"] = len(validation_result["issues"]) == 0
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def process_book_atomic(self, book_name: str, force_restart: bool = False) -> ProcessingResult:
        """
        Process a single book atomically with full error handling and checkpointing
        
        Args:
            book_name: Name of the book folder to process
            force_restart: If True, ignore existing checkpoints and restart from scratch
        
        Returns:
            ProcessingResult with detailed processing information
        """
        logger.info(f"Starting atomic processing of book: {book_name}")
        
        # Get book folder path
        book_folder = self.get_book_folder_path(book_name)
        if not book_folder:
            error_msg = f"Book folder not found: {book_name}"
            logger.error(error_msg)
            self.state_manager.mark_book_document_failed(book_name, error_msg)
            return ProcessingResult(
                book_title=book_name,
                author_name=None,
                total_chapters=0,
                total_chunks=0,
                errors=[error_msg]
            )
        
        # Validate book folder
        is_valid, validation_issues = self.validate_book_folder(book_folder)
        if not is_valid:
            error_msg = f"Book folder validation failed: {'; '.join(validation_issues)}"
            logger.error(error_msg)
            self.state_manager.mark_book_document_failed(book_name, error_msg)
            return ProcessingResult(
                book_title=book_name,
                author_name=None,
                total_chapters=0,
                total_chunks=0,
                errors=validation_issues
            )
        
        # Check if already completed and not forcing restart
        book_status = self.state_manager.get_book_status(book_name)
        if (book_status.document_status == ProcessingStatus.COMPLETED and 
            not force_restart):
            
            logger.info(f"Book {book_name} already completed, loading from checkpoint")
            chunks = self.load_book_checkpoint(book_name)
            if chunks:
                return ProcessingResult(
                    book_title=chunks[0].book_title if chunks else book_name,
                    author_name=chunks[0].author_name if chunks else None,
                    total_chapters=book_status.total_chapters,
                    total_chunks=len(chunks),
                    errors=[],
                    metadata={
                        "loaded_from_checkpoint": True,
                        "last_processed": book_status.last_document_update
                    }
                )
        
        # Mark processing as started
        self.state_manager.mark_book_document_started(book_name)
        
        try:
            # Use existing DocumentProcessor for the actual processing
            # This ensures 100% compatibility with existing logic
            result = self.document_processor.process_book(book_folder)
            
            # Get the processed chunks from the document processor
            # Filter chunks for this specific book
            book_chunks = [
                chunk for chunk in self.document_processor.processed_chunks
                if chunk.file_path and str(book_folder) in chunk.file_path
            ]
            
            if not book_chunks:
                # Fallback: try to match by book title
                book_title = self.document_processor.extract_book_title(book_folder.name)
                book_chunks = [
                    chunk for chunk in self.document_processor.processed_chunks
                    if chunk.book_title and chunk.book_title.lower() == book_title.lower()
                ]
            
            # Save checkpoint
            if book_chunks:
                checkpoint_saved = self.save_book_checkpoint(book_name, book_chunks)
                if not checkpoint_saved:
                    logger.warning(f"Failed to save checkpoint for {book_name}, but processing continued")
            
            # Update state based on results
            if result.errors:
                # Partial success - some errors occurred
                successful_chapters = result.total_chunks
                failed_chapters = len(result.errors)
                self.state_manager.mark_book_document_completed(
                    book_name, 
                    total_chapters=result.total_chapters,
                    successful=successful_chapters,
                    failed=failed_chapters
                )
                
                # Log errors but don't fail completely
                for error in result.errors:
                    logger.warning(f"Book {book_name} processing error: {error}")
            
            else:
                # Complete success
                self.state_manager.mark_book_document_completed(
                    book_name,
                    total_chapters=result.total_chapters,
                    successful=result.total_chunks,
                    failed=0
                )
            
            # Add metadata about atomic processing
            result.metadata = result.metadata or {}
            result.metadata.update({
                "atomic_processing": True,
                "checkpoint_saved": len(book_chunks) > 0,
                "processed_at": datetime.now().isoformat(),
                "chunks_in_checkpoint": len(book_chunks)
            })
            
            logger.info(f"Successfully processed book {book_name}: {result.total_chunks} chunks")
            return result
            
        except Exception as e:
            # Processing failed completely
            error_msg = f"Atomic processing failed for {book_name}: {str(e)}"
            logger.error(error_msg)
            self.state_manager.mark_book_document_failed(book_name, error_msg)
            
            return ProcessingResult(
                book_title=book_name,
                author_name=None,
                total_chapters=0,
                total_chunks=0,
                errors=[error_msg],
                metadata={
                    "atomic_processing": True,
                    "failed_at": datetime.now().isoformat(),
                    "error": str(e)
                }
            )
    
    def get_available_books(self) -> List[str]:
        """Get list of all available books in the book_chapters directory"""
        books = []
        
        if self.base_path.exists():
            for book_folder in self.base_path.iterdir():
                if book_folder.is_dir():
                    books.append(book_folder.name)
        
        return sorted(books)
    
    def get_processing_candidates(self) -> Dict[str, str]:
        """Get books that are candidates for processing with their status"""
        candidates = {}
        
        available_books = self.get_available_books()
        
        for book_name in available_books:
            book_status = self.state_manager.get_book_status(book_name)
            
            if book_status.document_status == ProcessingStatus.NOT_STARTED:
                candidates[book_name] = "ready_to_process"
            elif book_status.document_status == ProcessingStatus.FAILED:
                candidates[book_name] = "failed_needs_retry"
            elif book_status.document_status == ProcessingStatus.IN_PROGRESS:
                candidates[book_name] = "in_progress_may_need_recovery"
            elif book_status.document_status == ProcessingStatus.COMPLETED:
                candidates[book_name] = "completed"
        
        return candidates
    
    def cleanup_partial_processing(self, book_name: str):
        """Clean up any partial processing artifacts for a book"""
        try:
            # Remove checkpoint file
            checkpoint_file = self.state_manager.books_dir / f"{book_name}_chunks.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Removed checkpoint file for {book_name}")
            
            # Reset state
            self.state_manager.reset_book_status(book_name)
            logger.info(f"Reset state for {book_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup partial processing for {book_name}: {e}")
    
    def resume_processing(self, book_name: str) -> ProcessingResult:
        """
        Resume processing of a book that was interrupted
        
        This method intelligently handles various interruption scenarios:
        - Processing marked as in_progress but actually completed
        - Partial processing with some chapters completed
        - Complete failure requiring restart
        """
        logger.info(f"Attempting to resume processing for book: {book_name}")
        
        book_status = self.state_manager.get_book_status(book_name)
        
        # Check if we have a checkpoint
        existing_chunks = self.load_book_checkpoint(book_name)
        
        if existing_chunks and book_status.document_status == ProcessingStatus.IN_PROGRESS:
            # We have data but status is still in progress
            # Validate the data and potentially mark as completed
            
            book_folder = self.get_book_folder_path(book_name)
            if book_folder:
                # Count expected chapters
                pdf_files = list(book_folder.rglob("*.pdf"))
                actual_pdfs = [f for f in pdf_files if "processing_report" not in f.name]
                expected_chapters = len(actual_pdfs)
                
                if len(existing_chunks) >= expected_chapters:
                    # We have all expected chunks, mark as completed
                    logger.info(f"Found complete data for {book_name}, marking as completed")
                    self.state_manager.mark_book_document_completed(
                        book_name,
                        total_chapters=expected_chapters,
                        successful=len(existing_chunks),
                        failed=0
                    )
                    
                    return ProcessingResult(
                        book_title=existing_chunks[0].book_title if existing_chunks else book_name,
                        author_name=existing_chunks[0].author_name if existing_chunks else None,
                        total_chapters=expected_chapters,
                        total_chunks=len(existing_chunks),
                        errors=[],
                        metadata={
                            "resumed_from_checkpoint": True,
                            "completed_during_resume": True
                        }
                    )
        
        # If we get here, we need to restart processing
        logger.info(f"Restarting processing for {book_name}")
        return self.process_book_atomic(book_name, force_restart=True)


# Utility functions for backward compatibility
def process_single_book_legacy_compatible(book_name: str, state_manager: StateManager = None) -> ProcessingResult:
    """
    Process a single book with full backward compatibility
    This function can be used as a drop-in replacement for existing single book processing
    """
    if state_manager is None:
        state_manager = StateManager()
    
    processor = BookProcessor(state_manager)
    return processor.process_book_atomic(book_name)


if __name__ == "__main__":
    # Test the book processor
    logging.basicConfig(level=logging.INFO)
    
    # Initialize state manager and book processor
    state_manager = StateManager()
    book_processor = BookProcessor(state_manager)
    
    # Show available books and their status
    candidates = book_processor.get_processing_candidates()
    print("Available books and their processing status:")
    for book_name, status in candidates.items():
        print(f"  {book_name}: {status}")
    
    # Test with first available book if any
    ready_books = [name for name, status in candidates.items() if status == "ready_to_process"]
    if ready_books:
        test_book = ready_books[0]
        print(f"\nTesting atomic processing with: {test_book}")
        
        result = book_processor.process_book_atomic(test_book)
        print(f"Processing result:")
        print(f"  Success: {len(result.errors) == 0}")
        print(f"  Chunks: {result.total_chunks}")
        print(f"  Errors: {len(result.errors)}")
        
        # Test validation
        validation = book_processor.validate_book_integrity(test_book)
        print(f"  Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
    else:
        print("\nNo books ready for processing")