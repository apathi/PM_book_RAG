"""
Incremental embedding manager for robust embedding generation
Handles per-book embeddings with checkpointing and recovery
"""
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from datetime import datetime

from .embeddings_generator import EmbeddingsGenerator
from .schemas import ChunkMetadata
from .state_manager import StateManager, ProcessingStatus

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Incremental embedding generation with robust recovery
    
    Features:
    - Process embeddings per book with isolation
    - Checkpoint embeddings during generation
    - Resume from interruptions
    - Merge book embeddings into unified index
    - Compatible with existing EmbeddingsGenerator
    """
    
    def __init__(self, state_manager: StateManager, embedding_model: str = "text-embedding-3-small"):
        self.state_manager = state_manager
        self.embedding_model = embedding_model
        # Use existing embeddings generator for compatibility
        self.embeddings_generator = EmbeddingsGenerator(embedding_model)
    
    def load_book_chunks(self, book_name: str) -> Optional[List[ChunkMetadata]]:
        """Load processed chunks for a book"""
        try:
            books_dir = self.state_manager.books_dir
            chunks_file = books_dir / f"{book_name}_chunks.json"
            
            if not chunks_file.exists():
                logger.error(f"Chunks file not found for book: {book_name}")
                return None
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = [ChunkMetadata(**chunk_dict) for chunk_dict in chunks_data]
            logger.info(f"Loaded {len(chunks)} chunks for book: {book_name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load chunks for {book_name}: {e}")
            return None
    
    def save_book_embeddings(self, book_name: str, embeddings: List[List[float]], 
                           metadata: List[Dict], successful_count: int, failed_count: int) -> bool:
        """Save embeddings for a book as checkpoint"""
        try:
            books_dir = self.state_manager.books_dir
            
            # Save embeddings
            embeddings_file = books_dir / f"{book_name}_embeddings.pkl"
            temp_embeddings_file = embeddings_file.with_suffix('.tmp')
            
            with open(temp_embeddings_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'successful_count': successful_count,
                    'failed_count': failed_count,
                    'model': self.embedding_model,
                    'created_at': datetime.now().isoformat()
                }, f)
            
            temp_embeddings_file.rename(embeddings_file)
            
            # Create FAISS index for this book
            if embeddings:
                faiss_file = books_dir / f"{book_name}_faiss.bin"
                temp_faiss_file = faiss_file.with_suffix('.tmp')
                
                # Convert to numpy array
                embeddings_array = np.array(embeddings).astype('float32')
                
                # Create FAISS index
                dimension = len(embeddings[0])
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_array)
                
                # Save index
                faiss.write_index(index, str(temp_faiss_file))
                temp_faiss_file.rename(faiss_file)
                
                logger.info(f"Saved embeddings and FAISS index for {book_name}: {len(embeddings)} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embeddings for {book_name}: {e}")
            return False
    
    def load_book_embeddings(self, book_name: str) -> Optional[Dict]:
        """Load embeddings for a book from checkpoint"""
        try:
            books_dir = self.state_manager.books_dir
            embeddings_file = books_dir / f"{book_name}_embeddings.pkl"
            
            if not embeddings_file.exists():
                return None
            
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded embeddings for {book_name}: {len(data['embeddings'])} vectors")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load embeddings for {book_name}: {e}")
            return None
    
    def validate_book_embeddings(self, book_name: str) -> Dict[str, any]:
        """Validate integrity of book embeddings"""
        validation_result = {
            "book_name": book_name,
            "valid": True,
            "issues": [],
            "stats": {}
        }
        
        try:
            book_status = self.state_manager.get_book_status(book_name)
            
            # Check if marked as completed
            if book_status.embedding_status == ProcessingStatus.COMPLETED:
                # Load embeddings data
                embeddings_data = self.load_book_embeddings(book_name)
                
                if embeddings_data is None:
                    validation_result["valid"] = False
                    validation_result["issues"].append("Book marked as embedding-completed but no embeddings file found")
                else:
                    embeddings = embeddings_data['embeddings']
                    metadata = embeddings_data['metadata']
                    
                    # Check consistency
                    if len(embeddings) != len(metadata):
                        validation_result["issues"].append(
                            f"Embedding/metadata count mismatch: {len(embeddings)} vs {len(metadata)}"
                        )
                    
                    if len(embeddings) != book_status.successful_embeddings:
                        validation_result["issues"].append(
                            f"Embedding count mismatch with state: {len(embeddings)} vs {book_status.successful_embeddings}"
                        )
                    
                    # Check FAISS index exists
                    faiss_file = self.state_manager.books_dir / f"{book_name}_faiss.bin"
                    if not faiss_file.exists():
                        validation_result["issues"].append("FAISS index file missing")
                    else:
                        # Validate FAISS index
                        try:
                            index = faiss.read_index(str(faiss_file))
                            if index.ntotal != len(embeddings):
                                validation_result["issues"].append(
                                    f"FAISS index size mismatch: {index.ntotal} vs {len(embeddings)}"
                                )
                        except Exception as e:
                            validation_result["issues"].append(f"FAISS index validation error: {e}")
                    
                    validation_result["stats"] = {
                        "embedding_count": len(embeddings),
                        "dimension": len(embeddings[0]) if embeddings else 0,
                        "successful_embeddings": embeddings_data.get('successful_count', 0),
                        "failed_embeddings": embeddings_data.get('failed_count', 0),
                        "model_used": embeddings_data.get('model', 'unknown'),
                        "created_at": embeddings_data.get('created_at', 'unknown')
                    }
            
            validation_result["valid"] = len(validation_result["issues"]) == 0
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def process_book_embeddings(self, book_name: str, force_restart: bool = False) -> Dict[str, any]:
        """
        Process embeddings for a single book atomically
        
        Args:
            book_name: Name of the book to process
            force_restart: If True, ignore existing embeddings and restart
        
        Returns:
            Dict with processing results and statistics
        """
        logger.info(f"Starting embedding processing for book: {book_name}")
        
        # Check if already completed and not forcing restart
        book_status = self.state_manager.get_book_status(book_name)
        
        if (book_status.embedding_status == ProcessingStatus.COMPLETED and 
            not force_restart):
            
            logger.info(f"Book {book_name} embeddings already completed")
            existing_data = self.load_book_embeddings(book_name)
            if existing_data:
                return {
                    "success": True,
                    "book_name": book_name,
                    "total_embeddings": len(existing_data['embeddings']),
                    "successful": existing_data.get('successful_count', 0),
                    "failed": existing_data.get('failed_count', 0),
                    "loaded_from_checkpoint": True,
                    "processing_time": 0
                }
        
        # Check if document processing is completed
        if book_status.document_status != ProcessingStatus.COMPLETED:
            error_msg = f"Cannot process embeddings for {book_name}: document processing not completed"
            logger.error(error_msg)
            self.state_manager.mark_book_embedding_failed(book_name, error_msg)
            return {
                "success": False,
                "book_name": book_name,
                "error": error_msg
            }
        
        # Load chunks
        chunks = self.load_book_chunks(book_name)
        if not chunks:
            error_msg = f"No chunks found for book: {book_name}"
            logger.error(error_msg)
            self.state_manager.mark_book_embedding_failed(book_name, error_msg)
            return {
                "success": False,
                "book_name": book_name,
                "error": error_msg
            }
        
        # Mark embedding processing as started
        self.state_manager.mark_book_embedding_started(book_name)
        
        start_time = datetime.now()
        
        try:
            # Use existing embeddings generator for compatibility
            # Reset its state for this book
            self.embeddings_generator.embeddings = []
            self.embeddings_generator.metadata = []
            
            # Process chunks - this is the expensive OpenAI API operation
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            results = self.embeddings_generator.process_chunks(chunks)
            
            # Extract results
            embeddings = results["embeddings"]
            metadata = results["metadata"]
            successful_count = results["successful"]
            failed_count = results["failed"]
            
            # Save checkpoint immediately after successful generation
            if embeddings:
                checkpoint_saved = self.save_book_embeddings(
                    book_name, embeddings, metadata, successful_count, failed_count
                )
                
                if not checkpoint_saved:
                    error_msg = f"Failed to save embeddings checkpoint for {book_name}"
                    logger.error(error_msg)
                    self.state_manager.mark_book_embedding_failed(book_name, error_msg)
                    return {
                        "success": False,
                        "book_name": book_name,
                        "error": error_msg
                    }
            
            # Update state
            if failed_count > 0:
                # Partial success
                logger.warning(f"Book {book_name} had {failed_count} embedding failures")
            
            self.state_manager.mark_book_embedding_completed(
                book_name, successful_count, failed_count
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully processed embeddings for {book_name}: {successful_count} successful, {failed_count} failed")
            
            return {
                "success": True,
                "book_name": book_name,
                "total_embeddings": len(embeddings),
                "successful": successful_count,
                "failed": failed_count,
                "processing_time": processing_time,
                "checkpoint_saved": True
            }
            
        except Exception as e:
            # Complete failure
            error_msg = f"Embedding processing failed for {book_name}: {str(e)}"
            logger.error(error_msg)
            self.state_manager.mark_book_embedding_failed(book_name, error_msg)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": False,
                "book_name": book_name,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def get_books_ready_for_embedding(self) -> List[str]:
        """Get list of books ready for embedding processing"""
        books = []
        
        for book_name, book_status in self.state_manager.state.books.items():
            if (book_status.document_status == ProcessingStatus.COMPLETED and 
                book_status.embedding_status in [ProcessingStatus.NOT_STARTED, ProcessingStatus.FAILED]):
                books.append(book_name)
        
        return books
    
    def resume_embedding_processing(self, book_name: str) -> Dict[str, any]:
        """
        Resume embedding processing that was interrupted
        
        This handles cases where embedding generation was interrupted
        """
        logger.info(f"Attempting to resume embedding processing for: {book_name}")
        
        book_status = self.state_manager.get_book_status(book_name)
        
        # Check if we have existing embeddings
        existing_data = self.load_book_embeddings(book_name)
        
        if existing_data and book_status.embedding_status == ProcessingStatus.IN_PROGRESS:
            # We have embeddings but status is still in progress
            # Check if the data is complete
            chunks = self.load_book_chunks(book_name)
            
            if chunks and len(existing_data['embeddings']) >= len(chunks):
                # We have all embeddings, mark as completed
                logger.info(f"Found complete embeddings for {book_name}, marking as completed")
                
                self.state_manager.mark_book_embedding_completed(
                    book_name,
                    successful=existing_data.get('successful_count', len(existing_data['embeddings'])),
                    failed=existing_data.get('failed_count', 0)
                )
                
                return {
                    "success": True,
                    "book_name": book_name,
                    "total_embeddings": len(existing_data['embeddings']),
                    "successful": existing_data.get('successful_count', 0),
                    "failed": existing_data.get('failed_count', 0),
                    "resumed_and_completed": True
                }
        
        # If we get here, restart processing
        logger.info(f"Restarting embedding processing for {book_name}")
        return self.process_book_embeddings(book_name, force_restart=True)
    
    def cleanup_book_embeddings(self, book_name: str):
        """Clean up embedding artifacts for a book"""
        try:
            books_dir = self.state_manager.books_dir
            
            # Remove files
            for suffix in ['_embeddings.pkl', '_faiss.bin']:
                file_path = books_dir / f"{book_name}{suffix}"
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed {file_path}")
            
            # Reset embedding status
            book_status = self.state_manager.get_book_status(book_name)
            self.state_manager.update_book_status(
                book_name,
                embedding_status=ProcessingStatus.NOT_STARTED,
                successful_embeddings=0,
                failed_embeddings=0
            )
            
            logger.info(f"Cleaned up embeddings for {book_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup embeddings for {book_name}: {e}")


class UnifiedIndexBuilder:
    """
    Builds unified index from individual book embeddings
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
    
    def build_unified_index(self, force_rebuild: bool = False) -> Dict[str, any]:
        """
        Build unified index from all completed books
        
        Args:
            force_rebuild: If True, rebuild even if not needed
        
        Returns:
            Dict with build results
        """
        logger.info("Building unified index from book embeddings")
        
        # Check if rebuild is needed
        if not force_rebuild and not self.state_manager.needs_unified_index_rebuild():
            logger.info("Unified index is up to date")
            return {
                "success": True,
                "rebuilt": False,
                "reason": "index_up_to_date"
            }
        
        # Get completed books
        completed_books = self.state_manager.get_completed_books() 
        if not completed_books:
            logger.warning("No completed books found for unified index")
            return {
                "success": False,
                "error": "no_completed_books"
            }
        
        logger.info(f"Building unified index from {len(completed_books)} books")
        
        try:
            all_embeddings = []
            all_metadata = []
            books_included = []
            
            # Collect embeddings from all books
            embedding_manager = EmbeddingManager(self.state_manager)
            
            for book_name in completed_books:
                book_data = embedding_manager.load_book_embeddings(book_name)
                if book_data:
                    all_embeddings.extend(book_data['embeddings'])
                    all_metadata.extend(book_data['metadata'])
                    books_included.append(book_name)
                    logger.info(f"Added {len(book_data['embeddings'])} embeddings from {book_name}")
                else:
                    logger.warning(f"Could not load embeddings for completed book: {book_name}")
            
            if not all_embeddings:
                return {
                    "success": False,
                    "error": "no_embeddings_found"
                }
            
            # Create unified FAISS index
            embeddings_array = np.array(all_embeddings).astype('float32')
            dimension = embeddings_array.shape[1]
            
            unified_index = faiss.IndexFlatL2(dimension)
            unified_index.add(embeddings_array)
            
            # Save production files
            metadata_file = self.state_manager.data_dir / "chunks_metadata.json"
            faiss_file = self.state_manager.data_dir / "faiss_index.bin"
            embeddings_file = self.state_manager.data_dir / "embeddings.pkl"

            # Save metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)

            # Save FAISS index
            faiss.write_index(unified_index, str(faiss_file))

            # Save embeddings backup
            with open(embeddings_file, 'wb') as f:
                pickle.dump(all_embeddings, f)
            
            # Update state
            self.state_manager.mark_unified_index_built(books_included, len(all_metadata))
            
            logger.info(f"Successfully built unified index: {len(all_embeddings)} embeddings from {len(books_included)} books")
            
            return {
                "success": True,
                "rebuilt": True,
                "total_embeddings": len(all_embeddings),
                "books_included": books_included,
                "dimension": dimension
            }
            
        except Exception as e:
            error_msg = f"Failed to build unified index: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }


# Utility functions for backward compatibility
def process_book_embeddings_legacy_compatible(book_name: str, state_manager: StateManager = None) -> Dict:
    """
    Process embeddings for a book with full backward compatibility
    """
    if state_manager is None:
        state_manager = StateManager()
    
    embedding_manager = EmbeddingManager(state_manager)
    return embedding_manager.process_book_embeddings(book_name)


if __name__ == "__main__":
    # Test the embedding manager
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    state_manager = StateManager()
    embedding_manager = EmbeddingManager(state_manager)
    
    # Show books ready for embedding
    ready_books = embedding_manager.get_books_ready_for_embedding()
    print(f"Books ready for embedding processing: {ready_books}")
    
    # Test with first ready book if any
    if ready_books:
        test_book = ready_books[0]
        print(f"\nTesting embedding processing with: {test_book}")
        
        # Note: This would make actual OpenAI API calls
        print("Skipping actual processing to avoid API costs")
        print("Use embedding_manager.process_book_embeddings(book_name) to process")
        
        # Test validation instead
        validation = embedding_manager.validate_book_embeddings(test_book)
        print(f"Embedding validation result: {validation}")
    
    # Test unified index builder
    index_builder = UnifiedIndexBuilder(state_manager)
    result = index_builder.build_unified_index()
    print(f"\nUnified index build result: {result}")