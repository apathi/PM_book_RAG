"""
Generate embeddings for PM book chapters using OpenAI
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
from tqdm import tqdm
import pickle

from .schemas import ChunkMetadata

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """Generate and store embeddings for book chapters"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.embeddings = []
        self.metadata = []
        
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for a text chunk"""
        try:
            # Truncate text if too long (max ~8000 tokens)
            max_chars = 30000  # Conservative estimate
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.warning(f"Truncated text to {max_chars} characters")
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return None
    
    def process_chunks(self, chunks: List[ChunkMetadata]) -> Dict[str, any]:
        """Process all chunks and generate embeddings"""
        results = {
            "total_chunks": len(chunks),
            "successful": 0,
            "failed": 0,
            "embeddings": [],
            "metadata": []
        }
        
        logger.info(f"Processing {len(chunks)} chunks for embeddings...")
        
        for chunk in tqdm(chunks, desc="Generating embeddings"):
            # Create text for embedding (combine metadata with content)
            embedding_text = f"""
Book: {chunk.book_title}
Author: {chunk.author_name or 'Unknown'}
Chapter: {chunk.chapter}
Section: {chunk.section or 'Main'}

{chunk.content}
"""
            
            embedding = self.create_embedding(embedding_text)
            
            if embedding:
                results["embeddings"].append(embedding)
                results["metadata"].append(chunk.model_dump())
                results["successful"] += 1
            else:
                results["failed"] += 1
                logger.error(f"Failed to create embedding for {chunk.chunk_id}")
        
        self.embeddings = results["embeddings"]
        self.metadata = results["metadata"]
        
        return results
    
    def create_faiss_index(self, dimension: int = 1536) -> faiss.IndexFlatL2:
        """Create FAISS index from embeddings"""
        if not self.embeddings:
            raise ValueError("No embeddings to index")
        
        # Convert to numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        
        return index
    
    def save_index_and_metadata(self, index_path: str = "data/faiss_index.bin", 
                               metadata_path: str = "data/chunks_metadata.json"):
        """Save FAISS index and metadata to disk"""
        # Create data directory
        Path(index_path).parent.mkdir(exist_ok=True)
        
        # Create and save FAISS index
        index = self.create_faiss_index()
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Also save embeddings for backup
        embeddings_path = Path(metadata_path).parent / "embeddings.pkl"
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        logger.info(f"Saved embeddings backup to {embeddings_path}")
    
    def load_index_and_metadata(self, index_path: str = "data/faiss_index.bin",
                               metadata_path: str = "data/chunks_metadata.json"):
        """Load FAISS index and metadata from disk"""
        # Load FAISS index
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"Loaded {len(metadata)} metadata entries")
        
        return index, metadata
    
    def search_similar(self, query: str, index: faiss.IndexFlatL2, 
                      metadata: List[Dict], k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        # Create query embedding
        query_embedding = self.create_embedding(query)
        if not query_embedding:
            return []
        
        # Convert to numpy array
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = index.search(query_vector, k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata):
                result = metadata[idx].copy()
                result['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
                result['rank'] = i + 1
                results.append(result)
        
        return results


def process_all_books(chunks_file: str = "data/chunks_metadata.json"):
    """Process all chunks and create embeddings"""
    # Load processed chunks
    if not Path(chunks_file).exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Convert to ChunkMetadata objects
    chunks = [ChunkMetadata(**chunk_dict) for chunk_dict in chunks_data]
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
    
    # Initialize generator
    generator = EmbeddingsGenerator()
    
    # Process chunks
    results = generator.process_chunks(chunks)
    
    logger.info(f"Successfully processed {results['successful']} chunks")
    logger.info(f"Failed to process {results['failed']} chunks")
    
    # Save index and metadata
    generator.save_index_and_metadata()
    
    return results


if __name__ == "__main__":
    # Test with a small subset first
    print("Note: Make sure to set your OPENAI_API_KEY in the .env file!")
    
    # You can test with a small chunk first
    test_chunk = ChunkMetadata(
        book_title="Test Book",
        author_name="Test Author",
        chapter="Chapter 1",
        section=None,
        file_path="test.pdf",
        chunk_id="test_1",
        content="This is a test content for embedding generation.",
        folder_structure="test",
        word_count=10
    )
    
    generator = EmbeddingsGenerator()
    embedding = generator.create_embedding(test_chunk.content)
    
    if embedding:
        print(f"Successfully created embedding with dimension: {len(embedding)}")
    else:
        print("Failed to create embedding. Check your API key.")