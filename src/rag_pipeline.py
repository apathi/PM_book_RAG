"""
RAG Pipeline for PM Knowledge Assistant
Handles retrieval and generation for product management queries
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dotenv import load_dotenv

import faiss
import numpy as np
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMKnowledgeRAG:
    """RAG pipeline for Product Management knowledge queries"""
    
    def __init__(self, 
                 index_path: str = "data/faiss_index.bin",
                 metadata_path: str = "data/chunks_metadata.json",
                 embedding_model: str = "text-embedding-3-small",
                 chat_model: str = "gpt-4-turbo-preview"):
        
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load index and metadata
        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
        except Exception as e:
            logger.error(f"Error loading index/metadata: {str(e)}")
            raise
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=chat_model,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template for PM queries
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Product Management expert assistant with access to a comprehensive library of PM books. 
Your role is to provide detailed, evidence-based answers to product management questions using the provided context from multiple PM books.

When answering:
1. Synthesize information from multiple sources when available
2. Always cite which book and chapter the information comes from
3. Highlight when different authors have different perspectives
4. Provide practical, actionable insights
5. Use specific examples from the books when relevant"""),
            
            ("human", """Context from PM books:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Make sure to:
- Reference specific books and chapters
- Compare different viewpoints if applicable
- Provide practical insights for product managers
- Structure your answer clearly""")
        ])
    
    def create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create embedding for a text"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding).astype('float32')
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return None
    
    def filter_chunks_by_books(self, chunks: List[Dict], selected_books: List[str]) -> List[Dict]:
        """
        Filter chunks to only include those from selected books
        
        Args:
            chunks: List of chunk dictionaries with metadata
            selected_books: List of book names to include
            
        Returns:
            Filtered list of chunks
        """
        if not selected_books:
            return chunks
        
        # Create normalized versions of selected book names for matching
        normalized_selected = set()
        for book_name in selected_books:
            # Remove _chapters suffix and normalize
            clean_name = book_name.replace("_chapters", "")
            
            # Add multiple normalized variants
            variants = [
                clean_name.lower(),
                clean_name.replace("-", " ").lower(),
                clean_name.replace("_", " ").lower(),
                clean_name.replace("-", "").lower(),
                clean_name.replace("_", "").lower(),
            ]
            normalized_selected.update(variants)
        
        filtered_chunks = []
        for chunk in chunks:
            # Get book title and normalize it
            book_title = chunk.get('book_title', '').lower().strip()
            
            # Create variants of the chunk's book title
            title_variants = [
                book_title,
                book_title.replace(" ", "-"),
                book_title.replace(" ", "_"),
                book_title.replace(" ", ""),
            ]
            
            # Check if any title variant matches any selected variant
            if any(title_variant in normalized_selected for title_variant in title_variants):
                filtered_chunks.append(chunk)
        
        return filtered_chunks

    def retrieve_relevant_chunks(self, query: str, k: int = 5, selected_books: List[str] = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query with optional book filtering
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            selected_books: Optional list of book names to filter by
            
        Returns:
            List of relevant chunks
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        if query_embedding is None:
            return []
        
        # Reshape for FAISS
        query_vector = query_embedding.reshape(1, -1)
        
        # If we have book filtering, we may need to search more broadly first
        search_k = k * 3 if selected_books else k
        
        # Search
        distances, indices = self.index.search(query_vector, search_k)
        
        # Prepare results with metadata
        all_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                chunk = self.metadata[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + dist))
                all_results.append(chunk)
        
        # Apply book filtering if specified
        if selected_books:
            filtered_results = self.filter_chunks_by_books(all_results, selected_books)
            # Return top k from filtered results
            return filtered_results[:k]
        
        return all_results
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        # Group by book for better organization
        books = {}
        for chunk in chunks:
            book_key = f"{chunk['book_title']} by {chunk.get('author_name', 'Unknown')}"
            if book_key not in books:
                books[book_key] = []
            books[book_key].append(chunk)
        
        # Format each book's content
        for book_key, book_chunks in books.items():
            context_parts.append(f"\n📚 **{book_key}**\n")
            
            for chunk in book_chunks:
                chapter_info = f"Chapter: {chunk['chapter']}"
                if chunk.get('section'):
                    chapter_info += f", Section: {chunk['section']}"
                
                context_parts.append(f"\n{chapter_info}:\n")
                
                # Truncate content if too long
                content = chunk['content']
                if len(content) > 2000:
                    content = content[:2000] + "..."
                
                context_parts.append(content)
                context_parts.append("\n" + "-" * 50)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, k: int = 5, selected_books: List[str] = None) -> Dict:
        """
        Generate answer for a question using RAG with optional book filtering
        
        Args:
            question: The question to answer
            k: Number of source chunks to use
            selected_books: Optional list of book names to filter by
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant chunks with optional filtering
        relevant_chunks = self.retrieve_relevant_chunks(question, k, selected_books)
        
        if not relevant_chunks:
            filter_msg = f" from the selected books ({', '.join(selected_books)})" if selected_books else ""
            return {
                "answer": f"I couldn't find relevant information in the PM books{filter_msg} to answer your question.",
                "sources": [],
                "chunks_used": 0,
                "selected_books": selected_books or []
            }
        
        # Format context
        context = self.format_context(relevant_chunks)
        
        # Generate answer
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            response = chain.run(context=context, question=question)
            
            # Extract sources
            sources = []
            seen_books = set()
            for chunk in relevant_chunks:
                book_info = {
                    "book_title": chunk['book_title'],
                    "author": chunk.get('author_name', 'Unknown'),
                    "chapter": chunk['chapter'],
                    "section": chunk.get('section'),
                    "similarity_score": chunk['similarity_score']
                }
                
                # Avoid duplicate book entries
                book_key = f"{book_info['book_title']}_{book_info['chapter']}"
                if book_key not in seen_books:
                    sources.append(book_info)
                    seen_books.add(book_key)
            
            return {
                "answer": response,
                "sources": sources,
                "chunks_used": len(relevant_chunks),
                "selected_books": selected_books or []
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "chunks_used": 0,
                "selected_books": selected_books or []
            }
    
    def compare_perspectives(self, topic: str, k: int = 10, selected_books: List[str] = None) -> Dict:
        """Compare different authors' perspectives on a topic with optional book filtering"""
        # Retrieve more chunks to get diverse perspectives
        chunks = self.retrieve_relevant_chunks(topic, k, selected_books)
        
        # Group by author
        author_perspectives = {}
        for chunk in chunks:
            author = chunk.get('author_name', 'Unknown')
            book = chunk['book_title']
            key = f"{author} ({book})"
            
            if key not in author_perspectives:
                author_perspectives[key] = []
            author_perspectives[key].append(chunk)
        
        # Generate comparison
        if len(author_perspectives) < 2:
            return {
                "comparison": "Not enough diverse perspectives found for comparison.",
                "authors": list(author_perspectives.keys()),
                "chunks_used": len(chunks)
            }
        
        # Create comparison prompt
        comparison_context = []
        for author_book, chunks in author_perspectives.items():
            comparison_context.append(f"\n**{author_book}:**")
            for chunk in chunks[:2]:  # Limit to 2 chunks per author
                comparison_context.append(f"- {chunk['chapter']}: {chunk['content'][:500]}...")
        
        comparison_prompt = f"""Compare and contrast different authors' perspectives on: {topic}

Authors and their views:
{''.join(comparison_context)}

Please provide:
1. Key similarities in their approaches
2. Main differences in their perspectives
3. Unique insights from each author
4. Synthesis and recommendations for product managers"""
        
        try:
            response = self.llm.predict(comparison_prompt)
            
            return {
                "comparison": response,
                "authors": list(author_perspectives.keys()),
                "chunks_used": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating comparison: {str(e)}")
            return {
                "comparison": f"Error generating comparison: {str(e)}",
                "authors": [],
                "chunks_used": 0
            }


def test_rag_pipeline():
    """Test the RAG pipeline with sample queries"""
    # Initialize RAG
    rag = PMKnowledgeRAG()
    
    # Test queries
    test_queries = [
        "What are the key skills needed for a product manager?",
        "How should product managers prioritize features?",
        "What frameworks exist for product roadmapping?"
    ]
    
    print("Testing RAG Pipeline")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = rag.generate_answer(query, k=3)
        
        print(f"\nAnswer:\n{result['answer'][:500]}...")
        print(f"\nSources used: {len(result['sources'])}")
        for source in result['sources']:
            print(f"- {source['book_title']} by {source['author']}, {source['chapter']}")


if __name__ == "__main__":
    test_rag_pipeline()