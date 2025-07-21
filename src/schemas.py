"""
Metadata schemas for PM Knowledge Assistant
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Enhanced metadata schema for document chunks"""
    book_title: str = Field(..., description="Book title derived from folder name")
    author_name: Optional[str] = Field(None, description="Author name extracted from book content")
    chapter: str = Field(..., description="Chapter name/number from file or content")
    section: Optional[str] = Field(None, description="Section name if nested folders exist")
    file_path: str = Field(..., description="Full path to original chapter file")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Chapter text content")
    folder_structure: str = Field(..., description="Complete folder hierarchy")
    page_count: Optional[int] = Field(None, description="Number of pages in the chapter")
    word_count: Optional[int] = Field(None, description="Number of words in the content")
    
    class Config:
        extra = "allow"  # Allow additional fields


class ProcessingResult(BaseModel):
    """Result of processing a single book"""
    book_title: str
    author_name: Optional[str]
    total_chapters: int
    total_chunks: int
    errors: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)