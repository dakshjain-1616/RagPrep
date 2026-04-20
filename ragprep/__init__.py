"""
RAGPrep - A RAG-prep pipeline built on top of microsoft/markitdown.

This package provides tools to convert mixed-format documents into a queryable
ChromaDB vector store with smart chunking and local embeddings.
"""

__version__ = "0.1.0"
__author__ = "RAGPrep Team"

from ragprep.converter import DocumentConverter
from ragprep.chunker import MarkdownChunker
from ragprep.storage import StorageManager
from ragprep.ingest import IngestPipeline
from ragprep.query import QueryEngine

__all__ = [
    "DocumentConverter",
    "MarkdownChunker",
    "StorageManager",
    "IngestPipeline",
    "QueryEngine",
]
