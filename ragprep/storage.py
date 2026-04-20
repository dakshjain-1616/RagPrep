"""
Storage module for file hash tracking and ChromaDB vector storage.

Provides:
- SQLite-based file hash tracking for incremental updates
- ChromaDB client with sentence-transformers embeddings
- Metadata management for chunks
"""

import os
import json
import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ragprep.chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class FileRecord:
    """Record of a processed file."""
    filepath: str
    file_hash: str
    last_modified: float
    processed_at: str
    chunk_count: int
    status: str  # 'success', 'error', 'pending'
    error_message: Optional[str] = None


class HashTracker:
    """
    SQLite-based file hash tracker for incremental updates.
    
    Tracks which files have been processed and their hashes to avoid
    re-processing unchanged files.
    
    Example:
        >>> tracker = HashTracker("./data/ragprep.db")
        >>> tracker.record_file("doc.pdf", hash, chunks=5)
        >>> if tracker.needs_update("doc.pdf", new_hash):
        ...     print("File changed, re-processing needed")
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the hash tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db()
        logger.info(f"HashTracker initialized: {db_path}")
    
    def _ensure_db(self) -> None:
        """Create database tables if they don't exist."""
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_records (
                    filepath TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    last_modified REAL NOT NULL,
                    processed_at TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    files_processed INTEGER DEFAULT 0,
                    files_failed INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                )
            """)
            
            conn.commit()
    
    def compute_hash(self, filepath: str) -> str:
        """
        Compute SHA-256 hash of file contents.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def needs_update(self, filepath: str, current_hash: Optional[str] = None) -> bool:
        """
        Check if a file needs to be re-processed.
        
        Args:
            filepath: Path to the file
            current_hash: Optional pre-computed hash
            
        Returns:
            True if file is new or has changed
        """
        if current_hash is None:
            try:
                current_hash = self.compute_hash(filepath)
            except Exception as e:
                logger.error(f"Failed to compute hash for {filepath}: {e}")
                return True  # Re-process on error
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_hash FROM file_records WHERE filepath = ?",
                (filepath,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return True  # New file
            
            stored_hash = row[0]
            return stored_hash != current_hash
    
    def record_file(
        self, 
        filepath: str, 
        file_hash: str,
        chunk_count: int = 0,
        status: str = 'success',
        error_message: Optional[str] = None
    ) -> None:
        """
        Record a file as processed.
        
        Args:
            filepath: Path to the file
            file_hash: Hash of file contents
            chunk_count: Number of chunks created
            status: Processing status
            error_message: Error message if failed
        """
        last_modified = os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0
        processed_at = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_records 
                (filepath, file_hash, last_modified, processed_at, chunk_count, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (filepath, file_hash, last_modified, processed_at, chunk_count, status, error_message)
            )
            conn.commit()
    
    def get_record(self, filepath: str) -> Optional[FileRecord]:
        """
        Get record for a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            FileRecord if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM file_records WHERE filepath = ?",
                (filepath,)
            )
            row = cursor.fetchone()
            
            if row:
                return FileRecord(
                    filepath=row[0],
                    file_hash=row[1],
                    last_modified=row[2],
                    processed_at=row[3],
                    chunk_count=row[4],
                    status=row[5],
                    error_message=row[6]
                )
            return None
    
    def get_all_records(self) -> List[FileRecord]:
        """
        Get all file records.
        
        Returns:
            List of FileRecord objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM file_records")
            rows = cursor.fetchall()
            
            return [
                FileRecord(
                    filepath=row[0],
                    file_hash=row[1],
                    last_modified=row[2],
                    processed_at=row[3],
                    chunk_count=row[4],
                    status=row[5],
                    error_message=row[6]
                )
                for row in rows
            ]
    
    def delete_record(self, filepath: str) -> None:
        """
        Delete a file record.
        
        Args:
            filepath: Path to the file
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM file_records WHERE filepath = ?",
                (filepath,)
            )
            conn.commit()
    
    def start_ingest_run(self) -> int:
        """
        Start a new ingest run and return its ID.
        
        Returns:
            Run ID
        """
        started_at = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO ingest_runs (started_at) VALUES (?)",
                (started_at,)
            )
            conn.commit()
            return cursor.lastrowid
    
    def complete_ingest_run(
        self, 
        run_id: int, 
        files_processed: int = 0,
        files_failed: int = 0,
        total_chunks: int = 0
    ) -> None:
        """
        Mark an ingest run as complete.
        
        Args:
            run_id: The run ID
            files_processed: Number of files processed
            files_failed: Number of files that failed
            total_chunks: Total chunks created
        """
        completed_at = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE ingest_runs 
                SET completed_at = ?, files_processed = ?, files_failed = ?, 
                    total_chunks = ?, status = 'completed'
                WHERE id = ?
                """,
                (completed_at, files_processed, files_failed, total_chunks, run_id)
            )
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processed files.
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_files,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                    SUM(chunk_count) as total_chunks
                FROM file_records
            """)
            row = cursor.fetchone()
            
            return {
                'total_files': row[0] or 0,
                'success_count': row[1] or 0,
                'error_count': row[2] or 0,
                'total_chunks': row[3] or 0
            }


class StorageManager:
    """
    Manages ChromaDB vector storage with sentence-transformers embeddings.
    
    Provides a unified interface for:
    - Storing document chunks with embeddings
    - Querying similar chunks
    - Managing collection metadata
    
    Example:
        >>> storage = StorageManager("./data/chroma")
        >>> storage.add_chunks(chunks)
        >>> results = storage.query("what is AI?", n_results=5)
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(
        self, 
        chroma_path: str,
        collection_name: str = "ragprep_docs",
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the storage manager.
        
        Args:
            chroma_path: Path to ChromaDB persistence directory
            collection_name: Name of the collection
            embedding_model: Sentence-transformers model name (default: all-MiniLM-L6-v2)
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required but not installed. "
                "Install with: pip install chromadb"
            )
        
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required but not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model or self.DEFAULT_MODEL
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self._embedding_model = SentenceTransformer(self.embedding_model_name)
        self._embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._embedding_dim}")
        
        # Initialize ChromaDB client
        os.makedirs(chroma_path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"StorageManager initialized: {chroma_path}/{collection_name}")
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using the sentence-transformers model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = self._embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def add_chunks(
        self, 
        chunks: List[TextChunk],
        file_id: Optional[str] = None
    ) -> List[str]:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects
            file_id: Optional file identifier for grouping
            
        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []
        
        # Generate IDs
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ids = [f"chunk_{timestamp}_{i}" for i in range(len(chunks))]
        
        # Prepare data
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'chunk_type': chunk.chunk_type.value,
                'start_pos': chunk.start_pos,
                'end_pos': chunk.end_pos,
                'word_count': chunk.word_count,
                'char_count': chunk.char_count,
                'heading_path': json.dumps(chunk.heading_path),
            }
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
            if file_id:
                metadata['file_id'] = file_id
            metadatas.append(metadata)
        
        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to storage")
        return ids
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for similar chunks.
        
        Args:
            query_text: The query text
            n_results: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of result dictionaries with chunk content and metadata
        """
        # Embed query
        query_embedding = self.embed_texts([query_text])[0]
        
        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                result = {
                    'id': chunk_id,
                    'content': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                
                # Parse heading path from JSON
                if 'heading_path' in result['metadata']:
                    result['metadata']['heading_path'] = json.loads(
                        result['metadata']['heading_path']
                    )
                
                formatted_results.append(result)
        
        return formatted_results
    
    def delete_by_file(self, file_id: str) -> int:
        """
        Delete all chunks for a specific file.
        
        Args:
            file_id: The file identifier
            
        Returns:
            Number of chunks deleted
        """
        # Get all chunks for this file
        results = self._collection.get(
            where={"file_id": file_id}
        )
        
        if results['ids']:
            self._collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks for file {file_id}")
            return len(results['ids'])
        
        return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self._collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self._embedding_dim,
            'chroma_path': self.chroma_path
        }
    
    def reset_collection(self) -> None:
        """Delete all chunks from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset collection: {self.collection_name}")
