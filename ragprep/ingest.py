"""
Ingest pipeline module that orchestrates conversion → chunking → embedding → storage.

Provides incremental update logic using file hash tracking to only re-process
files that have changed since last ingest.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from ragprep.converter import DocumentConverter, ConversionStatus
from ragprep.chunker import MarkdownChunker, TextChunk
from ragprep.storage import StorageManager, HashTracker

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of an ingest operation."""
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    total_chunks: int = 0
    errors: List[Dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0
    run_id: Optional[int] = None


class IngestPipeline:
    """
    Orchestrates the document ingestion pipeline.
    
    Pipeline flow:
    1. Scan directory for supported files
    2. Check file hashes against stored records
    3. Convert new/changed files to Markdown
    4. Chunk Markdown with smart strategies
    5. Embed chunks using sentence-transformers
    6. Store in ChromaDB with metadata
    
    Example:
        >>> pipeline = IngestPipeline("./data")
        >>> result = pipeline.ingest("./documents")
        >>> print(f"Processed {result.files_processed} files, "
        ...       f"created {result.total_chunks} chunks")
    """
    
    def __init__(
        self,
        data_dir: str,
        collection_name: str = "ragprep_docs",
        embedding_model: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize the ingest pipeline.
        
        Args:
            data_dir: Directory for storing data (SQLite + ChromaDB)
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence-transformers model name
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            progress_callback: Optional callback(status, current, total)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.converter = DocumentConverter()
        self.chunker = MarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.tracker = HashTracker(str(self.data_dir / "ragprep.db"))
        self.storage = StorageManager(
            chroma_path=str(self.data_dir / "chroma"),
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        self.progress_callback = progress_callback
        
        logger.info(f"IngestPipeline initialized: {data_dir}")
    
    def _report_progress(self, status: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(status, current, total)
    
    def _scan_directory(self, directory: str) -> List[Path]:
        """
        Scan directory for supported files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        supported_exts = self.converter.get_supported_extensions()
        files = []
        
        for filepath in directory.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in supported_exts:
                files.append(filepath)
        
        logger.info(f"Found {len(files)} files in {directory}")
        return files
    
    def _process_file(self, filepath: Path) -> tuple[List[TextChunk], Optional[str]]:
        """
        Process a single file through the pipeline.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Tuple of (chunks, error_message)
        """
        try:
            # Convert to Markdown
            conversion_result = self.converter.convert(filepath)
            
            if conversion_result.status != ConversionStatus.SUCCESS:
                error_msg = conversion_result.error_message or "Unknown conversion error"
                logger.error(f"Conversion failed for {filepath}: {error_msg}")
                return [], error_msg
            
            # Chunk the Markdown
            chunks = self.chunker.chunk(
                conversion_result.markdown,
                source_file=str(filepath)
            )
            
            if not chunks:
                logger.warning(f"No chunks created for {filepath}")
                return [], None
            
            logger.info(f"Created {len(chunks)} chunks for {filepath}")
            return chunks, None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Processing failed for {filepath}: {error_msg}")
            return [], error_msg
    
    def ingest(
        self,
        source_dir: str,
        force_reprocess: bool = False,
        delete_missing: bool = False
    ) -> IngestResult:
        """
        Ingest documents from a directory.
        
        Args:
            source_dir: Directory containing documents to ingest
            force_reprocess: Re-process all files even if unchanged
            delete_missing: Remove chunks for files no longer in source
            
        Returns:
            IngestResult with statistics
        """
        start_time = datetime.utcnow()
        run_id = self.tracker.start_ingest_run()
        
        result = IngestResult(run_id=run_id)
        
        try:
            # Scan for files
            self._report_progress("scanning", 0, 1)
            files = self._scan_directory(source_dir)
            
            if not files:
                logger.warning(f"No supported files found in {source_dir}")
                return result
            
            total_files = len(files)
            self._report_progress("processing", 0, total_files)
            
            # Track files to detect deletions
            processed_paths = set()
            
            for i, filepath in enumerate(files):
                str_path = str(filepath)
                processed_paths.add(str_path)
                
                # Compute file hash
                try:
                    file_hash = self.tracker.compute_hash(str_path)
                except Exception as e:
                    logger.error(f"Failed to hash {filepath}: {e}")
                    result.files_failed += 1
                    result.errors.append({"file": str_path, "error": str(e)})
                    continue
                
                # Check if update needed
                if not force_reprocess and not self.tracker.needs_update(str_path, file_hash):
                    logger.info(f"Skipping unchanged file: {filepath}")
                    result.files_skipped += 1
                    self._report_progress("processing", i + 1, total_files)
                    continue
                
                # Delete old chunks for this file before re-processing
                # (applies whether force=True or the file hash changed)
                old_record = self.tracker.get_record(str_path)
                if old_record and old_record.chunk_count > 0:
                    self.storage.delete_by_file(str_path)
                
                # Process file
                chunks, error_msg = self._process_file(filepath)
                
                if error_msg:
                    result.files_failed += 1
                    result.errors.append({"file": str_path, "error": error_msg})
                    self.tracker.record_file(
                        str_path, 
                        file_hash, 
                        chunk_count=0,
                        status='error',
                        error_message=error_msg
                    )
                else:
                    # Store chunks
                    if chunks:
                        self.storage.add_chunks(chunks, file_id=str_path)
                        result.total_chunks += len(chunks)
                    
                    result.files_processed += 1
                    self.tracker.record_file(
                        str_path,
                        file_hash,
                        chunk_count=len(chunks),
                        status='success'
                    )
                
                self._report_progress("processing", i + 1, total_files)
            
            # Handle deleted files
            if delete_missing:
                all_records = self.tracker.get_all_records()
                for record in all_records:
                    if record.filepath not in processed_paths:
                        logger.info(f"Deleting chunks for removed file: {record.filepath}")
                        self.storage.delete_by_file(record.filepath)
                        self.tracker.delete_record(record.filepath)
            
            # Complete run
            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration
            
            self.tracker.complete_ingest_run(
                run_id=run_id,
                files_processed=result.files_processed,
                files_failed=result.files_failed,
                total_chunks=result.total_chunks
            )
            
            logger.info(
                f"Ingest complete: {result.files_processed} processed, "
                f"{result.files_skipped} skipped, {result.files_failed} failed, "
                f"{result.total_chunks} chunks in {duration:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            raise
        
        return result
    
    def ingest_single(
        self,
        filepath: str,
        force_reprocess: bool = False
    ) -> IngestResult:
        """
        Ingest a single file.
        
        Args:
            filepath: Path to the file
            force_reprocess: Re-process even if unchanged
            
        Returns:
            IngestResult with statistics
        """
        start_time = datetime.utcnow()
        run_id = self.tracker.start_ingest_run()
        
        result = IngestResult(run_id=run_id)
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not self.converter.is_supported(filepath):
            raise ValueError(f"Unsupported file type: {filepath.suffix}")
        
        try:
            # Compute hash
            str_path = str(filepath)
            file_hash = self.tracker.compute_hash(str_path)
            
            # Check if update needed
            if not force_reprocess and not self.tracker.needs_update(str_path, file_hash):
                logger.info(f"File unchanged, skipping: {filepath}")
                result.files_skipped = 1
                return result
            
            # Delete old chunks
            self.storage.delete_by_file(str_path)
            
            # Process file
            chunks, error_msg = self._process_file(filepath)
            
            if error_msg:
                result.files_failed = 1
                result.errors.append({"file": str_path, "error": error_msg})
                self.tracker.record_file(
                    str_path,
                    file_hash,
                    chunk_count=0,
                    status='error',
                    error_message=error_msg
                )
            else:
                # Store chunks
                if chunks:
                    self.storage.add_chunks(chunks, file_id=str_path)
                    result.total_chunks = len(chunks)
                
                result.files_processed = 1
                self.tracker.record_file(
                    str_path,
                    file_hash,
                    chunk_count=len(chunks),
                    status='success'
                )
            
            # Complete run
            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration
            
            self.tracker.complete_ingest_run(
                run_id=run_id,
                files_processed=result.files_processed,
                files_failed=result.files_failed,
                total_chunks=result.total_chunks
            )
            
        except Exception as e:
            logger.error(f"Single file ingest failed: {e}")
            raise
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ingest pipeline.
        
        Returns:
            Dictionary with statistics
        """
        tracker_stats = self.tracker.get_stats()
        storage_stats = self.storage.get_collection_stats()
        
        return {
            'tracker': tracker_stats,
            'storage': storage_stats,
            'data_dir': str(self.data_dir)
        }
