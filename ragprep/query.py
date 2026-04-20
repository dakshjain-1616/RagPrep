"""
Query module for ChromaDB similarity search with formatted results.

Provides a high-level interface for querying the vector store and
formatting results for display or downstream use.
"""

import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ragprep.storage import StorageManager

logger = logging.getLogger(__name__)


class ResultFormat(Enum):
    """Format for query results."""
    SIMPLE = "simple"      # Just content and distance
    DETAILED = "detailed"  # Full metadata
    MARKDOWN = "markdown"  # Markdown formatted
    JSON = "json"          # JSON serializable


@dataclass
class QueryResult:
    """A single query result."""
    content: str
    distance: float
    score: float  # Normalized similarity score (1 - distance for cosine)
    source_file: Optional[str] = None
    heading_path: List[str] = None
    chunk_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.heading_path is None:
            self.heading_path = []
        if self.metadata is None:
            self.metadata = {}


class QueryEngine:
    """
    Query engine for searching the vector store.
    
    Provides similarity search with result formatting and filtering.
    
    Example:
        >>> engine = QueryEngine(storage_manager)
        >>> results = engine.search("what is machine learning?", n_results=5)
        >>> for result in results:
        ...     print(f"{result.score:.3f}: {result.content[:100]}...")
    """
    
    def __init__(
        self,
        storage: StorageManager,
        default_n_results: int = 5,
        min_score_threshold: Optional[float] = None
    ):
        """
        Initialize the query engine.
        
        Args:
            storage: StorageManager instance
            default_n_results: Default number of results to return
            min_score_threshold: Minimum similarity score (0-1) to include results
        """
        self.storage = storage
        self.default_n_results = default_n_results
        self.min_score_threshold = min_score_threshold
        
        logger.info(f"QueryEngine initialized: n_results={default_n_results}")
    
    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[QueryResult]:
        """
        Search for similar chunks.
        
        Args:
            query: The search query
            n_results: Number of results to return (default: self.default_n_results)
            filter_dict: Optional metadata filter
            min_score: Minimum similarity score threshold
            
        Returns:
            List of QueryResult objects
        """
        n_results = n_results or self.default_n_results
        min_score = min_score or self.min_score_threshold
        
        preview = query[:50] + ("..." if len(query) > 50 else "")
        logger.info(f"Searching: '{preview}' (n={n_results})")
        
        # Query storage
        raw_results = self.storage.query(
            query_text=query,
            n_results=n_results,
            filter_dict=filter_dict
        )
        
        # Convert to QueryResult objects
        results = []
        for raw in raw_results:
            distance = raw['distance']
            score = 1.0 - distance  # Convert distance to similarity score
            
            # Apply score threshold
            if min_score is not None and score < min_score:
                continue
            
            metadata = raw['metadata']
            
            result = QueryResult(
                content=raw['content'],
                distance=distance,
                score=score,
                source_file=metadata.get('source_file'),
                heading_path=metadata.get('heading_path', []),
                chunk_type=metadata.get('chunk_type'),
                metadata=metadata
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def search_with_context(
        self,
        query: str,
        n_results: int = 3,
        context_chunks: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search with surrounding context chunks.
        
        Args:
            query: The search query
            n_results: Number of main results
            context_chunks: Number of chunks to include before/after each result
            
        Returns:
            List of result dicts with 'main' and 'context' entries
        """
        # Get main results
        main_results = self.search(query, n_results=n_results)
        
        # Build results with context
        contextualized = []
        for result in main_results:
            entry = {
                'main': result,
                'context_before': [],
                'context_after': []
            }
            
            # Note: Getting actual context chunks would require additional
            # storage queries based on position metadata. For now, we return
            # the main result with empty context.
            # TODO: Implement context retrieval based on file_id and position
            
            contextualized.append(entry)
        
        return contextualized
    
    def format_results(
        self,
        results: List[QueryResult],
        format_type: ResultFormat = ResultFormat.SIMPLE,
        max_content_length: Optional[int] = None
    ) -> str:
        """
        Format results as a string.
        
        Args:
            results: List of QueryResult objects
            format_type: Format type
            max_content_length: Maximum content length per result
            
        Returns:
            Formatted string
        """
        if not results:
            return "No results found."
        
        if format_type == ResultFormat.SIMPLE:
            return self._format_simple(results, max_content_length)
        elif format_type == ResultFormat.DETAILED:
            return self._format_detailed(results, max_content_length)
        elif format_type == ResultFormat.MARKDOWN:
            return self._format_markdown(results, max_content_length)
        elif format_type == ResultFormat.JSON:
            return self._format_json(results, max_content_length)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _format_simple(
        self, 
        results: List[QueryResult],
        max_content_length: Optional[int] = None
    ) -> str:
        """Format results in simple format."""
        lines = []
        lines.append(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            content = result.content
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            lines.append(f"[{i}] Score: {result.score:.3f}")
            lines.append(f"    {content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_detailed(
        self, 
        results: List[QueryResult],
        max_content_length: Optional[int] = None
    ) -> str:
        """Format results with full details."""
        lines = []
        lines.append(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            content = result.content
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            lines.append(f"=" * 60)
            lines.append(f"Result {i}")
            lines.append(f"=" * 60)
            lines.append(f"Score:      {result.score:.4f}")
            lines.append(f"Distance:   {result.distance:.4f}")
            lines.append(f"Source:     {result.source_file or 'Unknown'}")
            lines.append(f"Type:       {result.chunk_type or 'Unknown'}")
            
            if result.heading_path:
                lines.append(f"Headings:   {' > '.join(result.heading_path)}")
            
            lines.append(f"\nContent:")
            lines.append(f"-" * 40)
            lines.append(content)
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_markdown(
        self, 
        results: List[QueryResult],
        max_content_length: Optional[int] = None
    ) -> str:
        """Format results as Markdown."""
        lines = []
        lines.append(f"## Search Results ({len(results)} found)\n")
        
        for i, result in enumerate(results, 1):
            content = result.content
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            lines.append(f"### Result {i} (Score: {result.score:.3f})")
            
            if result.source_file:
                lines.append(f"**Source:** `{result.source_file}`")
            
            if result.heading_path:
                lines.append(f"**Path:** {' > '.join(result.heading_path)}")
            
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_json(
        self, 
        results: List[QueryResult],
        max_content_length: Optional[int] = None
    ) -> str:
        """Format results as JSON string."""
        import json
        
        data = []
        for result in results:
            content = result.content
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            data.append({
                'content': content,
                'score': round(result.score, 4),
                'distance': round(result.distance, 4),
                'source_file': result.source_file,
                'heading_path': result.heading_path,
                'chunk_type': result.chunk_type,
                'metadata': result.metadata
            })
        
        return json.dumps(data, indent=2)
    
    def query_and_format(
        self,
        query: str,
        n_results: Optional[int] = None,
        format_type: ResultFormat = ResultFormat.SIMPLE,
        max_content_length: Optional[int] = 500
    ) -> str:
        """
        Search and format results in one call.
        
        Args:
            query: The search query
            n_results: Number of results
            format_type: Format type
            max_content_length: Maximum content length per result
            
        Returns:
            Formatted results string
        """
        results = self.search(query, n_results=n_results)
        return self.format_results(results, format_type, max_content_length)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get query engine statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'default_n_results': self.default_n_results,
            'min_score_threshold': self.min_score_threshold,
            'storage_stats': self.storage.get_collection_stats()
        }
