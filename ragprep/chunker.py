"""
Smart Markdown chunking module with heading-aware, table-preserving, and sliding window strategies.

This module provides intelligent text chunking that:
- Respects Markdown headings (never cuts mid-section)
- Preserves table integrity (never splits mid-row)
- Uses sliding window with overlap for dense prose
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Iterator
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Type of content chunk."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    LIST = "list"
    MIXED = "mixed"


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    chunk_type: ChunkType
    start_pos: int
    end_pos: int
    heading_path: List[str]
    metadata: dict
    
    def __post_init__(self):
        """Clean up content after initialization."""
        self.content = self.content.strip()
    
    @property
    def word_count(self) -> int:
        """Return approximate word count."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Return character count."""
        return len(self.content)


class MarkdownChunker:
    """
    Intelligent Markdown chunker with multiple strategies.
    
    Features:
    - Heading-aware splitting: Never cuts between a heading and its content
    - Table-preserving: Never splits a Markdown table mid-row
    - Sliding window: Configurable overlap for dense prose
    
    Example:
        >>> chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.chunk("# Title\\n\\nContent...")
        >>> for chunk in chunks:
        ...     print(f"{chunk.heading_path}: {chunk.word_count} words")
    """
    
    # Regex patterns for Markdown elements
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    TABLE_PATTERN = re.compile(
        r'(?:\|[^\n]*\|(?:\n|$))+'  # Match table rows with pipes
        r'(?:\|[-:]+[-|:\s]*\|(?:\n|$))?'  # Optional separator row
        r'(?:\|[^\n]*\|(?:\n|$))*',  # Additional data rows
        re.MULTILINE
    )
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
    LIST_PATTERN = re.compile(r'^(?:[-*+]|\d+\.)\s+', re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        preserve_tables: bool = True,
        preserve_headings: bool = True,
        respect_section_boundaries: bool = True
    ):
        """
        Initialize the Markdown chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size in characters
            preserve_tables: Whether to keep tables intact (don't split mid-row)
            preserve_headings: Whether to keep headings with their content
            respect_section_boundaries: Whether to avoid splitting at section boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.preserve_tables = preserve_tables
        self.preserve_headings = preserve_headings
        self.respect_section_boundaries = respect_section_boundaries
        
        logger.info(
            f"MarkdownChunker initialized: size={chunk_size}, "
            f"overlap={chunk_overlap}, min_size={min_chunk_size}"
        )
    
    def chunk(self, markdown_text: str, source_file: Optional[str] = None) -> List[TextChunk]:
        """
        Chunk Markdown text intelligently.
        
        Args:
            markdown_text: The Markdown content to chunk
            source_file: Optional source file path for metadata
            
        Returns:
            List of TextChunk objects
        """
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty markdown text provided")
            return []
        
        # First pass: identify special elements (tables, code blocks)
        protected_spans = self._identify_protected_spans(markdown_text)
        
        # Second pass: split by headings to get sections
        sections = self._split_by_headings(markdown_text)
        
        # Third pass: chunk each section
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, protected_spans, source_file)
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(markdown_text)} characters")
        return chunks
    
    def _identify_protected_spans(self, text: str) -> List[tuple]:
        """
        Identify spans that should not be split (tables, code blocks).
        
        Args:
            text: The Markdown text
            
        Returns:
            List of (start, end) tuples representing protected spans
        """
        spans = []
        
        if self.preserve_tables:
            for match in self.TABLE_PATTERN.finditer(text):
                # Verify it's actually a table (has at least 2 rows with pipes)
                table_text = match.group()
                lines = table_text.strip().split('\n')
                if len(lines) >= 2 and all('|' in line for line in lines[:2]):
                    spans.append((match.start(), match.end()))
        
        # Code blocks are always protected
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            spans.append((match.start(), match.end()))
        
        # Sort by start position
        spans.sort(key=lambda x: x[0])
        
        # Merge overlapping spans
        merged = []
        for span in spans:
            if merged and span[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
            else:
                merged.append(span)
        
        return merged
    
    def _split_by_headings(self, text: str) -> List[dict]:
        """
        Split text into sections based on headings.
        
        Args:
            text: The Markdown text
            
        Returns:
            List of section dicts with content and heading info
        """
        sections = []
        current_heading_path = []
        last_pos = 0
        
        for match in self.HEADING_PATTERN.finditer(text):
            # Save content before this heading
            if match.start() > last_pos:
                content = text[last_pos:match.start()].strip()
                if content:
                    sections.append({
                        'content': content,
                        'heading_path': current_heading_path.copy(),
                        'start_pos': last_pos,
                        'end_pos': match.start()
                    })
            
            # Update heading path
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            
            # Truncate path to current level
            current_heading_path = current_heading_path[:level-1]
            current_heading_path.append(heading_text)
            
            last_pos = match.start()
        
        # Add final section
        if last_pos < len(text):
            content = text[last_pos:].strip()
            if content:
                sections.append({
                    'content': content,
                    'heading_path': current_heading_path.copy(),
                    'start_pos': last_pos,
                    'end_pos': len(text)
                })
        
        # If no sections found, treat entire text as one section
        if not sections and text.strip():
            sections.append({
                'content': text.strip(),
                'heading_path': [],
                'start_pos': 0,
                'end_pos': len(text)
            })
        
        return sections
    
    def _chunk_section(
        self, 
        section: dict, 
        protected_spans: List[tuple],
        source_file: Optional[str]
    ) -> List[TextChunk]:
        """
        Chunk a single section while respecting protected spans.
        
        Args:
            section: Section dict with content and heading info
            protected_spans: List of spans that cannot be split
            source_file: Source file path for metadata
            
        Returns:
            List of TextChunk objects
        """
        content = section['content']
        heading_path = section['heading_path']
        base_pos = section['start_pos']
        
        # If content is small enough, return as single chunk
        if len(content) <= self.chunk_size:
            return [TextChunk(
                content=content,
                chunk_type=self._detect_chunk_type(content),
                start_pos=base_pos,
                end_pos=base_pos + len(content),
                heading_path=heading_path,
                metadata={'source_file': source_file}
            )]
        
        chunks = []
        pos = 0
        
        while pos < len(content):
            # Determine chunk end position
            chunk_end = min(pos + self.chunk_size, len(content))
            
            # Check if we're inside a protected span
            protected_end = self._get_protected_end(pos, chunk_end, protected_spans, base_pos)
            if protected_end > chunk_end:
                chunk_end = protected_end
            
            # Try to find a good break point (end of paragraph, sentence, or word)
            if chunk_end < len(content):
                chunk_end = self._find_break_point(content, chunk_end)
            
            # Extract chunk content
            chunk_content = content[pos:chunk_end].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    content=chunk_content,
                    chunk_type=self._detect_chunk_type(chunk_content),
                    start_pos=base_pos + pos,
                    end_pos=base_pos + chunk_end,
                    heading_path=heading_path,
                    metadata={'source_file': source_file}
                ))
            
            # Move position with overlap
            pos = chunk_end - self.chunk_overlap
            if pos >= chunk_end:
                pos = chunk_end
        
        return chunks
    
    def _get_protected_end(
        self, 
        start: int, 
        end: int, 
        protected_spans: List[tuple],
        base_pos: int
    ) -> int:
        """
        Check if a range overlaps with protected spans and extend end if needed.
        
        Args:
            start: Start position in content
            end: Proposed end position
            protected_spans: List of protected (start, end) tuples
            base_pos: Base position offset
            
        Returns:
            Extended end position if needed
        """
        abs_start = base_pos + start
        abs_end = base_pos + end
        
        for span_start, span_end in protected_spans:
            # Check if we're cutting through a protected span
            if abs_start < span_end and abs_end > span_start and abs_end < span_end:
                # Extend to end of protected span
                return span_end - base_pos
        
        return end
    
    def _find_break_point(self, text: str, pos: int) -> int:
        """
        Find a good break point near the target position.
        
        Looks for: paragraph break, sentence end, or word boundary.
        
        Args:
            text: The text to search
            pos: Target position
            
        Returns:
            Best break point position
        """
        max_search = min(100, self.chunk_overlap)
        
        # Look for paragraph break (double newline)
        for i in range(pos, min(pos + max_search, len(text) - 1)):
            if text[i:i+2] == '\n\n':
                return i + 2
        
        # Look for sentence end
        for i in range(pos, min(pos + max_search, len(text))):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] in ' \n':
                return i + 1
        
        # Look for word boundary
        for i in range(pos, min(pos + max_search, len(text))):
            if text[i] in ' \n\t':
                return i + 1
        
        # Fall back to original position
        return pos
    
    def _detect_chunk_type(self, content: str) -> ChunkType:
        """
        Detect the type of content in a chunk.
        
        Args:
            content: The chunk content
            
        Returns:
            ChunkType enum value
        """
        content = content.strip()
        
        # Check for table
        if self.TABLE_PATTERN.match(content) and '|' in content:
            lines = content.split('\n')
            if len(lines) >= 2 and all('|' in line for line in lines[:2]):
                return ChunkType.TABLE
        
        # Check for code block
        if content.startswith('```') or self.CODE_BLOCK_PATTERN.match(content):
            return ChunkType.CODE
        
        # Check for heading
        if self.HEADING_PATTERN.match(content):
            return ChunkType.HEADING
        
        # Check for list
        if self.LIST_PATTERN.match(content):
            return ChunkType.LIST
        
        # Check if it's a single paragraph
        if '\n\n' not in content:
            return ChunkType.PARAGRAPH
        
        return ChunkType.MIXED
    
    def chunk_stream(
        self, 
        markdown_text: str, 
        source_file: Optional[str] = None
    ) -> Iterator[TextChunk]:
        """
        Chunk Markdown text as a stream (generator).
        
        Args:
            markdown_text: The Markdown content to chunk
            source_file: Optional source file path for metadata
            
        Yields:
            TextChunk objects one at a time
        """
        chunks = self.chunk(markdown_text, source_file)
        yield from chunks
