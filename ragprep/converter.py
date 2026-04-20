"""
Document converter module that wraps microsoft/markitdown.

Provides a robust interface for converting various document formats (PDF, DOCX, PPTX, 
XLSX, images, audio) to clean Markdown text.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

logger = logging.getLogger(__name__)


class ConversionStatus(Enum):
    """Status of document conversion."""
    SUCCESS = "success"
    ERROR = "error"
    UNSUPPORTED = "unsupported"
    NOT_FOUND = "not_found"


@dataclass
class ConversionResult:
    """Result of a document conversion operation."""
    filepath: str
    markdown: str
    status: ConversionStatus
    error_message: Optional[str] = None
    metadata: Optional[dict] = None


class DocumentConverter:
    """
    Converts various document formats to Markdown using markitdown.
    
    Supports: PDF, DOCX, PPTX, XLSX, images (with OCR), audio (transcription),
    HTML, and various text formats.
    
    Example:
        >>> converter = DocumentConverter()
        >>> result = converter.convert("document.pdf")
        >>> print(result.markdown)
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.pptx', '.xlsx', '.xls',
        '.html', '.htm', '.txt', '.md', '.markdown',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
        '.mp3', '.wav', '.mp4', '.m4a', '.flac', '.ogg', '.wma',
        '.zip',  # Some zip files contain documents
    }
    
    def __init__(self, enable_ocr: bool = True, enable_audio: bool = True):
        """
        Initialize the document converter.
        
        Args:
            enable_ocr: Whether to enable OCR for images (requires tesseract)
            enable_audio: Whether to enable audio transcription
        """
        if MarkItDown is None:
            raise ImportError(
                "markitdown is required but not installed. "
                "Install with: pip install markitdown"
            )
        
        self._md = MarkItDown()
        self.enable_ocr = enable_ocr
        self.enable_audio = enable_audio
        logger.info("DocumentConverter initialized")
    
    def convert(self, filepath: Union[str, Path]) -> ConversionResult:
        """
        Convert a single document to Markdown.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            ConversionResult with markdown content and status
        """
        filepath = Path(filepath)
        
        # Check if file exists
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return ConversionResult(
                filepath=str(filepath),
                markdown="",
                status=ConversionStatus.NOT_FOUND,
                error_message=f"File not found: {filepath}"
            )
        
        # Check if file is a directory
        if filepath.is_dir():
            logger.error(f"Path is a directory, not a file: {filepath}")
            return ConversionResult(
                filepath=str(filepath),
                markdown="",
                status=ConversionStatus.ERROR,
                error_message=f"Path is a directory: {filepath}"
            )
        
        # Check extension
        ext = filepath.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file extension: {ext}")
            return ConversionResult(
                filepath=str(filepath),
                markdown="",
                status=ConversionStatus.UNSUPPORTED,
                error_message=f"Unsupported file extension: {ext}"
            )
        
        # Perform conversion
        try:
            logger.info(f"Converting: {filepath}")
            result = self._md.convert(str(filepath))
            
            # Extract metadata if available
            metadata = {}
            if hasattr(result, 'metadata') and result.metadata:
                metadata = dict(result.metadata)
            
            return ConversionResult(
                filepath=str(filepath),
                markdown=result.text_content,
                status=ConversionStatus.SUCCESS,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Conversion failed for {filepath}: {e}")
            return ConversionResult(
                filepath=str(filepath),
                markdown="",
                status=ConversionStatus.ERROR,
                error_message=str(e)
            )
    
    def convert_batch(
        self, 
        filepaths: list[Union[str, Path]],
        progress_callback: Optional[callable] = None
    ) -> list[ConversionResult]:
        """
        Convert multiple documents to Markdown.
        
        Args:
            filepaths: List of paths to document files
            progress_callback: Optional callback function(current, total)
            
        Returns:
            List of ConversionResult objects
        """
        results = []
        total = len(filepaths)
        
        for i, filepath in enumerate(filepaths):
            result = self.convert(filepath)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def is_supported(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            filepath: Path to check
            
        Returns:
            True if the file extension is supported
        """
        filepath = Path(filepath)
        return filepath.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def get_supported_extensions(self) -> set[str]:
        """
        Get the set of supported file extensions.
        
        Returns:
            Set of supported extensions (e.g., {'.pdf', '.docx', ...})
        """
        return self.SUPPORTED_EXTENSIONS.copy()
