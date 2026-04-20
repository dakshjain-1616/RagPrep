#!/usr/bin/env python3
"""
Basic ingestion example for RAGPrep.

This script demonstrates how to use the RAGPrep library to ingest documents
from a directory into a ChromaDB vector store.

Usage:
    python basic_ingest.py <source_directory>

Example:
    python basic_ingest.py ./my_documents/
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragprep.ingest import IngestPipeline


def main():
    """Run the basic ingestion example."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python basic_ingest.py <source_directory>")
        print("Example: python basic_ingest.py ./documents/")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Directory not found: {source_dir}")
        sys.exit(1)
    
    if not os.path.isdir(source_dir):
        print(f"Error: Not a directory: {source_dir}")
        sys.exit(1)
    
    # Configuration
    data_dir = "./ragprep_data_example"
    collection_name = "example_docs"
    
    print("=" * 60)
    print("RAGPrep Basic Ingestion Example")
    print("=" * 60)
    print()
    print(f"Source directory: {source_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Collection: {collection_name}")
    print()
    
    # Create the ingest pipeline
    print("Initializing pipeline...")
    pipeline = IngestPipeline(
        data_dir=data_dir,
        collection_name=collection_name,
        embedding_model="all-MiniLM-L6-v2",  # Lightweight, fast model
        chunk_size=500,
        chunk_overlap=50,
        progress_callback=lambda status, current, total: (
            print(f"  {status}: {current}/{total}") if status == "processing" else None
        )
    )
    
    # Run the ingest
    print("Starting ingest...")
    print("-" * 60)
    
    try:
        result = pipeline.ingest(
            source_dir=source_dir,
            force_reprocess=False,  # Skip unchanged files
            delete_missing=False     # Keep chunks for removed files
        )
        
        print("-" * 60)
        print()
        print("✅ Ingest complete!")
        print()
        print("Results:")
        print(f"  Files processed: {result.files_processed}")
        print(f"  Files skipped: {result.files_skipped}")
        print(f"  Files failed: {result.files_failed}")
        print(f"  Total chunks: {result.total_chunks}")
        print(f"  Duration: {result.duration_seconds:.2f} seconds")
        print()
        
        # Show stats
        stats = pipeline.get_stats()
        print("Storage stats:")
        print(f"  Collection: {stats['storage']['collection_name']}")
        print(f"  Total chunks: {stats['storage']['total_chunks']}")
        print(f"  Embedding model: {stats['storage']['embedding_model']}")
        print(f"  Embedding dimension: {stats['storage']['embedding_dimension']}")
        print()
        
        if result.errors:
            print("⚠️  Errors encountered:")
            for error in result.errors[:5]:
                print(f"  - {error['file']}: {error['error'][:100]}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")
            print()
        
        print("Next steps:")
        print(f"  - Query the index: python query_example.py")
        print(f"  - Start server: ragprep serve")
        print()
        
    except Exception as e:
        print(f"❌ Ingest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
