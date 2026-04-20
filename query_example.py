#!/usr/bin/env python3
"""
Query example for RAGPrep.

This script demonstrates how to query the indexed documents using
the RAGPrep library.

Usage:
    python query_example.py "your search query"

Example:
    python query_example.py "what is machine learning?"
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragprep.storage import StorageManager
from ragprep.query import QueryEngine, ResultFormat


def main():
    """Run the query example."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python query_example.py \"<your search query>\"")
        print("Example: python query_example.py \"what is machine learning?\"")
        print()
        print("Interactive mode: python query_example.py --interactive")
        sys.exit(1)
    
    # Configuration
    data_dir = "./ragprep_data_example"
    collection_name = "example_docs"
    
    # Check for interactive mode
    interactive = sys.argv[1] == "--interactive"
    
    print("=" * 60)
    print("RAGPrep Query Example")
    print("=" * 60)
    print()
    print(f"Data directory: {data_dir}")
    print(f"Collection: {collection_name}")
    print()
    
    # Initialize storage and query engine
    try:
        print("Initializing query engine...")
        chroma_path = os.path.join(data_dir, "chroma")
        
        if not os.path.exists(chroma_path):
            print(f"❌ Error: No index found at {chroma_path}")
            print("   Run basic_ingest.py first to create an index.")
            sys.exit(1)
        
        storage = StorageManager(
            chroma_path=chroma_path,
            collection_name=collection_name
        )
        
        engine = QueryEngine(
            storage=storage,
            default_n_results=5,
            min_score_threshold=None
        )
        
        stats = storage.get_collection_stats()
        print(f"✅ Connected to collection: {stats['collection_name']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Embedding model: {stats['embedding_model']}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if interactive:
        # Interactive mode
        print("Interactive query mode (type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            print()
            query = input("Query: ").strip()
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            try:
                results = engine.search(query, n_results=5)
                
                if not results:
                    print("No results found.")
                    continue
                
                print(f"\nFound {len(results)} results:\n")
                
                for i, result in enumerate(results, 1):
                    print(f"[{i}] Score: {result.score:.3f}")
                    if result.source_file:
                        print(f"    Source: {result.source_file}")
                    if result.heading_path:
                        print(f"    Path: {' > '.join(result.heading_path)}")
                    print(f"    Content: {result.content[:300]}...")
                    print()
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
    
    else:
        # Single query mode
        query_text = ' '.join(sys.argv[1:])
        
        print(f"Query: '{query_text}'")
        print("-" * 60)
        
        try:
            # Search with simple format
            print("\n--- Simple Format ---\n")
            simple_output = engine.query_and_format(
                query_text,
                n_results=5,
                format_type=ResultFormat.SIMPLE,
                max_content_length=300
            )
            print(simple_output)
            
            # Search with detailed format
            print("\n--- Detailed Format ---\n")
            detailed_output = engine.query_and_format(
                query_text,
                n_results=3,
                format_type=ResultFormat.DETAILED,
                max_content_length=500
            )
            print(detailed_output)
            
            # Search with markdown format
            print("\n--- Markdown Format ---\n")
            markdown_output = engine.query_and_format(
                query_text,
                n_results=3,
                format_type=ResultFormat.MARKDOWN,
                max_content_length=400
            )
            print(markdown_output)
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
