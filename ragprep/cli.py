"""
Command-line interface for RAGPrep.

Provides CLI commands for:
- ingest: Convert and index documents
- query: Search the vector store
- serve: Start the FastAPI server
- status: Show statistics
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import click

from ragprep.ingest import IngestPipeline, IngestResult
from ragprep.query import QueryEngine, ResultFormat
from ragprep.storage import StorageManager, HashTracker

# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Default paths
DEFAULT_DATA_DIR = "./ragprep_data"
DEFAULT_COLLECTION = "ragprep_docs"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--data-dir', '-d', default=DEFAULT_DATA_DIR, 
              help=f'Directory for data storage (default: {DEFAULT_DATA_DIR})')
@click.pass_context
def cli(ctx, verbose: bool, data_dir: str):
    """
    RAGPrep - Convert documents to a queryable vector store.
    
    A RAG-prep pipeline built on top of microsoft/markitdown that converts
    any folder of mixed-format documents into a queryable ChromaDB vector store.
    
    Commands:
        ingest   Convert and index documents from a directory
        query    Search the indexed documents
        serve    Start the FastAPI server
        status   Show statistics about indexed documents
    
    Examples:
        ragprep ingest ./documents/
        ragprep query "what is machine learning?"
        ragprep serve --port 8000
        ragprep status
    """
    setup_logging(verbose)
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['data_dir'] = data_dir


@cli.command()
@click.argument('source_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--collection', '-c', default=DEFAULT_COLLECTION,
              help=f'Collection name (default: {DEFAULT_COLLECTION})')
@click.option('--model', '-m', default=DEFAULT_MODEL,
              help=f'Embedding model (default: {DEFAULT_MODEL})')
@click.option('--chunk-size', '-s', default=500, type=int,
              help='Target chunk size in characters (default: 500)')
@click.option('--chunk-overlap', '-o', default=50, type=int,
              help='Chunk overlap in characters (default: 50)')
@click.option('--force', '-f', is_flag=True,
              help='Force re-processing of all files')
@click.option('--delete-missing', is_flag=True,
              help='Remove chunks for files no longer in source')
@click.option('--progress/--no-progress', default=True,
              help='Show progress bar (default: True)')
@click.pass_context
def ingest(
    ctx,
    source_dir: str,
    collection: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    force: bool,
    delete_missing: bool,
    progress: bool
):
    """
    Ingest documents from SOURCE_DIR into the vector store.
    
    Converts all supported files (PDF, DOCX, PPTX, XLSX, images, audio) to Markdown,
    chunks them intelligently, embeds using sentence-transformers, and stores in ChromaDB.
    
    Only re-processes files that have changed since last ingest (based on file hash).
    
    Examples:
        ragprep ingest ./documents/
        ragprep ingest ./docs/ --collection my_docs --model all-mpnet-base-v2
        ragprep ingest ./docs/ --force --delete-missing
    """
    data_dir = ctx.obj['data_dir']
    
    click.echo(f"🚀 Starting ingest from: {source_dir}")
    click.echo(f"   Data directory: {data_dir}")
    click.echo(f"   Collection: {collection}")
    click.echo(f"   Model: {model}")
    click.echo(f"   Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    
    if force:
        click.echo("   Force mode: Re-processing all files")
    
    try:
        # Setup progress callback
        progress_callback = None
        if progress:
            def show_progress(status: str, current: int, total: int):
                if status == "scanning":
                    click.echo(f"📁 Scanning for files...")
                elif status == "processing":
                    if total > 0:
                        pct = (current / total) * 100
                        click.echo(f"   Processing {current}/{total} ({pct:.1f}%)", nl=False)
                        click.echo('\r', nl=False)
            progress_callback = show_progress
        
        # Create pipeline
        pipeline = IngestPipeline(
            data_dir=data_dir,
            collection_name=collection,
            embedding_model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback
        )
        
        # Run ingest
        result = pipeline.ingest(
            source_dir=source_dir,
            force_reprocess=force,
            delete_missing=delete_missing
        )
        
        if progress:
            click.echo()  # Newline after progress
        
        # Report results
        click.echo()
        click.echo("✅ Ingest complete!")
        click.echo(f"   Files processed: {result.files_processed}")
        click.echo(f"   Files skipped: {result.files_skipped}")
        click.echo(f"   Files failed: {result.files_failed}")
        click.echo(f"   Total chunks: {result.total_chunks}")
        click.echo(f"   Duration: {result.duration_seconds:.2f}s")
        
        if result.errors:
            click.echo()
            click.echo("⚠️  Errors:")
            for error in result.errors[:5]:  # Show first 5 errors
                click.echo(f"   - {error['file']}: {error['error'][:100]}")
            if len(result.errors) > 5:
                click.echo(f"   ... and {len(result.errors) - 5} more")
        
        # Exit with error code if any files failed
        if result.files_failed > 0:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Ingest failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('query_text', nargs=-1, required=True)
@click.option('--collection', '-c', default=DEFAULT_COLLECTION,
              help=f'Collection name (default: {DEFAULT_COLLECTION})')
@click.option('--n-results', '-n', default=5, type=int,
              help='Number of results (default: 5)')
@click.option('--format', '-f', 'format_type', 
              type=click.Choice(['simple', 'detailed', 'markdown', 'json']),
              default='simple',
              help='Output format (default: simple)')
@click.option('--min-score', type=float, default=None,
              help='Minimum similarity score threshold (0-1)')
@click.option('--max-length', '-l', default=500, type=int,
              help='Maximum content length per result (default: 500)')
@click.pass_context
def query(
    ctx,
    query_text: tuple,
    collection: str,
    n_results: int,
    format_type: str,
    min_score: Optional[float],
    max_length: int
):
    """
    Search the indexed documents.
    
    Performs similarity search against the vector store and returns matching chunks.
    
    Examples:
        ragprep query "what is machine learning?"
        ragprep query "API documentation" -n 10 --format detailed
        ragprep query "python examples" --min-score 0.7
    """
    data_dir = ctx.obj['data_dir']
    query_str = ' '.join(query_text)
    
    try:
        # Initialize storage and query engine
        storage = StorageManager(
            chroma_path=os.path.join(data_dir, "chroma"),
            collection_name=collection
        )
        
        engine = QueryEngine(
            storage=storage,
            default_n_results=n_results,
            min_score_threshold=min_score
        )
        
        # Search
        click.echo(f"🔍 Searching: '{query_str}'")
        click.echo(f"   Collection: {collection}")
        click.echo()
        
        results = engine.search(query_str, n_results=n_results)
        
        if not results:
            click.echo("No results found.")
            return
        
        # Format and display
        format_enum = ResultFormat(format_type)
        formatted = engine.format_results(results, format_enum, max_length)
        click.echo(formatted)
        
    except Exception as e:
        click.echo(f"❌ Query failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
@click.option('--port', '-p', default=8000, type=int, help='Port to bind to (default: 8000)')
@click.option('--collection', '-c', default=DEFAULT_COLLECTION,
              help=f'Collection name (default: {DEFAULT_COLLECTION})')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development)')
@click.pass_context
def serve(ctx, host: str, port: int, collection: str, reload: bool):
    """
    Start the FastAPI server with OpenAI-compatible /v1/embeddings endpoint.
    
    The server provides an OpenAI-compatible API that can be used with any RAG
    application without code changes.
    
    Endpoints:
        GET  /health          Health check
        POST /v1/embeddings   OpenAI-compatible embeddings endpoint
        POST /v1/search       Search endpoint with metadata
    
    Examples:
        ragprep serve
        ragprep serve --port 8080
        ragprep serve --host 127.0.0.1 --port 8000
    """
    data_dir = ctx.obj['data_dir']
    
    try:
        import uvicorn
        from ragprep.server import create_app
        
        # Create app with configuration
        app = create_app(
            data_dir=data_dir,
            collection_name=collection
        )
        
        click.echo(f"🚀 Starting server on http://{host}:{port}")
        click.echo(f"   Data directory: {data_dir}")
        click.echo(f"   Collection: {collection}")
        click.echo()
        click.echo("Endpoints:")
        click.echo(f"   Health:  http://{host}:{port}/health")
        click.echo(f"   Embeddings: POST http://{host}:{port}/v1/embeddings")
        click.echo(f"   Search: POST http://{host}:{port}/v1/search")
        click.echo()
        click.echo("Press Ctrl+C to stop")
        
        # Run server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info" if ctx.obj['verbose'] else "warning"
        )
        
    except ImportError as e:
        click.echo(f"❌ Server dependencies not installed: {e}", err=True)
        click.echo("Install with: pip install fastapi uvicorn", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Server failed: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--collection', '-c', default=DEFAULT_COLLECTION,
              help=f'Collection name (default: {DEFAULT_COLLECTION})')
@click.pass_context
def status(ctx, collection: str):
    """
    Show statistics about indexed documents.
    
    Displays information about:
    - Total files processed
    - Total chunks indexed
    - Storage location
    - Embedding model used
    
    Examples:
        ragprep status
        ragprep status --collection my_docs
    """
    data_dir = ctx.obj['data_dir']
    
    try:
        # Get tracker stats
        db_path = os.path.join(data_dir, "ragprep.db")
        if os.path.exists(db_path):
            tracker = HashTracker(db_path)
            tracker_stats = tracker.get_stats()
        else:
            tracker_stats = {
                'total_files': 0,
                'success_count': 0,
                'error_count': 0,
                'total_chunks': 0
            }
        
        # Get storage stats
        chroma_path = os.path.join(data_dir, "chroma")
        if os.path.exists(chroma_path):
            storage = StorageManager(
                chroma_path=chroma_path,
                collection_name=collection
            )
            storage_stats = storage.get_collection_stats()
        else:
            storage_stats = {
                'collection_name': collection,
                'total_chunks': 0,
                'embedding_model': DEFAULT_MODEL,
                'embedding_dimension': 384
            }
        
        # Display stats
        click.echo("📊 RAGPrep Status")
        click.echo("=" * 50)
        click.echo()
        click.echo("Storage:")
        click.echo(f"   Data directory: {data_dir}")
        click.echo(f"   ChromaDB path: {chroma_path}")
        click.echo(f"   SQLite path: {db_path}")
        click.echo()
        click.echo("Collection:")
        click.echo(f"   Name: {storage_stats['collection_name']}")
        click.echo(f"   Total chunks: {storage_stats['total_chunks']}")
        click.echo(f"   Embedding model: {storage_stats['embedding_model']}")
        click.echo(f"   Embedding dimension: {storage_stats['embedding_dimension']}")
        click.echo()
        click.echo("Files:")
        click.echo(f"   Total tracked: {tracker_stats['total_files']}")
        click.echo(f"   Successfully processed: {tracker_stats['success_count']}")
        click.echo(f"   Failed: {tracker_stats['error_count']}")
        
        if tracker_stats['total_files'] > 0:
            success_rate = (tracker_stats['success_count'] / tracker_stats['total_files']) * 100
            click.echo(f"   Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        click.echo(f"❌ Failed to get status: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
