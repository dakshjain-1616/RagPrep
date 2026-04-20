# RAGPrep

> Made Autonomously Using [NEO - Your Autonomous AI Engineering Agent](https://heyneo.com)
>
> [![VS Code Extension](https://img.shields.io/badge/VS%20Code-NEO%20Extension-blue?logo=visualstudiocode)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)  [![Cursor Extension](https://img.shields.io/badge/Cursor-NEO%20Extension-purple?logo=cursor)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)

## Architecture

![Architecture](architecture.svg)

A full RAG-prep pipeline built on top of `microsoft/markitdown` that converts any folder of mixed-format documents into a queryable ChromaDB vector store in one CLI command.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **📄 Universal Document Support**: PDF, DOCX, PPTX, XLSX, images (with OCR), audio (transcription), HTML, and text files
- **🧠 Smart Chunking**: Heading-aware splitting, table-preserving chunking, sliding window with overlap
- **🔒 Local Embeddings**: Uses `sentence-transformers` - no OpenAI key needed
- **💾 Incremental Updates**: Only re-processes files that changed (file hash tracking)
- **⚡ FastAPI Server**: OpenAI-compatible `/v1/embeddings` endpoint
- **🔍 Semantic Search**: Cosine similarity search with metadata filtering

## Installation

```bash
# Clone the repository
git clone https://github.com/dakshjain-1616/ragprep.git
cd ragprep

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies

## Quick Start

### 1. Ingest Documents

```bash
# Ingest a directory of documents
ragprep ingest ./my_documents/

# With options
ragprep ingest ./docs/ --collection my_docs --model all-mpnet-base-v2
```

### 2. Query Documents

```bash
# Search the indexed documents
ragprep query "what is machine learning?"

# With options
ragprep query "API documentation" -n 10 --format detailed
```

### 3. Start Server

```bash
# Start the FastAPI server
ragprep serve

# With custom port
ragprep serve --port 8080
```

### 4. Check Status

```bash
# Show statistics
ragprep status
```

## CLI Reference

### `ragprep ingest`

Convert and index documents from a directory.

```bash
ragprep ingest [OPTIONS] SOURCE_DIR

Options:
  -c, --collection TEXT    Collection name [default: ragprep_docs]
  -m, --model TEXT         Embedding model [default: all-MiniLM-L6-v2]
  -s, --chunk-size INTEGER Target chunk size [default: 500]
  -o, --chunk-overlap INTEGER Chunk overlap [default: 50]
  -f, --force              Force re-processing of all files
  --delete-missing         Remove chunks for files no longer in source
  --progress / --no-progress Show progress bar [default: True]
  --help                   Show help
```

### `ragprep query`

Search the indexed documents.

```bash
ragprep query [OPTIONS] QUERY_TEXT

Options:
  -c, --collection TEXT    Collection name [default: ragprep_docs]
  -n, --n-results INTEGER  Number of results [default: 5]
  -f, --format [simple|detailed|markdown|json] Output format [default: simple]
  --min-score FLOAT        Minimum similarity score threshold (0-1)
  -l, --max-length INTEGER Maximum content length per result [default: 500]
  --help                   Show help
```

### `ragprep serve`

Start the FastAPI server.

```bash
ragprep serve [OPTIONS]

Options:
  --host TEXT              Host to bind to [default: 0.0.0.0]
  -p, --port INTEGER       Port to bind to [default: 8000]
  -c, --collection TEXT    Collection name [default: ragprep_docs]
  --reload                 Enable auto-reload (development)
  --help                   Show help
```

### `ragprep status`

Show statistics about indexed documents.

```bash
ragprep status [OPTIONS]

Options:
  -c, --collection TEXT    Collection name [default: ragprep_docs]
  --help                   Show help
```

## API Reference

### Server Endpoints

#### Health Check

```bash
GET /health
```

Returns server status and collection statistics.

#### Embeddings (OpenAI-compatible)

```bash
POST /v1/embeddings
Content-Type: application/json

{
  "input": "text to embed",
  "model": "ragprep-default"
}
```

Compatible with OpenAI's `/v1/embeddings` API.

#### Search

```bash
POST /v1/search
Content-Type: application/json

{
  "query": "search query",
  "n_results": 5,
  "min_score": 0.7,
  "filter": {"chunk_type": "paragraph"}
}
```

## Python API

### Basic Usage

```python
from ragprep.ingest import IngestPipeline
from ragprep.query import QueryEngine
from ragprep.storage import StorageManager

# Create pipeline
pipeline = IngestPipeline(
    data_dir="./data",
    collection_name="my_docs"
)

# Ingest documents
result = pipeline.ingest("./documents/")
print(f"Created {result.total_chunks} chunks")

# Query
storage = StorageManager(
    chroma_path="./data/chroma",
    collection_name="my_docs"
)
engine = QueryEngine(storage)
results = engine.search("machine learning", n_results=5)

for result in results:
    print(f"{result.score:.3f}: {result.content[:100]}...")
```

### Advanced Usage

```python
from ragprep.converter import DocumentConverter
from ragprep.chunker import MarkdownChunker
from ragprep.storage import StorageManager, HashTracker

# Convert single file
converter = DocumentConverter()
result = converter.convert("document.pdf")
print(result.markdown)

# Chunk with custom settings
chunker = MarkdownChunker(
    chunk_size=1000,
    chunk_overlap=100,
    preserve_tables=True
)
chunks = chunker.chunk(markdown_text)

# Track file hashes
tracker = HashTracker("./data/ragprep.db")
if tracker.needs_update("document.pdf"):
    # Process file...
    tracker.record_file("document.pdf", file_hash, chunk_count=10)
```

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Source Files   │────▶│  Converter   │────▶│  Markdown   │
│ (PDF,DOCX,etc)  │     │ (markitdown) │     │    Text     │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                    │
                       ┌────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │     Chunker     │
              │  (heading-aware,│
              │ table-preserving)│
              └────────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│   SQLite   │  │ ChromaDB   │  │Embeddings  │
│(file hashes)│  │ (vectors)  │  │(sentence-  │
└────────────┘  └────────────┘  │transformers)│
                                └────────────┘
```

## Smart Chunking

RAGPrep uses intelligent chunking strategies:

1. **Heading-Aware**: Never cuts between a heading and its content
2. **Table-Preserving**: Never splits a Markdown table mid-row
3. **Sliding Window**: Configurable overlap for dense prose

```python
chunker = MarkdownChunker(
    chunk_size=500,      # Target chunk size in characters
    chunk_overlap=50,    # Overlap between chunks
    min_chunk_size=100   # Minimum chunk size
)
```

## Configuration

### Environment Variables

```bash
RAGPREP_DATA_DIR=./data          # Data directory
RAGPREP_COLLECTION=ragprep_docs  # Default collection
RAGPREP_HOST=0.0.0.0            # Server host
RAGPREP_PORT=8000               # Server port
```

### Embedding Models

Default: `all-MiniLM-L6-v2` (384 dimensions, fast)

Alternatives:
- `all-mpnet-base-v2` (768 dimensions, higher quality)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for semantic search)

## Examples

Runnable example scripts are included at the project root:

- `basic_ingest.py` - Ingest documents programmatically
- `query_example.py` - Query with different output formats

```bash
# Run examples
python basic_ingest.py ./documents/
python query_example.py "your query"
python query_example.py --interactive
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black ragprep/
ruff check ragprep/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

- [microsoft/markitdown](https://github.com/microsoft/markitdown) - Document conversion
- [sentence-transformers](https://www.sbert.net/) - Local embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector storage
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
