"""
FastAPI server with OpenAI-compatible /v1/embeddings endpoint.

Provides a REST API for:
- Health checks
- OpenAI-compatible embeddings generation
- Document search with metadata
"""

import os
import logging
from typing import List, Optional, Dict, Any, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ragprep.storage import StorageManager
from ragprep.query import QueryEngine

logger = logging.getLogger(__name__)

# Global storage and query engine instances
_storage: Optional[StorageManager] = None
_query_engine: Optional[QueryEngine] = None


def create_app(data_dir: str, collection_name: str = "ragprep_docs") -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        data_dir: Directory for data storage
        collection_name: Name of the ChromaDB collection
        
    Returns:
        Configured FastAPI application
    """
    global _storage, _query_engine
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifespan."""
        global _storage, _query_engine
        
        # Startup
        logger.info("Starting up RAGPrep server...")
        
        chroma_path = os.path.join(data_dir, "chroma")
        os.makedirs(chroma_path, exist_ok=True)
        
        _storage = StorageManager(
            chroma_path=chroma_path,
            collection_name=collection_name
        )
        
        _query_engine = QueryEngine(
            storage=_storage,
            default_n_results=5
        )
        
        logger.info(f"Server ready: collection={collection_name}")
        yield
        
        # Shutdown
        logger.info("Shutting down RAGPrep server...")
    
    app = FastAPI(
        title="RAGPrep API",
        description="OpenAI-compatible embeddings API for RAGPrep",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Request/Response models
    class EmbeddingRequest(BaseModel):
        """OpenAI-compatible embedding request."""
        input: str | List[str] = Field(
            ...,
            description="Text to embed (string or list of strings)"
        )
        model: Optional[str] = Field(
            "ragprep-default",
            description="Model name (ignored, uses configured model)"
        )
        encoding_format: Optional[Literal["float", "base64"]] = Field(
            "float",
            description="Encoding format for embeddings"
        )
        dimensions: Optional[int] = Field(
            None,
            description="Number of dimensions (ignored, uses model default)"
        )
    
    class EmbeddingData(BaseModel):
        """Single embedding result."""
        object: Literal["embedding"] = "embedding"
        embedding: List[float]
        index: int
    
    class EmbeddingUsage(BaseModel):
        """Token usage info (estimated)."""
        prompt_tokens: int
        total_tokens: int
    
    class EmbeddingResponse(BaseModel):
        """OpenAI-compatible embedding response."""
        object: Literal["list"] = "list"
        data: List[EmbeddingData]
        model: str
        usage: EmbeddingUsage
    
    class SearchRequest(BaseModel):
        """Search request."""
        query: str = Field(..., description="Search query")
        n_results: int = Field(5, ge=1, le=100, description="Number of results")
        min_score: Optional[float] = Field(
            None,
            ge=0,
            le=1,
            description="Minimum similarity score threshold"
        )
        filter: Optional[Dict[str, Any]] = Field(
            None,
            description="Metadata filter (e.g., {'chunk_type': 'paragraph'})"
        )
    
    class SearchResult(BaseModel):
        """Single search result."""
        content: str
        score: float = Field(..., ge=0, le=1)
        distance: float = Field(..., ge=0, le=2)
        source_file: Optional[str] = None
        heading_path: List[str] = Field(default_factory=list)
        chunk_type: Optional[str] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class SearchResponse(BaseModel):
        """Search response."""
        query: str
        n_results: int
        results: List[SearchResult]
        total_found: int
    
    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        version: str
        collection: str
        total_chunks: int
        embedding_model: str
        embedding_dimension: int
    
    class ErrorResponse(BaseModel):
        """Error response."""
        error: str
        detail: Optional[str] = None
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """
        Health check endpoint.
        
        Returns server status and collection statistics.
        """
        if _storage is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Storage not initialized"
            )
        
        stats = _storage.get_collection_stats()
        
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            collection=stats['collection_name'],
            total_chunks=stats['total_chunks'],
            embedding_model=stats['embedding_model'],
            embedding_dimension=stats['embedding_dimension']
        )
    
    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        """
        OpenAI-compatible embeddings endpoint.
        
        Generates embeddings for the provided text(s) using the configured
        sentence-transformers model.
        
        Compatible with OpenAI's /v1/embeddings API.
        """
        if _storage is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Storage not initialized"
            )
        
        # Normalize input to list
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input cannot be empty"
            )
        
        try:
            # Generate embeddings
            embeddings = _storage.embed_texts(texts)
            
            # Build response data
            data = [
                EmbeddingData(
                    object="embedding",
                    embedding=embedding,
                    index=i
                )
                for i, embedding in enumerate(embeddings)
            ]
            
            # Estimate token usage (rough approximation: 1 token ≈ 4 chars)
            total_chars = sum(len(text) for text in texts)
            estimated_tokens = total_chars // 4
            
            return EmbeddingResponse(
                object="list",
                data=data,
                model=_storage.embedding_model_name,
                usage=EmbeddingUsage(
                    prompt_tokens=estimated_tokens,
                    total_tokens=estimated_tokens
                )
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {str(e)}"
            )
    
    @app.post("/v1/search", response_model=SearchResponse)
    async def search_documents(request: SearchRequest) -> SearchResponse:
        """
        Search documents endpoint.
        
        Performs similarity search against the indexed documents and returns
        matching chunks with metadata.
        """
        if _query_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Query engine not initialized"
            )
        
        try:
            # Perform search
            results = _query_engine.search(
                query=request.query,
                n_results=request.n_results,
                filter_dict=request.filter,
                min_score=request.min_score
            )
            
            # Build response
            search_results = [
                SearchResult(
                    content=result.content,
                    score=result.score,
                    distance=result.distance,
                    source_file=result.source_file,
                    heading_path=result.heading_path,
                    chunk_type=result.chunk_type,
                    metadata=result.metadata
                )
                for result in results
            ]
            
            return SearchResponse(
                query=request.query,
                n_results=request.n_results,
                results=search_results,
                total_found=len(search_results)
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}"
            )
    
    @app.get("/v1/collections")
    async def list_collections() -> Dict[str, Any]:
        """
        List available collections.
        
        Returns information about the current collection.
        """
        if _storage is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Storage not initialized"
            )
        
        stats = _storage.get_collection_stats()
        
        return {
            "collections": [
                {
                    "name": stats['collection_name'],
                    "chunks": stats['total_chunks'],
                    "embedding_model": stats['embedding_model'],
                    "embedding_dimension": stats['embedding_dimension']
                }
            ]
        }
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        """Handle generic exceptions."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "detail": str(exc)}
        )
    
    return app


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    data_dir = os.environ.get("RAGPREP_DATA_DIR", "./ragprep_data")
    collection = os.environ.get("RAGPREP_COLLECTION", "ragprep_docs")
    host = os.environ.get("RAGPREP_HOST", "0.0.0.0")
    port = int(os.environ.get("RAGPREP_PORT", "8000"))
    
    app = create_app(data_dir=data_dir, collection_name=collection)
    
    uvicorn.run(app, host=host, port=port)
