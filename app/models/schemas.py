"""
app/models/schemas.py
=====================
PURPOSE: Define Pydantic request and response schemas for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class IngestRequest(BaseModel):
    """Path-based document ingestion request."""
    pdf_path: str = Field(
        ...,
        description="Path to the PDF file to ingest",
        examples=["./data/pdfs/sample.pdf"],
    )


class IngestResponse(BaseModel):
    """Response after ingesting a document."""
    message: str
    filename: str
    chunks_added: int
    pages_read: int = 0


class SourceChunk(BaseModel):
    """Individual retrieved text chunk with provenance metadata."""
    text: str
    source: str
    page: int
    score: Optional[float] = None


class QueryRequest(BaseModel):
    """Query request body."""
    question: str = Field(
        ...,
        description="Question to ask about the ingested documents",
        min_length=2,
        examples=["What are the key findings?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Optional override for number of chunks to retrieve",
        ge=1,
        le=20,
    )
    use_cache: Optional[bool] = Field(
        default=None,
        description="Whether to use semantic cache (defaults to system setting)",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Optional override for semantic cache similarity threshold (0.0 - 1.0)",
        ge=0.0,
        le=1.0,
    )
    force_model: Optional[str] = Field(
        default=None,
        description="Override the LLM router and use a specific model name (e.g. 'gpt-4o')",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn conversations. Enables query rewriting for follow-up questions.",
    )


class QueryResponse(BaseModel):
    """RAG query answer response."""
    answer: str
    sources: List[str] = Field(
        default_factory=list,
        description="Citations in 'filename (Page X)' format",
    )
    retrieved_chunks: List[SourceChunk] = Field(
        default_factory=list,
        description="Detailed retrieved chunk data with metadata and scores",
    )
    model_used: str = "gpt-3.5-turbo"
    latency_ms: Optional[float] = None
    from_cache: bool = False
    cache_similarity: Optional[float] = None
    matched_query: Optional[str] = None
    tokens_used_or_saved: Optional[int] = None
    estimated_cost_saved_usd: Optional[float] = None
    # Cost-aware routing metadata (Stage 4)
    routing_tier: Optional[str] = Field(
        default=None,
        description="Router tier selected: 'SIMPLE', 'COMPLEX', or None (from cache)",
    )
    routing_complexity_score: Optional[float] = Field(
        default=None,
        description="Heuristic complexity score [0.0–1.0] that drove the routing decision",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request tracing ID (UUID4)",
    )
    retrieval_method: Optional[str] = Field(
        default=None,
        description="Retrieval strategy used: 'dense', 'hybrid_rrf', 'dense_only', 'sparse_only'",
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Reformulated query after multi-turn rewriting (None if unchanged)",
    )


class RequestTrace(BaseModel):
    """Detailed end-to-end trace for an individual query request."""
    request_id: str
    timestamp: float
    iso_timestamp: str
    question: str
    cache_hit: bool
    cache_similarity_score: Optional[float] = None
    retrieval_time_ms: float = 0.0
    retrieved_documents: List[str] = Field(default_factory=list)
    selected_model: str
    routing_reason: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    total_latency_ms: float
    answer: str


class TraceListResponse(BaseModel):
    """Response containing list of recent traces."""
    total_traces: int
    traces: List[RequestTrace]


class DocumentInfo(BaseModel):
    """Summary of an indexed document."""
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """List of all documents stored in the vector store."""
    total_chunks: int
    documents: List[DocumentInfo]


class HealthResponse(BaseModel):
    """Health status and vector database statistics."""
    status: str
    vector_db_type: str = "FAISS"
    vector_db_connected: bool
    documents_in_db: int
    indexed_files: List[str] = Field(default_factory=list)
    semantic_cache_enabled: bool = True
    cached_entries: int = 0


class CacheSettingsUpdateRequest(BaseModel):
    """Payload to adjust cache settings dynamically."""
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    semantic_cache_enabled: Optional[bool] = None
    cache_ttl_seconds: Optional[int] = Field(None, ge=1)


class JudgeRequest(BaseModel):
    """Request payload for running LLM-as-a-Judge evaluation on a query/answer pair."""
    question: str
    answer: str
    context_chunks: List[str] = Field(default_factory=list)
    mock_mode: bool = False

