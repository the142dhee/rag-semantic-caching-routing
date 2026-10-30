"""
app/api/routes.py
=================
PURPOSE: FastAPI HTTP endpoints for document ingestion, querying, and semantic cache management.
"""

import io
import os
import shutil
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from app.models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    HealthResponse, DocumentListResponse, DocumentInfo,
    CacheSettingsUpdateRequest, RequestTrace, TraceListResponse,
    JudgeRequest
)
from app.services.pdf_loader import (
    load_and_split_pdf, extract_text_from_pdf_stream, chunk_documents
)
from app.services.vector_store import (
    add_documents, get_document_count, get_indexed_documents, clear_index, get_collection_id
)
from app.services.rag_pipeline import answer_question
from app.services.semantic_cache import semantic_cache
from app.services.metrics_tracker import tracker
from app.services.llm_router import llm_router
from app.services.tracing import tracer
from app.core.config import get_settings
from app.core.security import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "./data/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check API health, vector database connection status, and semantic cache state.
    """
    try:
        settings = get_settings()
        count = get_document_count()
        indexed_docs = get_indexed_documents()
        filenames = [d["filename"] for d in indexed_docs]
        cache_stats = semantic_cache.get_stats()

        return HealthResponse(
            status="ok",
            vector_db_type="FAISS",
            vector_db_connected=True,
            documents_in_db=count,
            indexed_files=filenames,
            semantic_cache_enabled=settings.semantic_cache_enabled,
            cached_entries=cache_stats["total_entries"],
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            vector_db_type="FAISS",
            vector_db_connected=False,
            documents_in_db=0,
            indexed_files=[],
            semantic_cache_enabled=False,
            cached_entries=0,
        )


@router.get("/ready", tags=["System"])
async def readiness_check():
    """
    Kubernetes/Docker component-level readiness probe.
    Verifies vector database and cache filesystem health.
    """
    settings = get_settings()
    checks = {
        "vector_store": False,
        "cache_storage": False,
        "api_key_configured": bool(settings.openai_api_key and settings.openai_api_key != "your-openai-api-key-here"),
    }
    try:
        get_document_count()
        checks["vector_store"] = True
    except Exception as e:
        logger.warning(f"Readiness check vector store error: {e}")

    try:
        os.makedirs(settings.cache_persist_dir, exist_ok=True)
        checks["cache_storage"] = os.path.exists(settings.cache_persist_dir)
    except Exception as e:
        logger.warning(f"Readiness check cache storage error: {e}")

    is_ready = checks["vector_store"] and checks["cache_storage"]
    return {
        "status": "ready" if is_ready else "not_ready",
        "backend": settings.vector_store_backend,
        "checks": checks,
    }


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """
    List all documents currently stored in the vector database with chunk statistics.
    """
    try:
        doc_stats = get_indexed_documents()
        total_chunks = sum(d["chunk_count"] for d in doc_stats)
        return DocumentListResponse(
            total_chunks=total_chunks,
            documents=[DocumentInfo(**d) for d in doc_stats],
        )
    except Exception as e:
        logger.exception("Failed to retrieve document list")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=IngestResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document directly via multipart/form-data.
    Validates, extracts text page-by-page, chunks it, and indexes it in FAISS.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Uploaded file: '{file.filename}'"
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{file.filename}' is empty (0 bytes).")

        # Save an archived copy to data/pdfs
        archive_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(archive_path, "wb") as f_out:
            f_out.write(content)

        # Parse and chunk in-memory
        stream = io.BytesIO(content)
        page_docs, total_pages = extract_text_from_pdf_stream(stream, file.filename)
        chunks = chunk_documents(page_docs)

        # Index in FAISS
        chunks_added = add_documents(chunks)

        return IngestResponse(
            message=f"Successfully uploaded and indexed '{file.filename}'.",
            filename=file.filename,
            chunks_added=chunks_added,
            pages_read=total_pages,
        )

    except ValueError as ve:
        logger.warning(f"Validation error processing uploaded PDF '{file.filename}': {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error ingesting '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest", response_model=IngestResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def ingest_file_path(request: IngestRequest):
    """
    Ingest a PDF from a filesystem path.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found at: {request.pdf_path}"
        )

    filename = os.path.basename(request.pdf_path)
    try:
        chunks = load_and_split_pdf(request.pdf_path)
        chunks_added = add_documents(chunks)

        return IngestResponse(
            message=f"Successfully ingested '{filename}'.",
            filename=filename,
            chunks_added=chunks_added,
            pages_read=len({c.metadata.get('page') for c in chunks}),
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception(f"Ingestion failed for {request.pdf_path}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/query", response_model=QueryResponse, tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def query_documents(request: QueryRequest):
    """
    Ask a question against the ingested document collection.
    Supports multi-turn conversations via session_id for query rewriting.
    Checks semantic cache first; if miss, executes hybrid/dense retrieval + LLM generation.
    """
    try:
        logger.info(
            f"Received query: '{request.question}' "
            f"(top_k={request.top_k}, use_cache={request.use_cache}, "
            f"threshold={request.similarity_threshold}, session_id={request.session_id})"
        )
        response = answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache,
            similarity_threshold=request.similarity_threshold,
            force_model=request.force_model,
            session_id=request.session_id,
        )
        return response
    except Exception as e:
        logger.exception(f"Error processing query: '{request.question}'")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.delete("/documents", tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def reset_index():
    """
    Reset and delete all documents in the vector database and invalidate the semantic cache.
    """
    try:
        clear_index()
        semantic_cache.clear()
        return {"message": "Vector index and associated semantic cache reset successfully."}
    except Exception as e:
        logger.exception("Failed to reset index")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================
# SEMANTIC CACHE MANAGEMENT & METRICS ENDPOINTS
# ========================================================

@router.get("/cache/stats", tags=["Semantic Cache"])
async def get_cache_statistics():
    """
    Return comprehensive telemetry, cache hit/miss counts, latency, and estimated cost savings.
    """
    try:
        summary = tracker.get_summary()
        cache_info = semantic_cache.get_stats()
        settings = get_settings()

        return {
            "metrics": summary,
            "cache_entries": cache_info,
            "settings": {
                "semantic_cache_enabled": settings.semantic_cache_enabled,
                "similarity_threshold": settings.similarity_threshold,
                "cache_ttl_seconds": settings.cache_ttl_seconds,
                "active_collection_id": get_collection_id(),
            }
        }
    except Exception as e:
        logger.exception("Error getting cache statistics")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache", tags=["Semantic Cache"])
async def clear_semantic_cache():
    """
    Clear all entries in the semantic cache.
    Note: Original documents and the FAISS vector index are completely preserved!
    """
    try:
        purged = semantic_cache.clear()
        return {
            "message": f"Successfully cleared {purged} entries from the semantic cache.",
            "documents_intact": True,
            "documents_count": get_document_count(),
        }
    except Exception as e:
        logger.exception("Failed to clear semantic cache")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/settings", tags=["Semantic Cache"])
async def update_cache_settings(payload: CacheSettingsUpdateRequest):
    """
    Adjust runtime semantic cache settings (threshold, enabled/disabled, TTL).
    """
    settings = get_settings()
    if payload.similarity_threshold is not None:
        settings.similarity_threshold = payload.similarity_threshold
    if payload.semantic_cache_enabled is not None:
        settings.semantic_cache_enabled = payload.semantic_cache_enabled
    if payload.cache_ttl_seconds is not None:
        settings.cache_ttl_seconds = payload.cache_ttl_seconds

    return {
        "message": "Semantic cache settings updated successfully.",
        "similarity_threshold": settings.similarity_threshold,
        "semantic_cache_enabled": settings.semantic_cache_enabled,
        "cache_ttl_seconds": settings.cache_ttl_seconds,
    }


# ========================================================
# LLM ROUTER ENDPOINTS (Stage 4)
# ========================================================

@router.get("/router/stats", tags=["LLM Router"])
async def get_router_statistics():
    """
    Return the LLM router's routing history and aggregate statistics.

    Includes:
    - Total queries routed and tier breakdown (SIMPLE vs COMPLEX)
    - Per-model call counts
    - Average complexity score and routing latency
    - Last 10 routing decisions with score breakdowns
    """
    try:
        stats = llm_router.get_routing_stats()
        settings = get_settings()
        return {
            "router_stats": stats,
            "router_config": {
                "routing_enabled": settings.llm_routing_enabled,
                "routing_threshold": settings.routing_threshold,
                "model_simple": settings.llm_model_simple,
                "model_complex": settings.llm_model_complex,
            },
        }
    except Exception as e:
        logger.exception("Error fetching router stats")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/router/stats", tags=["LLM Router"])
async def reset_router_stats():
    """
    Reset the in-memory routing decision history (does not affect models or config).
    """
    llm_router.reset_stats()
    return {"message": "Router statistics reset.", "status": "ok"}


# ========================================================
# OBSERVABILITY & REQUEST TRACES (Stage 5)
# ========================================================

@router.get("/traces", response_model=TraceListResponse, tags=["Observability"])
async def get_traces(limit: int = 50):
    """
    Return the most recent query request execution traces with end-to-end telemetry.
    """
    try:
        traces = tracer.get_traces(limit=limit)
        return TraceListResponse(
            total_traces=len(traces),
            traces=traces,
        )
    except Exception as e:
        logger.exception("Error retrieving traces")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/{request_id}", response_model=RequestTrace, tags=["Observability"])
async def get_trace_by_id(request_id: str):
    """
    Look up a specific request trace by its unique request ID.
    """
    trace = tracer.get_trace(request_id)
    if not trace:
        raise HTTPException(
            status_code=404,
            detail=f"Trace with request_id '{request_id}' not found."
        )
    return trace


@router.delete("/traces", tags=["Observability"])
async def clear_traces():
    """
    Clear all recorded in-memory request execution traces.
    """
    tracer.clear()
    return {"message": "All request traces cleared successfully.", "status": "ok"}


# ========================================================
# BENCHMARK & EVALUATION ENDPOINTS (Stage 5)
# ========================================================

@router.get("/benchmark/results", tags=["Evaluation"])
async def get_benchmark_results():
    """
    Return the most recent benchmark evaluation results.
    """
    results_path = os.path.join("data", "benchmark_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found. Run the benchmark first."
        )
    try:
        import json
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Error loading benchmark results")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark/run", tags=["Evaluation"])
async def run_benchmark_endpoint(mock_mode: bool = True):
    """
    Trigger an execution of the evaluation benchmark suite.
    """
    try:
        from app.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner()
        report = runner.run(mock_llm=mock_mode)
        return {
            "message": "Benchmark completed successfully.",
            "report": report,
        }
    except Exception as e:
        logger.exception("Benchmark run failed")
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


# ========================================================
# SESSION MANAGEMENT (Stage 6 — Multi-Turn Conversations)
# ========================================================

@router.delete("/sessions/{session_id}", tags=["Sessions"])
async def clear_session(session_id: str):
    """
    Clear the conversation history for a specific session ID.
    """
    from app.services.query_rewriter import session_manager
    session_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared.", "session_id": session_id}


@router.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session_history(session_id: str):
    """
    Retrieve the conversation history for a specific session.
    """
    from app.services.query_rewriter import session_manager
    history = session_manager.get_history(session_id)
    return {
        "session_id": session_id,
        "turns": len(history),
        "history": [{"question": q, "answer": a[:200]} for q, a in history],
    }


# ========================================================
# HYBRID RETRIEVAL SETTINGS (Stage 6)
# ========================================================

@router.get("/retrieval/settings", tags=["Retrieval"])
async def get_retrieval_settings():
    """
    Return current retrieval configuration: hybrid vs dense-only, weights, and BM25 stats.
    """
    from app.services.bm25 import bm25_index
    settings = get_settings()
    return {
        "hybrid_search_enabled": settings.hybrid_search_enabled,
        "dense_weight": settings.hybrid_dense_weight,
        "sparse_weight": settings.hybrid_sparse_weight,
        "rrf_k": settings.rrf_k,
        "query_rewriting_enabled": settings.query_rewriting_enabled,
        "bm25_corpus_size": bm25_index.corpus_size,
        "bm25_vocabulary_size": len(bm25_index.doc_freqs),
    }


# ========================================================
# LLM-AS-A-JUDGE (Stage 6 — Automated Evaluation)
# ========================================================

@router.post("/judge/evaluate", tags=["Evaluation"])
async def evaluate_rag_response(request: JudgeRequest):
    """
    Evaluate a question, answer, and context chunks along the RAG Triad:
    Faithfulness, Answer Relevance, and Context Relevance on a 1-5 scale.
    """
    try:
        from app.evaluation.llm_judge import llm_judge
        score = llm_judge.evaluate(
            question=request.question,
            answer=request.answer,
            context_chunks=request.context_chunks,
            mock_mode=request.mock_mode,
        )
        return score.model_dump()
    except Exception as e:
        logger.exception("LLM Judge evaluation failed")
        raise HTTPException(status_code=500, detail=f"Judge evaluation failed: {str(e)}")
