"""
app/services/hybrid_retriever.py
================================
PURPOSE: Hybrid Search fusing Dense Vector Search (FAISS) and Sparse Lexical Search (BM25)
using Reciprocal Rank Fusion (RRF).

Mathematical Formulation:
    RRF_score(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))
    where k = 60 (standard Cormack constant).
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.vector_store import similarity_search_with_score
from app.services.bm25 import bm25_index

logger = logging.getLogger(__name__)


def _doc_hash(doc: Document) -> str:
    """Compute a stable content hash for document deduplication in rank fusion."""
    text_sample = doc.page_content.strip()
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", 1)
    return hashlib.md5(f"{source}:{page}:{text_sample[:100]}".encode("utf-8")).hexdigest()


def hybrid_search(
    query: str,
    k: Optional[int] = None,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
    rrf_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Execute Hybrid Retrieval combining FAISS dense vectors and BM25 sparse keywords.

    Args:
        query: User question string.
        k: Number of final chunks to return (defaults to settings.top_k_results).
        dense_weight: Weight multiplier for dense ranks (default 0.5).
        sparse_weight: Weight multiplier for sparse ranks (default 0.5).
        rrf_k: Smoothing constant for reciprocal rank fusion (default 60).

    Returns:
        List of chunk dicts formatted with text, source, page, and RRF score.
    """
    settings = get_settings()
    top_k = k or settings.top_k_results
    w_dense = dense_weight if dense_weight is not None else settings.hybrid_dense_weight
    w_sparse = sparse_weight if sparse_weight is not None else settings.hybrid_sparse_weight
    constant_k = rrf_k if rrf_k is not None else settings.rrf_k

    fetch_k = top_k * 3  # Fetch a broader candidate pool for reranking

    # 1. Dense retrieval (FAISS)
    try:
        dense_results = similarity_search_with_score(query, k=fetch_k)
    except Exception as e:
        logger.warning(f"Dense vector retrieval failed during hybrid search: {e}")
        dense_results = []

    # 2. Sparse retrieval (BM25)
    try:
        sparse_results = bm25_index.search(query, top_k=fetch_k)
    except Exception as e:
        logger.warning(f"Sparse BM25 retrieval failed during hybrid search: {e}")
        sparse_results = []

    # If both empty, return empty
    if not dense_results and not sparse_results:
        return []

    # If only dense has documents
    if not sparse_results:
        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 1),
                "score": round(score, 4),
                "retrieval_method": "dense_only",
            }
            for doc, score in dense_results[:top_k]
        ]

    # If only sparse has documents
    if not dense_results:
        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 1),
                "score": round(score, 4),
                "retrieval_method": "sparse_only",
            }
            for doc, score in sparse_results[:top_k]
        ]

    # 3. Reciprocal Rank Fusion (RRF)
    doc_registry: Dict[str, Document] = {}
    rrf_scores: Dict[str, float] = {}

    # Accumulate dense ranks
    for rank_idx, (doc, _score) in enumerate(dense_results, start=1):
        d_id = _doc_hash(doc)
        doc_registry[d_id] = doc
        rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (w_dense / (constant_k + rank_idx))

    # Accumulate sparse ranks
    for rank_idx, (doc, _score) in enumerate(sparse_results, start=1):
        d_id = _doc_hash(doc)
        doc_registry[d_id] = doc
        rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (w_sparse / (constant_k + rank_idx))

    # Sort candidates by combined RRF score
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for d_id, fused_score in sorted_candidates[:top_k]:
        doc = doc_registry[d_id]
        results.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 1),
            "score": round(fused_score, 6),
            "retrieval_method": "hybrid_rrf",
        })

    logger.info(f"Hybrid RRF retrieved {len(results)} chunks (dense={len(dense_results)}, sparse={len(sparse_results)})")
    return results
