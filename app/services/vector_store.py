"""
app/services/vector_store.py
============================
PURPOSE: FAISS vector database service for embedding, indexing, retrieval, and disk persistence.

Key capabilities:
  - Generates embeddings via configurable model (default: OpenAI text-embedding-ada-002).
  - Indexes document chunks and preserves metadata (source document, page number).
  - Automatically loads existing FAISS index from disk at startup (no re-embedding).
  - Persists index to disk whenever new documents are ingested.
  - Returns similarity scores and provenance for each retrieved chunk.
"""

from __future__ import annotations
import os
import threading
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.core.config import get_settings
from app.services.bm25 import bm25_index

logger = logging.getLogger(__name__)

# Module-level singleton and thread lock for safe concurrent operations
_faiss_lock = threading.Lock()
_vector_store: Optional[FAISS] = None
_embedding_model: Optional[OpenAIEmbeddings] = None


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Lazily initialize and return the configured embedding model.
    """
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        if not settings.openai_api_key or settings.openai_api_key.strip() == "":
            raise ValueError(
                "OPENAI_API_KEY is not set. Please define OPENAI_API_KEY in your .env file."
            )
        _embedding_model = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        logger.info(f"Initialized embedding model: {settings.embedding_model}")
    return _embedding_model


def _init_pgvector_store(embed_model: OpenAIEmbeddings) -> Optional[Any]:
    """
    Initialize PGVector vector store connecting to PostgreSQL with the pgvector extension.
    """
    settings = get_settings()
    try:
        from langchain_community.vectorstores import PGVector
        safe_url = settings.postgres_connection_url.split("@")[-1] if "@" in settings.postgres_connection_url else "postgres"
        logger.info(f"Connecting to PGVector backend at: {safe_url}")
        store = PGVector(
            connection_string=settings.postgres_connection_url,
            embedding_function=embed_model,
            collection_name=settings.chroma_collection_name,
        )
        return store
    except Exception as e:
        logger.warning(f"Failed to initialize PGVector backend ({e}). Falling back to local FAISS.")
        return None


def get_vector_store() -> Optional[Any]:
    """
    Get or load the vector store (either PGVector or local FAISS index).
    Returns None if no index exists yet.
    """
    global _vector_store
    with _faiss_lock:
        if _vector_store is None:
            settings = get_settings()

            # Dual-backend check: pgvector vs faiss
            if settings.vector_store_backend.lower() == "pgvector":
                try:
                    embed_model = get_embedding_model()
                    pg_store = _init_pgvector_store(embed_model)
                    if pg_store is not None:
                        _vector_store = pg_store
                        return _vector_store
                except Exception as e:
                    logger.warning(f"PGVector setup error: {e}. Defaulting to local FAISS.")

            # Default: Local FAISS vector store
            index_dir = settings.faiss_index_dir
            index_file = os.path.join(index_dir, "index.faiss")
            pkl_file = os.path.join(index_dir, "index.pkl")

            if os.path.exists(index_file) and os.path.exists(pkl_file):
                try:
                    logger.info(f"Loading existing FAISS index from: {index_dir}")
                    embed_model = get_embedding_model()
                    _vector_store = FAISS.load_local(
                        folder_path=index_dir,
                        embeddings=embed_model,
                        allow_dangerous_deserialization=True,
                    )
                    count = len(_vector_store.docstore._dict) if hasattr(_vector_store, "docstore") else 0
                    logger.info(f"Loaded FAISS index with {count} chunks.")

                    # Synchronize BM25 index with loaded chunks if empty
                    if bm25_index.corpus_size == 0 and hasattr(_vector_store, "docstore"):
                        loaded_docs = [d for d in _vector_store.docstore._dict.values() if isinstance(d, Document)]
                        bm25_index.add_documents(loaded_docs)
                        logger.info(f"Synchronized BM25 index with {len(loaded_docs)} chunks from FAISS docstore.")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from {index_dir}: {e}")
                    _vector_store = None
            else:
                logger.info("No existing FAISS index found on disk. Ready for new documents.")
        return _vector_store


def save_vector_store() -> None:
    """
    Persist current FAISS index and metadata to disk.
    (No-op when using a database-backed vector store like PGVector).
    """
    global _vector_store
    if _vector_store is not None and hasattr(_vector_store, "save_local"):
        settings = get_settings()
        index_dir = settings.faiss_index_dir
        os.makedirs(index_dir, exist_ok=True)
        _vector_store.save_local(index_dir)
        logger.info(f"Saved FAISS index to {index_dir}")


def add_documents(chunks: List[Document]) -> int:
    """
    Embed and add document chunks to the vector store (FAISS or PGVector) and BM25 index.

    Args:
        chunks: List of Document chunks with metadata.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    global _vector_store
    embed_model = get_embedding_model()
    settings = get_settings()

    with _faiss_lock:
        if settings.vector_store_backend.lower() == "pgvector":
            if _vector_store is None:
                _vector_store = _init_pgvector_store(embed_model)
            if _vector_store is not None:
                _vector_store.add_documents(chunks)
                bm25_index.add_documents(chunks)
                return len(chunks)

        # FAISS implementation
        if _vector_store is None:
            logger.info(f"Creating new FAISS index with {len(chunks)} chunks...")
            _vector_store = FAISS.from_documents(chunks, embed_model)
        else:
            logger.info(f"Adding {len(chunks)} chunks to existing FAISS index...")
            _vector_store.add_documents(chunks)

        # Synchronize BM25 sparse index
        bm25_index.add_documents(chunks)
        save_vector_store()

    return len(chunks)


def similarity_search_with_score(
    query: str,
    k: Optional[int] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k chunks with their L2 distance/similarity scores.

    Args:
        query: User question string.
        k: Number of chunks to retrieve.

    Returns:
        List of (Document, score) tuples.
    """
    vs = get_vector_store()
    if vs is None:
        return []

    settings = get_settings()
    k_val = k or settings.top_k_results
    return vs.similarity_search_with_score(query, k=k_val)


def similarity_search(
    query: str,
    k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks formatted as dictionaries with text, metadata, and distance.

    Args:
        query: User question.
        k: Number of chunks to retrieve.

    Returns:
        List of dicts with keys: 'text', 'metadata', 'score', 'source', 'page'.
    """
    results_with_score = similarity_search_with_score(query, k=k)
    output: List[Dict[str, Any]] = []

    for doc, score in results_with_score:
        metadata = doc.metadata or {}
        output.append({
            "text": doc.page_content,
            "metadata": metadata,
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", 1),
            "score": round(float(score), 4),
        })

    return output


def get_document_count() -> int:
    """
    Return the total number of chunks stored in the vector store.
    """
    vs = get_vector_store()
    if vs is None:
        return 0
    if hasattr(vs, "docstore"):
        return len(vs.docstore._dict)
    return bm25_index.corpus_size


def get_indexed_documents() -> List[Dict[str, Any]]:
    """
    Return a summary of indexed documents: filename and chunk count.
    """
    vs = get_vector_store()
    if vs is None:
        return []

    docs = vs.docstore._dict.values() if hasattr(vs, "docstore") else bm25_index.documents
    doc_counts: Dict[str, int] = {}
    for doc in docs:
        if isinstance(doc, Document):
            src = doc.metadata.get("source", "unknown")
            doc_counts[src] = doc_counts.get(src, 0) + 1

    return [
        {"filename": filename, "chunk_count": count}
        for filename, count in sorted(doc_counts.items())
    ]


def get_collection_id() -> str:
    """
    Compute a fingerprint hash of the current document collection.
    If any document is added, updated, or removed, this ID changes,
    ensuring semantic cache isolation and invalidation.
    """
    import hashlib
    docs = get_indexed_documents()
    if not docs:
        return "empty_collection"
    doc_str = "|".join(f"{d['filename']}:{d['chunk_count']}" for d in docs)
    return hashlib.sha256(doc_str.encode("utf-8")).hexdigest()[:16]


def clear_index() -> bool:
    """
    Reset and delete the vector index (FAISS or PGVector) and synchronized BM25 index.
    """
    global _vector_store
    with _faiss_lock:
        settings = get_settings()

        # If PGVector, drop collection if method available
        if _vector_store is not None and hasattr(_vector_store, "delete_collection"):
            try:
                _vector_store.delete_collection()
            except Exception as e:
                logger.warning(f"Error dropping PGVector collection: {e}")

        _vector_store = None

        # Clean local FAISS files if present
        index_dir = settings.faiss_index_dir
        if os.path.exists(index_dir):
            for fname in os.listdir(index_dir):
                fpath = os.path.join(index_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception as e:
                    logger.warning(f"Error removing {fpath}: {e}")

        bm25_index.clear()
        logger.info("Vector index and BM25 index cleared successfully.")
        return True

