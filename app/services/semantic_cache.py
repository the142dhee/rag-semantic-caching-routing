"""
app/services/semantic_cache.py
==============================
PURPOSE: Semantic caching engine using vector embedding similarity search,
collection isolation, TTL expiration, and local persistence.

Key concepts:
  - Vector similarity lookup: Evaluates cosine similarity between incoming query
    embedding and stored question vectors.
  - Multi-layer correctness safeguards:
      1. Collection isolation: Cache entries are tied to the specific document
         collection state (hash). Changing the documents invalidates old entries.
      2. Embedding model isolation: Prevents comparing vectors across different
         models or dimensions.
      3. Time-to-Live (TTL): Entries expire after a configurable duration.
      4. Strict similarity thresholding: Uncertain matches (< threshold) are
         treated as cache misses.
"""

from __future__ import annotations
import os
import json
import time
import pickle
import threading
import logging
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single semantically cached question-answer record."""
    query: str
    embedding: List[float]
    answer: str
    sources: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    collection_id: str
    embedding_model: str
    llm_model: str
    timestamp: float
    ttl_seconds: int
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check whether this cache entry has exceeded its TTL."""
        now = current_time or time.time()
        return (now - self.timestamp) > self.ttl_seconds


@dataclass
class CacheLookupResult:
    """Result returned by a semantic cache lookup."""
    hit: bool
    entry: Optional[CacheEntry] = None
    similarity_score: float = 0.0
    lookup_latency_ms: float = 0.0
    matched_query: Optional[str] = None
    reason: str = "MISS"


class SemanticCache:
    """Thread-safe vector-based semantic cache with persistence."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: List[CacheEntry] = []
        self._load_from_disk()

    def _get_storage_path(self) -> str:
        settings = get_settings()
        os.makedirs(settings.cache_persist_dir, exist_ok=True)
        return os.path.join(settings.cache_persist_dir, "semantic_cache.pkl")

    def _save_to_disk(self):
        """Persist cache entries to disk."""
        path = self._get_storage_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(self._entries, f)
            logger.debug(f"Saved {len(self._entries)} semantic cache entries to {path}")
        except Exception as e:
            logger.error(f"Failed to persist semantic cache: {e}")

    def _load_from_disk(self):
        """Load cache entries from disk if available."""
        path = self._get_storage_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._entries = pickle.load(f)
                logger.info(f"Loaded {len(self._entries)} semantic cache entries from disk.")
            except Exception as e:
                logger.warning(f"Could not load semantic cache from disk: {e}. Starting fresh.")
                self._entries = []
        else:
            self._entries = []

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute standard cosine similarity between two 1-D vectors:
        cos_sim(A, B) = (A . B) / (||A|| * ||B||)
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def lookup(
        self,
        query: str,
        query_embedding: List[float],
        collection_id: str,
        similarity_threshold: Optional[float] = None,
    ) -> CacheLookupResult:
        """
        Search the semantic cache for a sufficiently similar previous question.

        Validation rules for a valid Cache Hit:
          1. Same collection_id (ensures knowledge base state matches).
          2. Same embedding_model (ensures vector space compatibility).
          3. Entry is not expired (TTL check).
          4. Cosine similarity >= similarity_threshold.
        """
        start_time = time.perf_counter()
        settings = get_settings()
        threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
        current_time = time.time()
        query_vec = np.array(query_embedding, dtype=np.float32)

        best_entry: Optional[CacheEntry] = None
        best_similarity: float = -1.0

        with self._lock:
            for entry in self._entries:
                # Rule 1: Document collection must match
                if entry.collection_id != collection_id:
                    continue

                # Rule 2: Embedding model must match
                if entry.embedding_model != settings.embedding_model:
                    continue

                # Rule 3: Check TTL expiration
                if entry.is_expired(current_time):
                    continue

                # Rule 4: Compute cosine similarity
                entry_vec = np.array(entry.embedding, dtype=np.float32)
                sim = self._cosine_similarity(query_vec, entry_vec)

                if sim > best_similarity:
                    best_similarity = sim
                    best_entry = entry

        lookup_latency = (time.perf_counter() - start_time) * 1000.0

        # Evaluate threshold
        if best_entry is not None and best_similarity >= threshold:
            logger.info(
                f"[SEMANTIC CACHE HIT] Query: '{query}' matched '{best_entry.query}' "
                f"(Similarity: {best_similarity:.4f} >= {threshold:.2f}, Latency: {lookup_latency:.2f}ms)"
            )
            return CacheLookupResult(
                hit=True,
                entry=best_entry,
                similarity_score=round(best_similarity, 4),
                lookup_latency_ms=round(lookup_latency, 2),
                matched_query=best_entry.query,
                reason="HIT",
            )

        logger.info(
            f"[SEMANTIC CACHE MISS] Query: '{query}' "
            f"(Best similarity: {max(best_similarity, 0.0):.4f} < {threshold:.2f}, Latency: {lookup_latency:.2f}ms)"
        )
        return CacheLookupResult(
            hit=False,
            entry=None,
            similarity_score=round(max(best_similarity, 0.0), 4),
            lookup_latency_ms=round(lookup_latency, 2),
            matched_query=best_entry.query if best_entry else None,
            reason="BELOW_THRESHOLD" if best_entry else "NO_CANDIDATES",
        )

    def store(
        self,
        query: str,
        query_embedding: List[float],
        answer: str,
        sources: List[str],
        retrieved_chunks: List[Dict[str, Any]],
        collection_id: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttl_seconds: Optional[int] = None,
    ) -> CacheEntry:
        """
        Store a new question, embedding, and generated answer in the semantic cache.
        """
        settings = get_settings()
        ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds

        new_entry = CacheEntry(
            query=query.strip(),
            embedding=query_embedding,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            collection_id=collection_id,
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
            timestamp=time.time(),
            ttl_seconds=ttl,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        with self._lock:
            # Check if identical query already exists for this collection to update in-place
            replaced = False
            for idx, existing in enumerate(self._entries):
                if existing.query.lower() == query.strip().lower() and existing.collection_id == collection_id:
                    self._entries[idx] = new_entry
                    replaced = True
                    break

            if not replaced:
                self._entries.append(new_entry)

            self._save_to_disk()

        logger.info(f"Cached new question-answer pair. Total cache entries: {len(self._entries)}")
        return new_entry

    def clear(self) -> int:
        """
        Purge all entries from the semantic cache (does NOT affect the FAISS index).
        """
        with self._lock:
            count = len(self._entries)
            self._entries = []
            self._save_to_disk()
            logger.info(f"Cleared all {count} entries from semantic cache.")
            return count

    def invalidate_collection(self, collection_id: str) -> int:
        """
        Invalidate all cache entries associated with a specific collection ID.
        """
        with self._lock:
            initial = len(self._entries)
            self._entries = [e for e in self._entries if e.collection_id != collection_id]
            removed = initial - len(self._entries)
            if removed > 0:
                self._save_to_disk()
                logger.info(f"Invalidated {removed} cache entries for collection '{collection_id}'.")
            return removed

    def get_stats(self) -> Dict[str, Any]:
        """
        Return inspection statistics about current cache entries.
        """
        now = time.time()
        with self._lock:
            active_entries = [e for e in self._entries if not e.is_expired(now)]
            expired_entries = len(self._entries) - len(active_entries)
            collections = list({e.collection_id for e in self._entries})

            entries_summary = [
                {
                    "query": e.query,
                    "answer_preview": e.answer[:80] + "..." if len(e.answer) > 80 else e.answer,
                    "collection_id": e.collection_id,
                    "age_seconds": round(now - e.timestamp, 1),
                    "ttl_seconds": e.ttl_seconds,
                    "is_expired": e.is_expired(now),
                    "sources": e.sources,
                }
                for e in reversed(self._entries[-20:])  # last 20 entries
            ]

            return {
                "total_entries": len(self._entries),
                "active_entries": len(active_entries),
                "expired_entries": expired_entries,
                "collections_tracked": collections,
                "recent_entries": entries_summary,
            }


# Global singleton instance
semantic_cache = SemanticCache()
