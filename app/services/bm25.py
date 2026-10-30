"""
app/services/bm25.py
====================
PURPOSE: Native, lightweight BM25Okapi lexical retrieval engine.

Provides sparse keyword-based retrieval without external dependencies,
complementing dense FAISS vector embeddings for Hybrid Search.
"""

import math
import re
import string
import threading
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

_bm25_lock = threading.Lock()


def tokenize(text: str) -> List[str]:
    """Lowercase and strip punctuation to extract word tokens."""
    if not text:
        return []
    clean = text.lower()
    clean = "".join(ch if ch.isalnum() else " " for ch in clean)
    return [t for t in clean.split() if len(t) > 1]


class BM25Index:
    """Thread-safe BM25Okapi inverted index for document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0

    def add_documents(self, documents: List[Document]) -> int:
        """Index a list of Document objects."""
        if not documents:
            return 0

        with _bm25_lock:
            for doc in documents:
                tokens = tokenize(doc.page_content)
                self.documents.append(doc)
                self.doc_len.append(len(tokens))

                # Update document frequencies
                unique_tokens = set(tokens)
                for t in unique_tokens:
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

            self.corpus_size = len(self.documents)
            self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0

            # Recompute IDF table
            self.idf.clear()
            for token, freq in self.doc_freqs.items():
                # Lucene/Okapi standard smoothed IDF
                self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        return len(documents)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """
        Rank indexed documents against the query using BM25Okapi scoring.

        Returns:
            List of (Document, score) tuples sorted in descending order of score.
        """
        query_tokens = tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        with _bm25_lock:
            for q_term in query_tokens:
                if q_term not in self.idf:
                    continue
                term_idf = self.idf[q_term]

                for doc_idx, doc in enumerate(self.documents):
                    # Compute term frequency in document
                    doc_tokens = tokenize(doc.page_content)
                    tf = doc_tokens.count(q_term)
                    if tf == 0:
                        continue

                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_idx] / (self.avgdl or 1.0)))
                    scores[doc_idx] += term_idf * (numerator / denominator)

        # Pair scores with documents and sort
        scored_docs = [
            (self.documents[idx], scores[idx])
            for idx in range(self.corpus_size)
            if scores[idx] > 0.0
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def clear(self):
        """Reset the BM25 index."""
        with _bm25_lock:
            self.documents.clear()
            self.doc_len.clear()
            self.avgdl = 0.0
            self.doc_freqs.clear()
            self.idf.clear()
            self.corpus_size = 0


# Global BM25 index instance
bm25_index = BM25Index()
