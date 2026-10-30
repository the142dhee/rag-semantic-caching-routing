"""
app/evaluation/metrics.py
=========================
PURPOSE: Scientific evaluation metrics engine for RAG retrieval, answer quality,
and system telemetry.

Includes:
  1. Retrieval Metrics:
     - Precision@K
     - Recall@K
     - Retrieval Latency
  2. Answer Quality Metrics:
     - Answer Correctness (Token F1 / Lexical overlap against reference answer)
     - Faithfulness / Groundedness (measuring context adherence & hallucination prevention)
     - Context Relevance (fraction of retrieved context chunks relevant to query)
     - Answer Completeness (coverage of key reference information)
  3. System Observability Metrics:
     - Mean, p50, p90, p95 Latency percentiles
     - Cache Hit Rate & Cost Savings
     - LLM Calls Executed vs Avoided
"""

import re
import string
import numpy as np
from typing import List, Dict, Any, Optional, Set


# ============================================================
# TEXT TOKENIZATION & NORMALIZATION UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def get_tokens(text: str) -> List[str]:
    """Tokenize normalized text into words."""
    norm = normalize_text(text)
    return norm.split() if norm else []


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def calculate_precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: Optional[int] = None) -> float:
    """
    Calculate Precision@K: (relevant chunks in top K) / K.
    """
    if not retrieved_sources:
        return 0.0
    k_val = k if (k is not None and k > 0) else len(retrieved_sources)
    top_k = retrieved_sources[:k_val]

    if not relevant_sources:
        # If no sources are supposed to be relevant (e.g. unanswerable), precision is 1 if 0 retrieved, else 0
        return 1.0 if not top_k else 0.0

    hits = 0
    for ret in top_k:
        if any(rel.lower() in ret.lower() or ret.lower() in rel.lower() for rel in relevant_sources):
            hits += 1

    return hits / len(top_k)


def calculate_recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: Optional[int] = None) -> float:
    """
    Calculate Recall@K: (relevant chunks in top K) / (total relevant chunks).
    """
    if not relevant_sources:
        return 1.0 if not retrieved_sources else 0.0

    k_val = k if (k is not None and k > 0) else len(retrieved_sources)
    top_k = retrieved_sources[:k_val]

    matched_relevant = set()
    for ret in top_k:
        for rel in relevant_sources:
            if rel.lower() in ret.lower() or ret.lower() in rel.lower():
                matched_relevant.add(rel)

    return len(matched_relevant) / len(relevant_sources)


# ============================================================
# ANSWER QUALITY METRICS
# ============================================================

def calculate_token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute standard Token-level Precision, Recall, and F1 (SQuAD standard).
    """
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)

    if not pred_tokens or not ref_tokens:
        exact = int(normalize_text(prediction) == normalize_text(reference))
        return {"precision": float(exact), "recall": float(exact), "f1": float(exact)}

    common = set(pred_tokens) & set(ref_tokens)
    num_same = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common)

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def calculate_faithfulness(answer: str, context_chunks: List[str], is_answerable: bool = True) -> float:
    """
    Evaluate faithfulness/groundedness.
    - If unanswerable: returns 1.0 if the answer contains the refusal phrase.
    - If answerable: evaluates proportion of content words in answer supported by context.
    """
    lower_ans = answer.lower()
    refusal_cues = [
        "don't have enough information",
        "do not have enough information",
        "not enough information",
        "insufficient information",
        "provided documents do not",
    ]

    if not is_answerable:
        # Ground truth is unanswerable: model must refuse
        if any(cue in lower_ans for cue in refusal_cues):
            return 1.0
        return 0.0

    # Answerable question: if model refused unnecessarily, score is low
    if any(cue in lower_ans for cue in refusal_cues):
        return 0.0

    ans_tokens = get_tokens(answer)
    if not ans_tokens:
        return 0.0

    combined_context = " ".join(context_chunks)
    context_tokens = set(get_tokens(combined_context))

    # Stopwords filter
    stopwords = {"and", "or", "in", "on", "at", "to", "for", "with", "by", "from", "of", "is", "are", "was", "were", "be", "been", "that", "this", "it", "as"}
    content_tokens = [t for t in ans_tokens if t not in stopwords and len(t) > 2]

    if not content_tokens:
        return 1.0

    grounded_count = sum(1 for t in content_tokens if t in context_tokens)
    return round(grounded_count / len(content_tokens), 4)


def calculate_context_relevance(question: str, context_chunks: List[str]) -> float:
    """
    Proportion of retrieved chunks that share non-trivial semantic keywords with the question.
    """
    if not context_chunks:
        return 0.0

    q_tokens = set(t for t in get_tokens(question) if len(t) > 3)
    if not q_tokens:
        return 1.0

    relevant_chunks = 0
    for chunk in context_chunks:
        chunk_tokens = set(get_tokens(chunk))
        if q_tokens & chunk_tokens:
            relevant_chunks += 1

    return round(relevant_chunks / len(context_chunks), 4)


def calculate_answer_completeness(answer: str, reference: str) -> float:
    """
    Fraction of key content words from reference answer covered by the generated answer.
    """
    ref_tokens = set(t for t in get_tokens(reference) if len(t) > 3)
    if not ref_tokens:
        return 1.0

    ans_tokens = set(get_tokens(answer))
    covered = ref_tokens & ans_tokens
    return round(len(covered) / len(ref_tokens), 4)


# ============================================================
# SYSTEM TELEMETRY PERCENTILES
# ============================================================

def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate mean, p50, p90, and p95 latency percentiles from a list of latencies.
    """
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(latencies)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }
