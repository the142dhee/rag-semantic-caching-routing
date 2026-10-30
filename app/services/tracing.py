"""
app/services/tracing.py
=======================
PURPOSE: Request-level distributed-style tracing and observability for RAG queries.

Provides:
  - Unique request IDs (UUID4) for end-to-end auditability.
  - Granular stage latencies (cache lookup, retrieval, LLM generation, total).
  - Provenance: retrieved document chunks, routing reason, token usage, cost.
  - Security hardening: automatic credential redaction (never logs API keys).
  - Thread-safe rolling trace buffer.
"""

import re
import time
import uuid
import datetime
import threading
import logging
from typing import List, Optional, Dict, Any
from app.models.schemas import RequestTrace

logger = logging.getLogger(__name__)

# Patterns for sensitive keys to redact
_API_KEY_PATTERN = re.compile(r"(sk-[a-zA-Z0-9_\-]{15,}|Bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE)


def sanitize_text(text: Any) -> str:
    """Redact any accidentally leaked API keys or bearer tokens."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)


class RequestTracer:
    """Thread-safe in-memory rolling storage for request execution traces."""

    def __init__(self, max_traces: int = 200):
        self.max_traces = max_traces
        self._traces: List[RequestTrace] = []
        self._traces_by_id: Dict[str, RequestTrace] = {}
        self._lock = threading.Lock()

    def create_trace_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    def record_trace(
        self,
        request_id: str,
        question: str,
        cache_hit: bool,
        selected_model: str,
        routing_reason: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        total_latency_ms: float,
        answer: str,
        cache_similarity_score: Optional[float] = None,
        retrieval_time_ms: float = 0.0,
        retrieved_documents: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
    ) -> RequestTrace:
        """
        Record a completed query trace with automatic redaction of credentials.
        """
        ts = timestamp if timestamp is not None else time.time()
        iso_ts = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()

        clean_question = sanitize_text(question)
        clean_answer = sanitize_text(answer)
        clean_reason = sanitize_text(routing_reason)

        trace = RequestTrace(
            request_id=str(request_id),
            timestamp=float(ts),
            iso_timestamp=iso_ts,
            question=clean_question,
            cache_hit=bool(cache_hit),
            cache_similarity_score=round(float(cache_similarity_score), 4) if cache_similarity_score is not None else None,
            retrieval_time_ms=round(float(retrieval_time_ms), 2),
            retrieved_documents=[str(d) for d in (retrieved_documents or [])],
            selected_model=str(selected_model),
            routing_reason=clean_reason,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            estimated_cost_usd=round(float(estimated_cost_usd), 6),
            total_latency_ms=round(float(total_latency_ms), 2),
            answer=clean_answer,
        )

        with self._lock:
            self._traces.append(trace)
            self._traces_by_id[trace.request_id] = trace
            if len(self._traces) > self.max_traces:
                oldest = self._traces.pop(0)
                self._traces_by_id.pop(oldest.request_id, None)

        return trace

    def get_traces(self, limit: int = 50) -> List[RequestTrace]:
        """Return the most recent traces in reverse chronological order."""
        with self._lock:
            return list(reversed(self._traces[-limit:]))

    def get_trace(self, request_id: str) -> Optional[RequestTrace]:
        """Look up a specific trace by request ID."""
        with self._lock:
            return self._traces_by_id.get(request_id)

    def clear(self):
        """Clear all stored traces."""
        with self._lock:
            self._traces.clear()
            self._traces_by_id.clear()

    def count(self) -> int:
        """Get the total number of traces stored."""
        with self._lock:
            return len(self._traces)


# Global singleton instance
tracer = RequestTracer()
