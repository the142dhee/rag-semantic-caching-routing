"""
app/services/metrics_tracker.py
===============================
PURPOSE: Telemetry and performance metric tracking for the RAG system and Semantic Cache.

Tracks:
  - Request volume (total, hits, misses, hit rate)
  - Latency breakdown (cache lookup vs total response, hit vs miss)
  - LLM calls avoided
  - Token consumption & cost savings (with clear distinction between measured & estimated)
  - Recent query log for dashboard transparency
"""

import time
import threading
from typing import Dict, Any, List, Optional
from app.core.config import get_settings

_metrics_lock = threading.Lock()


class MetricsTracker:
    """Thread-safe performance and cost metrics collector."""

    def __init__(self):
        self.reset()

    def reset(self):
        with _metrics_lock:
            self.total_requests: int = 0
            self.cache_hits: int = 0
            self.cache_misses: int = 0

            self.total_hit_latency_ms: float = 0.0
            self.total_miss_latency_ms: float = 0.0
            self.total_lookup_latency_ms: float = 0.0

            self.total_tokens_saved: int = 0
            self.total_tokens_used: int = 0

            self.total_cost_saved_usd: float = 0.0
            self.total_cost_incurred_usd: float = 0.0

            # Routing counters
            self.simple_route_count: int = 0
            self.complex_route_count: int = 0
            self.model_call_counts: Dict[str, int] = {}

            self.recent_logs: List[Dict[str, Any]] = []

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate estimated cost in USD based on standard token pricing.
        """
        settings = get_settings()
        input_cost = (prompt_tokens / 1000.0) * settings.cost_per_1k_input_tokens
        output_cost = (completion_tokens / 1000.0) * settings.cost_per_1k_output_tokens
        return round(input_cost + output_cost, 6)

    def record_hit(
        self,
        lookup_latency_ms: float,
        total_latency_ms: float,
        prompt_tokens_saved: int,
        completion_tokens_saved: int,
        query: str,
        matched_query: str,
        similarity_score: float,
    ):
        cost_saved = self.calculate_cost(prompt_tokens_saved, completion_tokens_saved)
        total_tokens = prompt_tokens_saved + completion_tokens_saved

        with _metrics_lock:
            self.total_requests += 1
            self.cache_hits += 1
            self.total_hit_latency_ms += total_latency_ms
            self.total_lookup_latency_ms += lookup_latency_ms
            self.total_tokens_saved += total_tokens
            self.total_cost_saved_usd += cost_saved

            # Add to rolling log (last 50 requests)
            self.recent_logs.append({
                "timestamp": time.time(),
                "query": query,
                "type": "CACHE_HIT",
                "matched_query": matched_query,
                "similarity_score": round(similarity_score, 4),
                "lookup_latency_ms": round(lookup_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "tokens_saved": total_tokens,
                "cost_saved_usd": cost_saved,
            })
            if len(self.recent_logs) > 50:
                self.recent_logs.pop(0)

    def record_miss(
        self,
        lookup_latency_ms: float,
        total_latency_ms: float,
        prompt_tokens_used: int,
        completion_tokens_used: int,
        query: str,
        model_used: str = "",
        routing_tier: str = "",
        complexity_score: Optional[float] = None,
    ):
        cost_incurred = self.calculate_cost(prompt_tokens_used, completion_tokens_used)
        total_tokens = prompt_tokens_used + completion_tokens_used

        with _metrics_lock:
            self.total_requests += 1
            self.cache_misses += 1
            self.total_miss_latency_ms += total_latency_ms
            self.total_lookup_latency_ms += lookup_latency_ms
            self.total_tokens_used += total_tokens
            self.total_cost_incurred_usd += cost_incurred

            # Track routing tier
            if routing_tier == "SIMPLE":
                self.simple_route_count += 1
            elif routing_tier == "COMPLEX":
                self.complex_route_count += 1

            # Track per-model call counts
            if model_used:
                self.model_call_counts[model_used] = self.model_call_counts.get(model_used, 0) + 1

            self.recent_logs.append({
                "timestamp": time.time(),
                "query": query,
                "type": "CACHE_MISS",
                "model_used": model_used,
                "routing_tier": routing_tier,
                "complexity_score": round(complexity_score, 4) if complexity_score is not None else None,
                "lookup_latency_ms": round(lookup_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "tokens_used": total_tokens,
                "cost_incurred_usd": cost_incurred,
            })
            if len(self.recent_logs) > 50:
                self.recent_logs.pop(0)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return comprehensive telemetry metrics for reporting & dashboard.
        """
        with _metrics_lock:
            hit_rate = (self.cache_hits / self.total_requests * 100.0) if self.total_requests > 0 else 0.0
            avg_hit_latency = (self.total_hit_latency_ms / self.cache_hits) if self.cache_hits > 0 else 0.0
            avg_miss_latency = (self.total_miss_latency_ms / self.cache_misses) if self.cache_misses > 0 else 0.0
            avg_lookup_latency = (self.total_lookup_latency_ms / self.total_requests) if self.total_requests > 0 else 0.0

            # Calculate latency speedup factor
            speedup_factor = (avg_miss_latency / avg_hit_latency) if avg_hit_latency > 0 and avg_miss_latency > 0 else 0.0

            # Routing breakdown
            total_routed = self.simple_route_count + self.complex_route_count
            simple_pct = (self.simple_route_count / total_routed * 100) if total_routed > 0 else 0.0
            complex_pct = (self.complex_route_count / total_routed * 100) if total_routed > 0 else 0.0

            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate_pct": round(hit_rate, 2),
                "llm_calls_avoided": self.cache_hits,
                "avg_hit_latency_ms": round(avg_hit_latency, 2),
                "avg_miss_latency_ms": round(avg_miss_latency, 2),
                "avg_lookup_latency_ms": round(avg_lookup_latency, 2),
                "speedup_factor": round(speedup_factor, 1),
                "total_tokens_saved": self.total_tokens_saved,
                "total_cost_saved_usd": round(self.total_cost_saved_usd, 6),
                "total_tokens_used": self.total_tokens_used,
                "total_cost_incurred_usd": round(self.total_cost_incurred_usd, 6),
                "routing": {
                    "simple_count": self.simple_route_count,
                    "complex_count": self.complex_route_count,
                    "simple_pct": round(simple_pct, 2),
                    "complex_pct": round(complex_pct, 2),
                    "model_call_counts": dict(self.model_call_counts),
                },
                "pricing_reference": {
                    "input_per_1k_usd": get_settings().cost_per_1k_input_tokens,
                    "output_per_1k_usd": get_settings().cost_per_1k_output_tokens,
                    "note": "Savings are estimated from token counts using standard model pricing."
                },
                "recent_requests": list(reversed(self.recent_logs[-10:])),
            }


# Global singleton instance
tracker = MetricsTracker()
