"""
tests/test_stage4_router.py
============================
PURPOSE: Test suite for Stage 4 — Cost-Aware LLM Routing.

WHAT WE'RE TESTING:
  1. Complexity scoring — individual signal contributions
  2. Routing decisions — threshold crossing logic
  3. Model selection — correct tier ↔ model mapping
  4. Configuration — env-var driven model names and threshold
  5. Statistics — aggregation accuracy
  6. RAG pipeline integration — router is called on cache miss

TEST PHILOSOPHY:
  All tests are pure unit tests — no API keys, no real LLM calls, no network.
  The router uses deterministic heuristics, so every test is fully predictable.
"""

import pytest
from unittest.mock import patch, MagicMock


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch):
    """Ensure each test starts with fresh settings (no LRU cache bleed-over)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stage4")
    monkeypatch.setenv("LLM_MODEL_SIMPLE", "gpt-3.5-turbo")
    monkeypatch.setenv("LLM_MODEL_COMPLEX", "gpt-4o-mini")
    monkeypatch.setenv("ROUTING_THRESHOLD", "0.45")
    monkeypatch.setenv("LLM_ROUTING_ENABLED", "true")

    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def router():
    """Fresh LLMRouter instance for each test (bypasses global singleton)."""
    from app.services.llm_router import LLMRouter
    return LLMRouter(routing_threshold=0.45)


# ============================================================
# 1. COMPLEXITY SCORING — INDIVIDUAL SIGNALS
# ============================================================

class TestComplexityScoring:

    def test_very_simple_query_scores_low(self, router):
        """Single-word factual lookups should score below threshold."""
        score, breakdown = router._score_complexity("What is the title?", num_chunks=1)
        assert score < 0.45, f"Expected low score for simple query, got {score}"
        assert breakdown["simple_penalty"] < 0, "Simple query should receive a penalty"

    def test_complex_query_scores_high(self, router):
        """Multi-keyword analytical questions should score above threshold."""
        query = "Compare and analyze the differences between all methodologies across chapters"
        score, breakdown = router._score_complexity(query, num_chunks=4)
        assert score >= 0.45, f"Expected high score for complex query, got {score}"
        assert breakdown["complex_keyword"] > 0

    def test_word_count_signal_increases_with_length(self, router):
        """Longer queries should generate a higher word_count signal."""
        short_score, _ = router._score_complexity("What is risk?", num_chunks=0)
        long_query = "Can you provide a comprehensive and detailed explanation of all the risk factors mentioned throughout the entire document including their interactions?"
        long_score, _ = router._score_complexity(long_query, num_chunks=0)
        assert long_score > short_score, "Longer query should score higher"

    def test_multi_part_signal_detects_multiple_questions(self, router):
        """Queries with multiple question marks should trigger multi-part signal."""
        score, breakdown = router._score_complexity(
            "What are the risks? How do they interact? What should we do?", num_chunks=0
        )
        assert breakdown["multi_part"] > 0, "Multi-part signal should fire for multiple '?'"

    def test_chunk_count_signal_increases_with_more_chunks(self, router):
        """More retrieved chunks should increase complexity signal."""
        score_1, b1 = router._score_complexity("What is risk?", num_chunks=1)
        score_4, b4 = router._score_complexity("What is risk?", num_chunks=4)
        assert b4["chunk_load"] > b1["chunk_load"], "More chunks should increase chunk_load score"

    def test_score_is_clamped_between_0_and_1(self, router):
        """Score must always be in [0.0, 1.0]."""
        extremes = [
            ("a", 0),
            ("compare analyze evaluate summarize elaborate explain contrast differences similarities", 10),
        ]
        for q, n in extremes:
            score, _ = router._score_complexity(q, num_chunks=n)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range for query: '{q}'"

    def test_breakdown_keys_are_complete(self, router):
        """Breakdown dict must contain all expected signal keys."""
        _, breakdown = router._score_complexity("What are the key findings?", num_chunks=2)
        expected_keys = {"complex_keyword", "word_count", "multi_part", "chunk_load", "simple_penalty"}
        assert set(breakdown.keys()) == expected_keys


# ============================================================
# 2. ROUTING DECISIONS — THRESHOLD CROSSING
# ============================================================

class TestRoutingDecisions:

    def test_simple_query_routes_to_simple_tier(self, router):
        """A factual query must route to the SIMPLE tier."""
        decision = router.route("What is the author's name?", num_chunks=1)
        assert decision.tier == "SIMPLE"
        assert decision.complexity_score < 0.45

    def test_complex_query_routes_to_complex_tier(self, router):
        """An analytical query must route to the COMPLEX tier."""
        decision = router.route(
            "Analyze and compare the key differences between all the risk assessment methodologies described",
            num_chunks=4,
        )
        assert decision.tier == "COMPLEX"
        assert decision.complexity_score >= 0.45

    def test_routing_decision_contains_all_fields(self, router):
        """Every routing decision must include all required fields."""
        decision = router.route("What is the title?", num_chunks=2)
        assert decision.query != ""
        assert decision.tier in ("SIMPLE", "COMPLEX")
        assert decision.model_selected != ""
        assert 0.0 <= decision.complexity_score <= 1.0
        assert decision.routing_latency_ms >= 0.0
        assert isinstance(decision.score_breakdown, dict)
        assert decision.reason != ""

    def test_routing_latency_is_sub_millisecond(self, router):
        """Heuristic classification must complete in well under 10ms."""
        decision = router.route("Explain the main findings", num_chunks=3)
        assert decision.routing_latency_ms < 10.0, (
            f"Routing took too long: {decision.routing_latency_ms}ms (expected < 10ms)"
        )

    def test_to_dict_serializes_correctly(self, router):
        """RoutingDecision.to_dict() must produce a JSON-serializable dict."""
        decision = router.route("What is risk?", num_chunks=1)
        d = decision.to_dict()
        assert "tier" in d
        assert "complexity_score" in d
        assert isinstance(d["score_breakdown"], dict)
        # Verify no non-serializable types
        import json
        json.dumps(d)  # should not raise


# ============================================================
# 3. MODEL SELECTION — CORRECT TIER-MODEL MAPPING
# ============================================================

class TestModelSelection:

    def test_simple_tier_uses_simple_model(self, router, monkeypatch):
        """SIMPLE tier must select the llm_model_simple from settings."""
        from app.core.config import get_settings
        monkeypatch.setenv("LLM_MODEL_SIMPLE", "gpt-3.5-turbo")
        get_settings.cache_clear()

        decision = router.route("What is the title?", num_chunks=1)
        if decision.tier == "SIMPLE":
            settings = get_settings()
            assert decision.model_selected == settings.llm_model_simple

    def test_complex_tier_uses_complex_model(self, router, monkeypatch):
        """COMPLEX tier must select the llm_model_complex from settings."""
        from app.core.config import get_settings
        monkeypatch.setenv("LLM_MODEL_COMPLEX", "gpt-4o-mini")
        get_settings.cache_clear()

        decision = router.route(
            "Analyze and compare all risk methodologies across the document thoroughly",
            num_chunks=5,
        )
        if decision.tier == "COMPLEX":
            settings = get_settings()
            assert decision.model_selected == settings.llm_model_complex

    def test_custom_threshold_changes_routing(self):
        """A higher threshold should classify more queries as SIMPLE."""
        from app.services.llm_router import LLMRouter
        strict_router = LLMRouter(routing_threshold=0.99)  # Near-impossible to exceed
        decision = strict_router.route(
            "Compare and analyze all the key findings across chapters", num_chunks=4
        )
        # With threshold=0.99, almost every query should be SIMPLE
        assert decision.tier == "SIMPLE"

    def test_low_threshold_routes_everything_complex(self):
        """A threshold of 0.0 should route virtually everything as COMPLEX."""
        from app.services.llm_router import LLMRouter
        permissive_router = LLMRouter(routing_threshold=0.0)
        decision = permissive_router.route("What is the title?", num_chunks=1)
        # Even the simplest query gets routed to COMPLEX at threshold=0.0
        assert decision.tier == "COMPLEX"


# ============================================================
# 4. ROUTER STATISTICS AGGREGATION
# ============================================================

class TestRouterStatistics:

    def test_empty_router_returns_zero_stats(self, router):
        """A fresh router with no decisions should return zero stats."""
        stats = router.get_routing_stats()
        assert stats["total_routed"] == 0
        assert stats["simple_count"] == 0
        assert stats["complex_count"] == 0

    def test_stats_count_correctly_after_routes(self, router):
        """Stats must accurately count routed decisions."""
        # Force known outcomes by using extreme threshold boundaries
        router._threshold = 0.0  # everything → COMPLEX
        router.route("What is the title?", num_chunks=1)
        router.route("What are the risks?", num_chunks=2)
        router._threshold = 0.99  # everything → SIMPLE
        router.route("Explain the findings", num_chunks=1)

        stats = router.get_routing_stats()
        assert stats["total_routed"] == 3
        assert stats["complex_count"] == 2
        assert stats["simple_count"] == 1

    def test_model_usage_is_tracked(self, router, monkeypatch):
        """Model names must appear in model_usage dict after routing."""
        from app.core.config import get_settings
        monkeypatch.setenv("LLM_MODEL_SIMPLE", "gpt-3.5-turbo")
        monkeypatch.setenv("LLM_MODEL_COMPLEX", "gpt-4o-mini")
        get_settings.cache_clear()

        router._threshold = 0.0   # everything → COMPLEX
        router.route("What?", num_chunks=1)
        router._threshold = 0.99  # everything → SIMPLE
        router.route("Summarize everything in the document including all chapters.", num_chunks=4)

        stats = router.get_routing_stats()
        model_usage = stats["model_usage"]
        assert len(model_usage) >= 1  # At least one model tracked

    def test_recent_decisions_are_listed(self, router):
        """get_routing_stats() should include recent decision list."""
        router.route("What is X?", num_chunks=1)
        stats = router.get_routing_stats()
        assert "recent_decisions" in stats
        assert len(stats["recent_decisions"]) >= 1
        assert "tier" in stats["recent_decisions"][0]

    def test_reset_clears_log(self, router):
        """reset_stats() must clear all routing history."""
        router.route("What?", num_chunks=1)
        router.route("Analyze everything.", num_chunks=4)
        router.reset_stats()
        stats = router.get_routing_stats()
        assert stats["total_routed"] == 0

    def test_percentages_sum_to_100(self, router):
        """simple_pct + complex_pct must equal 100% (within float tolerance)."""
        router._threshold = 0.45
        queries = [
            ("What is the author?", 1),
            ("Compare and analyze all methodologies across the document", 4),
            ("When was the document published?", 1),
            ("Summarize the key findings and explain the implications", 3),
        ]
        for q, n in queries:
            router.route(q, num_chunks=n)

        stats = router.get_routing_stats()
        total = stats["simple_pct"] + stats["complex_pct"]
        assert abs(total - 100.0) < 0.01, f"Percentages should sum to 100, got {total}"


# ============================================================
# 5. RAG PIPELINE INTEGRATION — ROUTER CALLED ON CACHE MISS
# ============================================================

class TestRAGPipelineIntegration:
    """
    Verify the router is invoked by the RAG pipeline on cache misses.
    All LLM and embedding calls are mocked — no API keys needed.
    """

    @patch("app.services.rag_pipeline.get_collection_id", return_value="test_coll_123")
    @patch("app.services.rag_pipeline.get_embedding_model")
    @patch("app.services.rag_pipeline.similarity_search")
    @patch("app.services.rag_pipeline.get_llm_chain")
    @patch("app.services.rag_pipeline.semantic_cache")
    @patch("app.services.rag_pipeline.llm_router")
    def test_router_is_called_on_cache_miss(
        self,
        mock_router,
        mock_cache,
        mock_llm_chain,
        mock_similarity,
        mock_embed_model,
        mock_coll_id,
        monkeypatch,
    ):
        """The router.route() method must be invoked when the cache misses."""
        from app.core.config import get_settings
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_ROUTING_ENABLED", "true")
        get_settings.cache_clear()

        # Cache always misses
        mock_cache.lookup.return_value = MagicMock(hit=False, lookup_latency_ms=0.5)

        # Return two fake chunks
        mock_similarity.return_value = [
            {"text": "Capital is Paris.", "source": "doc.pdf", "page": 1, "score": 0.95},
            {"text": "France is in Europe.", "source": "doc.pdf", "page": 2, "score": 0.88},
        ]

        # Router returns a SIMPLE decision
        mock_decision = MagicMock()
        mock_decision.tier = "SIMPLE"
        mock_decision.model_selected = "gpt-3.5-turbo"
        mock_decision.complexity_score = 0.15
        mock_router.route.return_value = mock_decision

        # LLM chain returns a fake answer
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Paris is the capital of France."
        mock_llm_chain.return_value = mock_chain

        from app.services.rag_pipeline import answer_question
        response = answer_question("What is the capital of France?", use_cache=False)

        # Router must have been invoked
        mock_router.route.assert_called_once()
        call_kwargs = mock_router.route.call_args
        assert call_kwargs is not None

    @patch("app.services.rag_pipeline.get_collection_id", return_value="test_coll_123")
    @patch("app.services.rag_pipeline.get_embedding_model")
    @patch("app.services.rag_pipeline.similarity_search")
    @patch("app.services.rag_pipeline.get_llm_chain")
    @patch("app.services.rag_pipeline.semantic_cache")
    @patch("app.services.rag_pipeline.llm_router")
    def test_force_model_bypasses_router(
        self,
        mock_router,
        mock_cache,
        mock_llm_chain,
        mock_similarity,
        mock_embed_model,
        mock_coll_id,
        monkeypatch,
    ):
        """When force_model is set, the router should NOT be called."""
        from app.core.config import get_settings
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_ROUTING_ENABLED", "true")
        get_settings.cache_clear()

        mock_cache.lookup.return_value = MagicMock(hit=False, lookup_latency_ms=0.5)
        mock_similarity.return_value = [
            {"text": "Answer text.", "source": "doc.pdf", "page": 1, "score": 0.9}
        ]
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Forced answer."
        mock_llm_chain.return_value = mock_chain

        from app.services.rag_pipeline import answer_question
        response = answer_question(
            "What is the answer?",
            use_cache=False,
            force_model="gpt-4o",
        )

        # Router must NOT have been called
        mock_router.route.assert_not_called()
        # Response should name the forced model
        assert response.model_used == "gpt-4o"

    @patch("app.services.rag_pipeline.get_collection_id", return_value="test_coll_123")
    @patch("app.services.rag_pipeline.get_embedding_model")
    @patch("app.services.rag_pipeline.semantic_cache")
    def test_cache_hit_skips_router(
        self, mock_cache, mock_embed_model, mock_coll_id, monkeypatch
    ):
        """On a cache hit the router should never be invoked."""
        from app.core.config import get_settings
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        get_settings.cache_clear()

        # Use spec=None to get a plain MagicMock, then set real values for Pydantic-validated fields
        cached_entry = MagicMock(spec=[
            "answer", "sources", "retrieved_chunks", "llm_model",
            "prompt_tokens", "completion_tokens", "query",
        ])
        cached_entry.answer = "Cached answer."
        cached_entry.sources = ["doc.pdf (Page 1)"]
        cached_entry.retrieved_chunks = []   # empty list — avoids SourceChunk parse issues
        cached_entry.llm_model = "gpt-3.5-turbo"
        cached_entry.prompt_tokens = 50
        cached_entry.completion_tokens = 20
        cached_entry.query = "What is the answer?"   # must be str, not MagicMock attr

        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 10
        mock_embed_model.return_value = mock_embed

        mock_cache.lookup.return_value = MagicMock(
            hit=True,
            entry=cached_entry,
            lookup_latency_ms=0.2,
            similarity_score=0.95,
        )

        with patch("app.services.rag_pipeline.llm_router") as mock_router:
            from app.services.rag_pipeline import answer_question
            response = answer_question("What is the answer?", use_cache=True)

            mock_router.route.assert_not_called()
            assert response.from_cache is True
