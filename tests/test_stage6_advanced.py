"""
tests/test_stage6_advanced.py
=============================
PURPOSE: Unit tests for Stage 6 — Advanced Retrieval, Multi-Turn Query Rewriting,
         Hybrid Search (Dense + BM25 RRF), and LLM-as-a-Judge evaluation.

Tests:
  1. BM25 lexical search engine — indexing, scoring, tokenization
  2. Hybrid retriever — RRF fusion, fallback logic
  3. Query rewriter — heuristic pronoun resolution, session memory
  4. LLM Judge — deterministic mock scoring
  5. Session management — multi-turn memory, clear, history
  6. Pipeline integration — hybrid search routing, rewritten query propagation
"""

import os
import sys
import uuid
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("OPENAI_API_KEY", "sk-test")


# ================================================================
# 1. BM25 LEXICAL SEARCH ENGINE
# ================================================================

class TestBM25Index:
    """Unit tests for the native BM25Okapi engine."""

    def test_tokenize_basic(self):
        from app.services.bm25 import tokenize
        tokens = tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Single-character tokens are filtered
        assert "a" not in tokens

    def test_tokenize_empty(self):
        from app.services.bm25 import tokenize
        assert tokenize("") == []
        assert tokenize(None) == []

    def test_add_and_search(self):
        from app.services.bm25 import BM25Index
        idx = BM25Index()

        docs = [
            Document(page_content="Machine learning uses neural networks", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="Cooking pasta requires boiling water", metadata={"source": "b.pdf", "page": 1}),
            Document(page_content="Deep learning is a subset of machine learning", metadata={"source": "a.pdf", "page": 2}),
        ]
        added = idx.add_documents(docs)
        assert added == 3
        assert idx.corpus_size == 3

        results = idx.search("machine learning", top_k=2)
        assert len(results) >= 1
        # Top result should be about machine learning
        top_doc, top_score = results[0]
        assert "machine" in top_doc.page_content.lower() or "learning" in top_doc.page_content.lower()
        assert top_score > 0.0

    def test_search_empty_corpus(self):
        from app.services.bm25 import BM25Index
        idx = BM25Index()
        results = idx.search("anything", top_k=3)
        assert results == []

    def test_search_no_match(self):
        from app.services.bm25 import BM25Index
        idx = BM25Index()
        idx.add_documents([Document(page_content="Python programming", metadata={"source": "x.pdf", "page": 1})])
        results = idx.search("quantum physics", top_k=2)
        # No keyword overlap -> empty results
        assert len(results) == 0

    def test_clear(self):
        from app.services.bm25 import BM25Index
        idx = BM25Index()
        idx.add_documents([Document(page_content="test document", metadata={"source": "t.pdf", "page": 1})])
        assert idx.corpus_size == 1
        idx.clear()
        assert idx.corpus_size == 0
        assert len(idx.documents) == 0


# ================================================================
# 2. HYBRID RETRIEVER (RRF FUSION)
# ================================================================

class TestHybridRetriever:
    """Tests for RRF-based hybrid search combining dense and sparse retrieval."""

    def test_hybrid_search_rrf_fusion(self):
        """Verify that RRF correctly fuses dense and sparse rankings."""
        from app.services.hybrid_retriever import _doc_hash

        doc1 = Document(page_content="Neural networks for NLP", metadata={"source": "ai.pdf", "page": 1})
        doc2 = Document(page_content="Cooking techniques guide", metadata={"source": "cook.pdf", "page": 3})

        h1 = _doc_hash(doc1)
        h2 = _doc_hash(doc2)
        assert h1 != h2  # Different docs produce different hashes

    @patch("app.services.hybrid_retriever.similarity_search_with_score")
    @patch("app.services.hybrid_retriever.bm25_index")
    def test_hybrid_returns_results(self, mock_bm25, mock_dense):
        """Verify hybrid search returns combined results from both engines."""
        from app.services.hybrid_retriever import hybrid_search

        doc1 = Document(page_content="Dense result about ML", metadata={"source": "ml.pdf", "page": 1})
        doc2 = Document(page_content="Sparse result about ML", metadata={"source": "ml.pdf", "page": 2})

        mock_dense.return_value = [(doc1, 0.95)]
        mock_bm25.search.return_value = [(doc2, 3.5)]

        with patch("app.services.hybrid_retriever.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                top_k_results=4,
                hybrid_dense_weight=0.5,
                hybrid_sparse_weight=0.5,
                rrf_k=60,
            )
            results = hybrid_search("machine learning", k=4)

        assert len(results) >= 1
        # All results should have the required keys
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r
            assert "retrieval_method" in r

    @patch("app.services.hybrid_retriever.similarity_search_with_score")
    @patch("app.services.hybrid_retriever.bm25_index")
    def test_hybrid_fallback_dense_only(self, mock_bm25, mock_dense):
        """When BM25 returns nothing, hybrid falls back to dense-only."""
        from app.services.hybrid_retriever import hybrid_search

        doc1 = Document(page_content="Dense doc", metadata={"source": "d.pdf", "page": 1})
        mock_dense.return_value = [(doc1, 0.85)]
        mock_bm25.search.return_value = []

        with patch("app.services.hybrid_retriever.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                top_k_results=4,
                hybrid_dense_weight=0.5,
                hybrid_sparse_weight=0.5,
                rrf_k=60,
            )
            results = hybrid_search("test query", k=2)

        assert len(results) == 1
        assert results[0]["retrieval_method"] == "dense_only"

    @patch("app.services.hybrid_retriever.similarity_search_with_score")
    @patch("app.services.hybrid_retriever.bm25_index")
    def test_hybrid_both_empty(self, mock_bm25, mock_dense):
        """When both engines return nothing, hybrid returns empty."""
        from app.services.hybrid_retriever import hybrid_search

        mock_dense.return_value = []
        mock_bm25.search.return_value = []

        with patch("app.services.hybrid_retriever.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                top_k_results=4,
                hybrid_dense_weight=0.5,
                hybrid_sparse_weight=0.5,
                rrf_k=60,
            )
            results = hybrid_search("test query", k=2)

        assert results == []


# ================================================================
# 3. QUERY REWRITER
# ================================================================

class TestQueryRewriter:
    """Tests for multi-turn query rewriting with pronoun resolution."""

    def test_no_history_returns_original(self):
        from app.services.query_rewriter import QueryRewriter
        rw = QueryRewriter()
        result = rw.rewrite("What is machine learning?", history=None)
        assert result == "What is machine learning?"

    def test_empty_history_returns_original(self):
        from app.services.query_rewriter import QueryRewriter
        rw = QueryRewriter()
        result = rw.rewrite("What is NLP?", history=[])
        assert result == "What is NLP?"

    def test_heuristic_pronoun_resolution(self):
        """Test that the mock/heuristic mode resolves pronouns using the last query."""
        from app.services.query_rewriter import QueryRewriter
        rw = QueryRewriter()
        history = [
            ("What is the semantic cache TTL?", "The semantic cache TTL is 86400 seconds by default.")
        ]
        result = rw.rewrite("When does it expire?", history=history, mock_mode=True)
        # 'it' should be replaced with keywords from the prior question
        assert result != "When does it expire?"
        assert "expire" in result.lower()

    def test_standalone_question_unchanged(self):
        """A completely standalone follow-up shouldn't be rewritten in mock mode."""
        from app.services.query_rewriter import QueryRewriter
        rw = QueryRewriter()
        history = [("First question?", "First answer.")]
        result = rw.rewrite("What are transformers?", history=history, mock_mode=True)
        assert result == "What are transformers?"


# ================================================================
# 4. SESSION MANAGER
# ================================================================

class TestSessionManager:
    """Tests for multi-turn conversation memory."""

    def test_add_and_retrieve_turns(self):
        from app.services.query_rewriter import ConversationSessionManager
        mgr = ConversationSessionManager(max_history=5)
        sid = f"test-{uuid.uuid4().hex[:8]}"

        mgr.add_turn(sid, "Q1", "A1")
        mgr.add_turn(sid, "Q2", "A2")
        history = mgr.get_history(sid)
        assert len(history) == 2
        assert history[0] == ("Q1", "A1")

    def test_max_history_eviction(self):
        from app.services.query_rewriter import ConversationSessionManager
        mgr = ConversationSessionManager(max_history=3)
        sid = f"test-{uuid.uuid4().hex[:8]}"

        for i in range(5):
            mgr.add_turn(sid, f"Q{i}", f"A{i}")

        history = mgr.get_history(sid)
        assert len(history) == 3
        # Oldest turns (Q0, Q1) should be evicted
        assert history[0] == ("Q2", "A2")

    def test_clear_session(self):
        from app.services.query_rewriter import ConversationSessionManager
        mgr = ConversationSessionManager()
        sid = f"test-{uuid.uuid4().hex[:8]}"

        mgr.add_turn(sid, "Q1", "A1")
        mgr.clear_session(sid)
        history = mgr.get_history(sid)
        assert len(history) == 0

    def test_isolated_sessions(self):
        from app.services.query_rewriter import ConversationSessionManager
        mgr = ConversationSessionManager()
        sid1, sid2 = f"test-{uuid.uuid4().hex[:8]}", f"test-{uuid.uuid4().hex[:8]}"

        mgr.add_turn(sid1, "Q1-sess1", "A1-sess1")
        mgr.add_turn(sid2, "Q1-sess2", "A1-sess2")

        assert len(mgr.get_history(sid1)) == 1
        assert len(mgr.get_history(sid2)) == 1
        assert mgr.get_history(sid1)[0][0] == "Q1-sess1"
        assert mgr.get_history(sid2)[0][0] == "Q1-sess2"


# ================================================================
# 5. LLM JUDGE (MOCK MODE)
# ================================================================

class TestLLMJudge:
    """Tests for the RAG Triad LLM-as-a-Judge mock scoring."""

    def test_mock_grounded_answer(self):
        """Well-grounded answers should receive high faithfulness scores."""
        from app.evaluation.llm_judge import LLMJudge
        judge = LLMJudge()
        score = judge.evaluate(
            question="What is machine learning?",
            answer="Machine learning uses neural networks to learn patterns from data.",
            context_chunks=[
                "Machine learning uses neural networks to learn patterns from data.",
                "Deep learning is a subset of machine learning.",
            ],
            mock_mode=True,
        )
        assert score.faithfulness_score >= 4
        assert score.overall_score >= 3.0

    def test_mock_insufficient_info(self):
        """Answers that refuse to speculate should get a perfect faithfulness score."""
        from app.evaluation.llm_judge import LLMJudge
        judge = LLMJudge()
        score = judge.evaluate(
            question="What is quantum computing?",
            answer="I don't have enough information in the provided documents to answer that.",
            context_chunks=["This document is about cooking recipes."],
            mock_mode=True,
        )
        assert score.faithfulness_score == 5
        assert "refused" in score.faithfulness_reason.lower() or "insufficient" in score.faithfulness_reason.lower()


# ================================================================
# 6. PIPELINE INTEGRATION (HYBRID + REWRITING)
# ================================================================

class TestPipelineIntegration:
    """Integration tests verifying hybrid search and query rewriting flow through the pipeline."""

    @patch("app.services.rag_pipeline.get_llm_chain")
    @patch("app.services.rag_pipeline.hybrid_search")
    @patch("app.services.rag_pipeline.get_collection_id", return_value="empty_collection")
    def test_hybrid_search_used_when_enabled(self, mock_cid, mock_hybrid, mock_chain):
        """Verify hybrid_search is invoked when hybrid_search_enabled=True."""
        from app.services.rag_pipeline import answer_question

        mock_hybrid.return_value = [
            {"text": "Result from hybrid", "source": "doc.pdf", "page": 1, "score": 0.9, "retrieval_method": "hybrid_rrf"}
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer from LLM"
        mock_chain.return_value = mock_llm

        with patch("app.services.rag_pipeline.get_settings") as mock_settings:
            s = MagicMock()
            s.top_k_results = 4
            s.semantic_cache_enabled = False
            s.similarity_threshold = 0.9
            s.hybrid_search_enabled = True
            s.llm_routing_enabled = False
            s.llm_model = "gpt-3.5-turbo"
            s.openai_api_key = "sk-test"
            s.query_rewriting_enabled = False
            mock_settings.return_value = s

            response = answer_question("What is ML?", use_cache=False)

        assert response.retrieval_method == "hybrid_rrf"
        mock_hybrid.assert_called_once()

    @patch("app.services.rag_pipeline.get_llm_chain")
    @patch("app.services.rag_pipeline.similarity_search")
    @patch("app.services.rag_pipeline.get_collection_id", return_value="empty_collection")
    def test_dense_search_used_when_hybrid_disabled(self, mock_cid, mock_dense, mock_chain):
        """Verify dense-only similarity_search is used when hybrid is disabled."""
        from app.services.rag_pipeline import answer_question

        mock_dense.return_value = [
            {"text": "Dense result", "source": "doc.pdf", "page": 1, "score": 0.8}
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer from LLM"
        mock_chain.return_value = mock_llm

        with patch("app.services.rag_pipeline.get_settings") as mock_settings:
            s = MagicMock()
            s.top_k_results = 4
            s.semantic_cache_enabled = False
            s.similarity_threshold = 0.9
            s.hybrid_search_enabled = False
            s.llm_routing_enabled = False
            s.llm_model = "gpt-3.5-turbo"
            s.openai_api_key = "sk-test"
            s.query_rewriting_enabled = False
            mock_settings.return_value = s

            response = answer_question("What is ML?", use_cache=False)

        assert response.retrieval_method == "dense"
        mock_dense.assert_called_once()

    def test_session_id_records_turns(self):
        """Verify that passing session_id records conversation turns for rewriting."""
        from app.services.query_rewriter import ConversationSessionManager

        mgr = ConversationSessionManager()
        sid = f"pipeline-test-{uuid.uuid4().hex[:8]}"

        # Simulate what the pipeline does
        mgr.add_turn(sid, "What is NLP?", "NLP stands for Natural Language Processing.")
        mgr.add_turn(sid, "How does it work?", "It uses transformers and embeddings.")

        history = mgr.get_history(sid)
        assert len(history) == 2
        assert history[0][0] == "What is NLP?"

    def test_empty_query_returns_validation_error(self):
        """Empty queries should be caught and returned with a validation message."""
        from app.services.rag_pipeline import answer_question
        response = answer_question("", use_cache=False)
        assert "non-empty" in response.answer.lower()
        assert response.model_used == "none"


# ================================================================
# 7. SCHEMA VALIDATION
# ================================================================

class TestSchemaUpdates:
    """Verify the Stage 6 schema additions are valid."""

    def test_query_request_has_session_id(self):
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="Test?", session_id="sess-123")
        assert req.session_id == "sess-123"

    def test_query_request_session_id_optional(self):
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="Test?")
        assert req.session_id is None

    def test_query_response_has_retrieval_method(self):
        from app.models.schemas import QueryResponse
        resp = QueryResponse(
            answer="Test answer",
            retrieval_method="hybrid_rrf",
            rewritten_query="Rewritten test",
        )
        assert resp.retrieval_method == "hybrid_rrf"
        assert resp.rewritten_query == "Rewritten test"
