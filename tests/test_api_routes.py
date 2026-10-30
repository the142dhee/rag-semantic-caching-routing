"""
tests/test_api_routes.py
========================
PURPOSE: Test FastAPI endpoints (/health, /query, /ingest) using TestClient.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Create a FastAPI test client with mocked environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.main import app
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_root_endpoint(client):
    """GET / should return 200 and basic API links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs"] == "/docs"


def test_health_endpoint(client):
    """GET /api/v1/health should report vector DB status."""
    with patch("app.api.routes.get_document_count", return_value=5), \
         patch("app.api.routes.get_indexed_documents", return_value=[{"filename": "test.pdf", "chunk_count": 5}]):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["vector_db_connected"] is True
        assert data["documents_in_db"] == 5
        assert "test.pdf" in data["indexed_files"]


def test_ingest_missing_file(client):
    """POST /api/v1/ingest with nonexistent file should return 404."""
    response = client.post(
        "/api/v1/ingest",
        json={"pdf_path": "./data/pdfs/nonexistent_file_123.pdf"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_query_endpoint(client):
    """POST /api/v1/query should return answer and sources."""
    from app.models.schemas import QueryResponse
    fake_response = QueryResponse(
        answer="FastAPI is a modern web framework.",
        sources=["Source 1"],
        retrieved_chunks=[],
        model_used="gpt-3.5-turbo",
        latency_ms=120.5,
        from_cache=False,
        cache_similarity=None,
        matched_query=None,
        tokens_used_or_saved=100,
        estimated_cost_saved_usd=0.0,
        routing_tier="SIMPLE",
        routing_complexity_score=0.2,
    )
    with patch("app.api.routes.answer_question", return_value=fake_response):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is FastAPI?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "FastAPI is a modern web framework."
        assert data["from_cache"] is False
        assert data["model_used"] == "gpt-3.5-turbo"
        assert data["routing_tier"] == "SIMPLE"
        assert data["routing_complexity_score"] == 0.2


def test_retrieval_settings_endpoint(client):
    """GET /api/v1/retrieval/settings should return hybrid and BM25 config."""
    response = client.get("/api/v1/retrieval/settings")
    assert response.status_code == 200
    data = response.json()
    assert "hybrid_search_enabled" in data
    assert "dense_weight" in data
    assert "sparse_weight" in data
    assert "bm25_corpus_size" in data


def test_session_management_endpoints(client):
    """GET and DELETE /api/v1/sessions/{session_id} should manage session history."""
    sid = "test-session-123"
    from app.services.query_rewriter import session_manager
    session_manager.add_turn(sid, "What is RAG?", "Retrieval-augmented generation.")

    # Get history
    get_res = client.get(f"/api/v1/sessions/{sid}")
    assert get_res.status_code == 200
    assert get_res.json()["turns"] == 1

    # Delete history
    del_res = client.delete(f"/api/v1/sessions/{sid}")
    assert del_res.status_code == 200
    assert del_res.json()["session_id"] == sid
    assert len(session_manager.get_history(sid)) == 0


def test_judge_evaluate_endpoint(client):
    """POST /api/v1/judge/evaluate should score responses along RAG triad."""
    payload = {
        "question": "What is machine learning?",
        "answer": "Machine learning is a field of artificial intelligence.",
        "context_chunks": ["Machine learning is an application of artificial intelligence."],
        "mock_mode": True,
    }
    response = client.post("/api/v1/judge/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "faithfulness_score" in data
    assert "answer_relevance_score" in data
    assert "context_relevance_score" in data
    assert "overall_score" in data
    assert 1 <= data["faithfulness_score"] <= 5

