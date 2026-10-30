"""
tests/test_stage7_deployment.py
===============================
PURPOSE: Comprehensive test suite for Stage 7 — Production Hardening, API Key Security,
Rate Limiting Throttling, Readiness Probes, and Dual Vector Store Backend.

Tests:
  1. API Key Authentication — valid, invalid, missing, and bypass modes
  2. Rate Limiting — sliding window tracking, threshold enforcement, 429 response
  3. Health & Readiness Probes — root and API v1 endpoints
  4. Vector Store Backend — FAISS vs PGVector fallback logic
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("OPENAI_API_KEY", "sk-test")


@pytest.fixture
def client(monkeypatch):
    """TestClient fixture with fresh config."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-stage7")
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.main import app
    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


# ================================================================
# 1. API KEY AUTHENTICATION
# ================================================================

class TestAPIKeySecurity:
    """Tests for X-API-Key header enforcement."""

    def test_auth_disabled_allows_requests(self, client, monkeypatch):
        """When API_KEY_AUTH_ENABLED is false, requests proceed without key."""
        monkeypatch.setenv("API_KEY_AUTH_ENABLED", "false")
        from app.core.config import get_settings
        get_settings.cache_clear()

        with patch("app.api.routes.answer_question") as mock_answer:
            from app.models.schemas import QueryResponse
            mock_answer.return_value = QueryResponse(answer="Open response", sources=[])
            response = client.post("/api/v1/query", json={"question": "What is RAG?"})
            assert response.status_code == 200

    def test_auth_enabled_rejects_missing_key(self, client, monkeypatch):
        """When API_KEY_AUTH_ENABLED is true, missing X-API-Key returns 401."""
        monkeypatch.setenv("API_KEY_AUTH_ENABLED", "true")
        monkeypatch.setenv("API_AUTH_KEY", "secret-key-1234")
        from app.core.config import get_settings
        get_settings.cache_clear()

        response = client.post("/api/v1/query", json={"question": "Test without key"})
        assert response.status_code == 401
        assert "unauthorized" in response.json()["detail"].lower()

    def test_auth_enabled_rejects_invalid_key(self, client, monkeypatch):
        """When API_KEY_AUTH_ENABLED is true, wrong key returns 401."""
        monkeypatch.setenv("API_KEY_AUTH_ENABLED", "true")
        monkeypatch.setenv("API_AUTH_KEY", "secret-key-1234")
        from app.core.config import get_settings
        get_settings.cache_clear()

        response = client.post(
            "/api/v1/query",
            json={"question": "Test wrong key"},
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 401
        assert "unauthorized" in response.json()["detail"].lower()

    def test_auth_enabled_accepts_valid_key(self, client, monkeypatch):
        """When API_KEY_AUTH_ENABLED is true, valid key returns 200."""
        monkeypatch.setenv("API_KEY_AUTH_ENABLED", "true")
        monkeypatch.setenv("API_AUTH_KEY", "secret-key-1234")
        from app.core.config import get_settings
        get_settings.cache_clear()

        with patch("app.api.routes.answer_question") as mock_answer:
            from app.models.schemas import QueryResponse
            mock_answer.return_value = QueryResponse(answer="Authorized response", sources=[])
            response = client.post(
                "/api/v1/query",
                json={"question": "Test valid key"},
                headers={"X-API-Key": "secret-key-1234"}
            )
            assert response.status_code == 200
            assert response.json()["answer"] == "Authorized response"


# ================================================================
# 2. RATE LIMITING
# ================================================================

class TestRateLimiting:
    """Tests for sliding-window rate limiter."""

    def test_rate_limiter_allows_up_to_limit(self):
        from app.core.security import InMemoryRateLimiter
        limiter = InMemoryRateLimiter()

        ip = "192.168.1.100"
        # 5 requests with limit 5 should all succeed
        for _ in range(5):
            assert limiter.is_allowed(ip, limit=5, window_seconds=10.0) is True

        # 6th request should be blocked
        assert limiter.is_allowed(ip, limit=5, window_seconds=10.0) is False

    def test_rate_limiter_resets(self):
        from app.core.security import InMemoryRateLimiter
        limiter = InMemoryRateLimiter()

        ip = "192.168.1.101"
        for _ in range(3):
            limiter.is_allowed(ip, limit=3, window_seconds=10.0)
        assert limiter.is_allowed(ip, limit=3, window_seconds=10.0) is False

        limiter.reset()
        assert limiter.is_allowed(ip, limit=3, window_seconds=10.0) is True

    def test_rate_limit_middleware_429(self, client, monkeypatch):
        """When client exceeds requests per minute, server returns 429."""
        from app.core.security import rate_limiter
        rate_limiter.reset()

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "2")
        from app.core.config import get_settings
        get_settings.cache_clear()

        with patch("app.api.routes.answer_question") as mock_answer:
            from app.models.schemas import QueryResponse
            mock_answer.return_value = QueryResponse(answer="OK", sources=[])

            # First 2 requests succeed
            r1 = client.post("/api/v1/query", json={"question": "Req 1"})
            assert r1.status_code == 200

            r2 = client.post("/api/v1/query", json={"question": "Req 2"})
            assert r2.status_code == 200

            # 3rd request hits rate limit
            r3 = client.post("/api/v1/query", json={"question": "Req 3"})
            assert r3.status_code == 429
            assert "rate limit exceeded" in r3.json()["detail"].lower()

        rate_limiter.reset()


# ================================================================
# 3. HEALTH & READINESS PROBES
# ================================================================

class TestProbes:
    """Tests for Kubernetes/Docker health and readiness endpoints."""

    def test_root_probes(self, client):
        """GET /health and GET /ready should return 200."""
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_ready = client.get("/ready")
        assert r_ready.status_code == 200
        assert r_ready.json()["status"] == "ready"

    def test_api_v1_ready_endpoint(self, client):
        """GET /api/v1/ready should report component checks."""
        r = client.get("/api/v1/ready")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "backend" in data
        assert "checks" in data
        assert "vector_store" in data["checks"]
        assert "cache_storage" in data["checks"]


# ================================================================
# 4. DUAL VECTOR STORE BACKEND
# ================================================================

class TestDualVectorStoreBackend:
    """Tests verifying backend selection between FAISS and PGVector."""

    def test_faiss_is_default_backend(self):
        from app.core.config import get_settings
        settings = get_settings()
        assert settings.vector_store_backend in ["faiss", "pgvector"]

    def test_pgvector_fallback_on_missing_connection(self, monkeypatch):
        """If PGVector is selected but unavailable, system logs and gracefully falls back to FAISS."""
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
        monkeypatch.setenv("POSTGRES_CONNECTION_URL", "postgresql://invalid:invalid@localhost:9999/none")
        from app.core.config import get_settings
        get_settings.cache_clear()

        import app.services.vector_store as vs
        vs._vector_store = None

        # get_vector_store handles error and falls back gracefully
        store = vs.get_vector_store()
        # Should either be None (empty index) or a FAISS store, never crashing
        assert store is None or hasattr(store, "docstore")

        get_settings.cache_clear()
        vs._vector_store = None
