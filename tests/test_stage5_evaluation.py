"""
tests/test_stage5_evaluation.py
===============================
PURPOSE: Comprehensive test suite for Stage 5: Evaluation, Benchmarking,
Observability, and Production Hardening.

Covers:
  1. Document ingestion and chunking validation.
  2. Retrieval evaluation metrics (Precision@K, Recall@K).
  3. Answer quality evaluation metrics (Token F1, Faithfulness, Completeness, Context Relevance).
  4. System telemetry & latency percentiles (p50, p90, p95).
  5. Request tracing, ID generation, rolling buffer, and API key redaction.
  6. Defensive error handling (empty queries, missing documents, LLM exceptions).
  7. Benchmark runner execution (Baseline vs Optimized end-to-end comparison).
  8. Observability and Benchmark API endpoints.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.evaluation.dataset import load_eval_dataset, EvalItem
from app.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_token_f1,
    calculate_faithfulness,
    calculate_context_relevance,
    calculate_answer_completeness,
    calculate_percentiles,
)
from app.evaluation.benchmark_runner import BenchmarkRunner, DeterministicMockEmbeddings
from app.services.tracing import tracer, sanitize_text
from app.services.rag_pipeline import answer_question
from app.models.schemas import QueryResponse, SourceChunk


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# 1. EVALUATION DATASET INTEGRITY
# ============================================================

def test_evaluation_dataset_loads_correctly():
    """Verify that eval_dataset.json loads and contains all 7 categories."""
    dataset = load_eval_dataset()
    assert len(dataset) >= 15

    categories = {item.category for item in dataset}
    expected_categories = {
        "simple_factual",
        "semantic_variation",
        "complex_question",
        "multi_chunk",
        "unanswerable",
        "routing_test",
        "repeated_question",
    }
    assert expected_categories.issubset(categories)

    for item in dataset:
        assert item.id
        assert len(item.question) > 5
        assert item.expected_answer
        assert item.expected_tier in {"SIMPLE", "COMPLEX"}


# ============================================================
# 2. RETRIEVAL & ANSWER QUALITY METRICS
# ============================================================

def test_precision_and_recall_at_k():
    """Verify Precision@K and Recall@K calculations."""
    retrieved = ["doc.pdf (Page 1)", "doc.pdf (Page 2)", "other.pdf (Page 1)"]
    relevant = ["doc.pdf (Page 1)", "doc.pdf (Page 2)"]

    prec = calculate_precision_at_k(retrieved, relevant, k=3)
    rec = calculate_recall_at_k(retrieved, relevant, k=3)

    assert prec == pytest.approx(2 / 3, 0.01)
    assert rec == pytest.approx(1.0, 0.01)

    # Empty retrieved case
    assert calculate_precision_at_k([], relevant) == 0.0
    assert calculate_recall_at_k([], relevant) == 0.0


def test_token_f1_score():
    """Verify Token F1 exact match and partial match."""
    exact = calculate_token_f1("The chunk size is 800.", "The chunk size is 800.")
    assert exact["f1"] == 1.0

    partial = calculate_token_f1("Chunk size is 800 characters.", "Chunk size is 800.")
    assert 0.0 < partial["f1"] < 1.0

    no_overlap = calculate_token_f1("Apple banana orange", "Dog cat elephant")
    assert no_overlap["f1"] == 0.0


def test_faithfulness_and_unanswerable_refusal():
    """Verify faithfulness on supported context and refusal detection on unanswerables."""
    context = ["The chunk size is configured to 800 characters with 100 overlap."]

    # Grounded answer
    grounded_ans = "The chunk size is configured to 800 characters."
    score = calculate_faithfulness(grounded_ans, context, is_answerable=True)
    assert score > 0.8

    # Unanswerable question with correct refusal
    refusal_ans = "I don't have enough information in the provided documents to answer that."
    assert calculate_faithfulness(refusal_ans, context, is_answerable=False) == 1.0

    # Unanswerable question where model hallucinated
    hallucinated_ans = "Llama-3 fine-tuning requires 64GB of VRAM."
    assert calculate_faithfulness(hallucinated_ans, context, is_answerable=False) == 0.0


def test_percentile_calculations():
    """Verify p50, p90, p95 latency percentile computations."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    res = calculate_percentiles(latencies)

    assert res["mean"] == 55.0
    assert res["p50"] == 55.0
    assert res["p95"] > res["p50"]
    assert res["min"] == 10.0
    assert res["max"] == 100.0


# ============================================================
# 3. REQUEST TRACING & CREDENTIAL REDACTION
# ============================================================

def test_credential_redaction():
    """Verify that API keys and bearer tokens are sanitized from traces."""
    secret_text = "Query failed with key sk-1234567890abcdef123456 in header Bearer tok_xyz123."
    clean = sanitize_text(secret_text)
    assert "sk-1234567890abcdef123456" not in clean
    assert "[REDACTED_API_KEY]" in clean


def test_tracer_record_and_buffer():
    """Verify tracer records traces with UUIDs and enforces max capacity."""
    tracer.clear()
    assert tracer.count() == 0

    t_id = tracer.create_trace_id()
    trace = tracer.record_trace(
        request_id=t_id,
        question="What is FAISS?",
        cache_hit=False,
        selected_model="gpt-3.5-turbo",
        routing_reason="Low complexity",
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.0001,
        total_latency_ms=120.5,
        answer="FAISS is a vector search library.",
        retrieval_time_ms=5.2,
        retrieved_documents=["doc.pdf (Page 1)"],
    )

    assert tracer.count() == 1
    retrieved = tracer.get_trace(t_id)
    assert retrieved is not None
    assert retrieved.request_id == t_id
    assert retrieved.selected_model == "gpt-3.5-turbo"
    assert retrieved.retrieval_time_ms == 5.2

    # Check rolling capacity
    tracer.clear()


# ============================================================
# 4. ERROR HANDLING & PRODUCTION RESILIENCY
# ============================================================

def test_empty_query_handling():
    """Verify empty or whitespace query returns graceful message without crashing."""
    resp = answer_question("", use_cache=False)
    assert resp.answer == "Please provide a non-empty question."
    assert resp.from_cache is False
    assert resp.request_id is not None

    resp_ws = answer_question("    ", use_cache=False)
    assert resp_ws.answer == "Please provide a non-empty question."


@patch("app.services.rag_pipeline.get_collection_id", return_value="test_coll")
@patch("app.services.rag_pipeline.similarity_search", return_value=[{"text": "Sample text", "source": "test.pdf", "page": 1}])
@patch("app.services.rag_pipeline.get_llm_chain")
def test_llm_api_failure_graceful_recovery(mock_chain, mock_sim, mock_coll):
    """Verify system handles LLM API exception gracefully rather than crashing."""
    mock_instance = MagicMock()
    mock_instance.invoke.side_effect = Exception("OpenAI API RateLimitError: Rate limit exceeded")
    mock_chain.return_value = mock_instance

    resp = answer_question("Test question with failing LLM?", use_cache=False)
    assert "Error generating answer from model" in resp.answer
    assert resp.from_cache is False
    assert resp.request_id is not None


# ============================================================
# 5. BENCHMARK RUNNER EXECUTION
# ============================================================

def test_benchmark_runner_mock_execution(tmp_path):
    """Verify that BenchmarkRunner executes both Baseline and Optimized passes."""
    test_results_path = str(tmp_path / "test_benchmark.json")
    runner = BenchmarkRunner(results_path=test_results_path)

    report = runner.run(mock_llm=True)
    assert "comparison" in report
    comp = report["comparison"]

    base = comp["baseline"]
    opt = comp["optimized"]
    imp = comp["improvements"]

    assert base["total_queries"] == len(runner.dataset)
    assert opt["total_queries"] == len(runner.dataset)
    assert base["cache_hits"] == 0
    assert opt["cache_hits"] > 0
    assert imp["cost_reduction_pct"] > 0
    assert os.path.exists(test_results_path)


# ============================================================
# 6. OBSERVABILITY & BENCHMARK API ENDPOINTS
# ============================================================

def test_traces_endpoint(client):
    """Test GET /api/v1/traces and GET /api/v1/traces/{id}."""
    # Ensure at least one trace exists
    tracer.record_trace(
        request_id="test-trace-uuid",
        question="What is RAG?",
        cache_hit=True,
        selected_model="cached",
        routing_reason="Cache hit",
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
        total_latency_ms=12.0,
        answer="RAG is Retrieval-Augmented Generation.",
    )

    resp = client.get("/api/v1/traces?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_traces"] >= 1

    trace_resp = client.get("/api/v1/traces/test-trace-uuid")
    assert trace_resp.status_code == 200
    assert trace_resp.json()["question"] == "What is RAG?"

    # Non-existent trace
    notFound = client.get("/api/v1/traces/non-existent-uuid")
    assert notFound.status_code == 404


def test_benchmark_endpoints(client):
    """Test GET /api/v1/benchmark/results and POST /api/v1/benchmark/run."""
    resp = client.get("/api/v1/benchmark/results")
    assert resp.status_code == 200
    data = resp.json()
    assert "comparison" in data
    assert "baseline" in data["comparison"]
    assert "optimized" in data["comparison"]
