# Changelog

All notable changes to this project are documented in this file, adhering to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-20

### Added
- **Stage 1 (System Architecture)**: Designed core RAG architecture with semantic cache and router specifications.
- **Stage 2 (Basic PDF RAG Pipeline)**:
  - Document ingestion via `PyPDF` with metadata preservation (filename, page numbers).
  - Configurable recursive character chunking ($800$ chars, $100$ overlap).
  - FAISS vector store integration with disk persistence.
  - LLM synthesis with grounded citations format: `[Doc: filename, Page X]`.
- **Stage 3 (Semantic Caching)**:
  - Vector cosine similarity cache ($\ge 0.90$ threshold).
  - SHA-256 collection-level cache fingerprinting for instant invalidation upon document changes.
  - Sub-20ms cache hit response times with 100% token cost avoidance.
- **Stage 4 (Cost-Aware Model Routing)**:
  - Zero-overhead heuristic complexity classifier ($< 1\text{ ms}$).
  - Multi-feature scoring based on question word count, keyword density, multi-part intent, and chunk count.
  - Dynamic routing to simple model (`gpt-3.5-turbo`) or complex reasoning model (`gpt-4o-mini`).
- **Stage 5 (Evaluation, Benchmarking & Observability)**:
  - 15-question evaluation dataset with ground-truth answers and citations.
  - Empirical benchmarking harness measuring Precision@K, Recall@K, Token F1, Faithfulness, and latency percentiles (p50, p90, p95).
  - End-to-end request tracing with API key sanitization.
- **Stage 6 (Advanced Retrieval & Multi-Turn Intelligence)**:
  - Native BM25Okapi lexical search engine.
  - Hybrid Search combining dense FAISS and sparse BM25 via Reciprocal Rank Fusion ($k=60$).
  - Multi-turn conversation session manager with contextual pronoun resolution and query rewriting.
  - Automated LLM-as-a-Judge RAG Triad framework (Faithfulness, Answer Relevance, Context Relevance on 1–5 scale).
- **Stage 7 (Deployment, Security & Production Hardening)**:
  - Dual vector database support (local FAISS and PostgreSQL + `pgvector`).
  - API Key authentication (`X-API-Key`) with configurable bypass for development.
  - Sliding-window IP rate limiter middleware with HTTP 429 response.
  - Component-level readiness probes (`/ready`).
  - Production multi-stage `Dockerfile` (non-root `appuser`) and `docker-compose.yml`.
  - GitHub Actions CI workflow with automated test runs and secret scanning.
  - Recruiter-ready technical documentation and interview preparation guides.
