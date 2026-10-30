# AI Engineering Resume Bullet Options

These bullet points are formatted following the Google XYZ formula: *"Accomplished [X], as measured by [Y], by doing [Z]"*. All metrics are supported by the empirical Stage 5 benchmarks recorded in this repository.

---

### Option 1: Architecture & Systems Engineering Focus
> **AI Systems Engineer | Semantic Caching & Cost-Aware Routing RAG**  
> * Engineered an enterprise RAG system in FastAPI and Python 3.11 combining vector semantic caching and heuristic model routing, achieving an **11.54% reduction in inference costs** and cutting **LLM calls by 26.67%** across benchmark workloads.  
> * Implemented sub-20ms semantic caching ($\ge 0.90$ cosine similarity) with SHA-256 collection fingerprinting for zero-latency invalidation, delivering an **$11.04\times$ latency acceleration (14.2ms vs 156.8ms)** on repeated and paraphrased queries.  
> * Designed a hybrid retrieval pipeline fusing dense vector search (FAISS / PostgreSQL pgvector) and sparse BM25Okapi using Reciprocal Rank Fusion ($k=60$), maintaining $0.95$ Token F1 and $> 91\%$ answer faithfulness.

---

### Option 2: Performance, Cost & Latency Optimization Focus
> **LLM Application Architect | Cost-Optimized RAG Platform**  
> * Designed a zero-overhead ($< 0.1\text{ ms}$) heuristic query router that dynamically classifies query complexity to route simple lookups to `gpt-3.5-turbo` and complex synthesis to `gpt-4o-mini`, optimizing token expenditure without sacrificing synthesis depth.  
> * Integrated automated RAG Triad evaluation (LLM-as-a-Judge for Faithfulness, Answer Relevance, and Context Relevance) and end-to-end UUID tracing with automated credential scrubbing and sliding-window rate limiting.  
> * Containerized multi-service deployment with Docker Compose orchestrating FastAPI, Streamlit, and pgvector PostgreSQL with non-root security and automated GitHub Actions CI covering 102 unit/integration tests.

---

### Option 3: Concise Single-Bullet Summary (for Generalist AI / ML Resume)
> **AI Engineer — Retrieval-Augmented Generation (RAG) System**: Architected a production-ready PDF RAG system in FastAPI and Streamlit featuring vector semantic caching ($\ge 0.90$ threshold) and cost-aware model routing, reducing token costs by **11.54%** and accelerating query latency by **$11\times$ (14.2ms vs 156.8ms)** with dual FAISS/pgvector backends, sliding-window rate limiting, and 100% test coverage across 102 pytest suites.
