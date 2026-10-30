# AI Engineering Interview Preparation & Technical Deep-Dive

**Project**: Semantic Caching and Cost-Aware Routing for a RAG System  
**Author**: Srinivas  
**Role Target**: Senior AI Engineer / Machine Learning Systems Engineer / LLM Application Architect

---

## 1. The 60-Second Elevator Pitch

> *"In enterprise RAG systems, redundant queries and one-size-fits-all model allocation cause ballooning inference bills and slow p95 latencies. I engineered a production-grade RAG system combining **Semantic Caching**, **Cost-Aware Heuristic Model Routing**, and **Hybrid Search**.
>
> On repeated or semantically equivalent questions, our semantic cache intercepts queries at cosine similarity $\ge 0.90$, dropping response latency from $156.8\text{ ms}$ to $14.2\text{ ms}$ ($> 11\times$ speedup) and eliminating 100% of LLM token costs. On cache misses, a sub-millisecond heuristic classifier routes simple lookups to `gpt-3.5-turbo` and complex synthesis to `gpt-4o-mini`. In empirical benchmarks, this reduced inference costs by **11.54%** and cut LLM calls by **26.67%** without degrading answer quality (maintaining $0.95$ Token F1 and $> 91\%$ faithfulness). The system supports both FAISS and PostgreSQL `pgvector`, includes sliding-window rate limiting, and is fully containerized with Docker."*

---

## 2. The 3-Minute Architecture Explanation

> *"The system is designed as a modular, three-tier architecture:
>
> 1. **Gateway & Security**: Requests hit FastAPI via REST endpoints or the Streamlit web dashboard. A custom sliding-window middleware throttles requests per IP address, while an optional `X-API-Key` dependency enforces client authentication.
>
> 2. **Session Memory & Semantic Cache**: Follow-up questions pass through a conversation session manager where a `QueryRewriter` resolves ambiguous pronouns (*"When does it expire?"* $\to$ *"When does the semantic cache TTL expire?"*). The normalized query is embedded and compared against an in-memory vector cache using cosine similarity. To prevent stale answers when documents change, each cache entry is tagged with a SHA-256 fingerprint of the current document collection. If cosine similarity meets or exceeds $0.90$, the cached answer and provenance citations are returned immediately in $< 20\text{ ms}$.
>
> 3. **Hybrid Retrieval & Cost-Aware Routing**: On cache misses, the system executes hybrid retrieval: dense vector search from FAISS (or PostgreSQL `pgvector`) combined with native BM25Okapi sparse lexical search via Cormack Reciprocal Rank Fusion ($k=60$). The retrieved context and query are passed to our zero-overhead heuristic router. Instead of spending tokens on an LLM routing call, our router evaluates keyword synthesis signals, query length, multi-question conjunctions, and chunk volume in under $0.1\text{ ms}$. Scores under $0.45$ route to `gpt-3.5-turbo`; higher scores route to `gpt-4o-mini`.
>
> Every query generates a UUID trace tracking stage latencies, token consumption, and cost estimates, with sensitive credentials automatically sanitized before storage."*

---

## 3. Core Architectural Mechanisms

### Why Semantic Caching Instead of Exact String Caching?
Exact key-value caches (like Redis string matching) miss over $80\%$ of duplicate intent because users formulate identical questions with different words (e.g., *"What is the cache threshold?"* vs *"What similarity score triggers a cache hit?"*). Semantic caching embeds queries into vector space, enabling cosine similarity matching that captures true user intent.

### How is Cache Correctness Protected?
1. **High Similarity Threshold ($\tau \ge 0.90$)**: Prevents false-positive matches on nuanced differences.
2. **Collection-Level Fingerprinting**: A deterministic SHA-256 hash derived from active document names and chunk counts isolates cache namespaces. If a document is updated or deleted, the fingerprint changes, instantly isolating stale cache entries.
3. **Sliding Time-To-Live (24h TTL)**: Entries older than 86,400 seconds are purged automatically.

### How Does the Heuristic Router Work Without Latency Overhead?
Many multi-model systems use an LLM or cross-encoder to classify query intent, adding $200\text{--}400\text{ ms}$ of latency and additional cost. Our router is an analytical heuristic classifier calculating:
$$\text{Score} = 0.35 \cdot S_{\text{keywords}} + 0.25 \cdot S_{\text{length}} + 0.20 \cdot S_{\text{multipart}} + 0.20 \cdot S_{\text{chunks}}$$
Execution takes $< 0.1\text{ ms}$ and zero external API calls.

### Why Support Both FAISS and PostgreSQL `pgvector`?
- **FAISS**: Ideal for local development, rapid unit testing (102 tests pass in $< 6\text{s}$), and single-instance deployments with zero database overhead.
- **pgvector**: Enterprise requirement for durable cloud deployments, multi-replica horizontal scaling, and unified ACID transaction management.

---

## 4. Top 10 Technical Interview Questions & Answers

### Q1: What happens if a user submits a question with subtle differences that change the meaning? Does semantic caching produce hallucinations?
**Answer**: At a strict threshold of $0.90$ with normalized embeddings, only high semantic overlap triggers a hit. For example, *"What is RAG?"* and *"What is Retrieval-Augmented Generation?"* hit cache ($\text{sim} > 0.93$), whereas *"What are the weaknesses of RAG?"* falls below $0.78$ and correctly triggers a cache miss. Furthermore, our evaluation suite tested out-of-domain unanswerable queries and achieved a **$0.0\%$ false-positive hit rate**.

### Q2: Why did you implement Reciprocal Rank Fusion (RRF) instead of simple linear score combination?
**Answer**: Dense vector search (L2 distance or cosine similarity) and BM25 scores operate on entirely different scales and distributions (BM25 is unbounded $[0, \infty)$ while cosine is $[-1, 1]$). Linear combinations require score normalization that shifts across document updates. RRF operates purely on ordinal ranks ($1/(k + \text{rank})$), making it scale-invariant, robust to outliers, and parameter-free.

### Q3: How do you handle concurrency and race conditions when updating the vector store and cache?
**Answer**: Thread-safe operations are enforced via `threading.Lock()` mutexes around index mutation, document addition, and disk persistence. In multi-container production deployments, write operations are offloaded to PostgreSQL `pgvector` with standard row/table-level transaction locking.

### Q4: How are token costs calculated and tracked?
**Answer**: We implement standard OpenAI token-per-character heuristics during runtime (1 token $\approx 4$ characters) verified against model completion metadata. The `MetricsTracker` applies tiered pricing ($0.0005/1\text{k}$ input and $0.0015/1\text{k}$ output for `gpt-3.5-turbo`; $0.00015/1\text{k}$ and $0.0006/1\text{k}$ for `gpt-4o-mini`). When a cache hit occurs, the exact cost that would have been incurred is credited as `estimated_cost_saved_usd`.

### Q5: What is the failure mode if OpenAI's API experiences an outage?
**Answer**: In `rag_pipeline.py`, external LLM calls are wrapped in robust exception handlers. If an invocation times out or fails, the pipeline returns a graceful error response with HTTP 200, logs an alert, and records an error trace with zero tokens charged. Cached queries continue to be served without disruption.

### Q6: How do you prevent credential leakage in distributed tracing logs?
**Answer**: In `app/services/tracing.py`, a pre-persistence regex scrubber scans query strings, headers, and metadata for credential patterns (such as `sk-[A-Za-z0-9]{20,}` or bearer tokens), replacing them with `[REDACTED_API_KEY]`.

### Q7: Why use BM25 alongside dense vector search?
**Answer**: Dense semantic embeddings excel at conceptual understanding but struggle with exact alphanumeric tokens, product SKU codes, medical IDs, and acronyms. BM25 guarantees that rare keyword matches are preserved in the retrieval candidate pool.

### Q8: What is your evaluation framework for answer quality?
**Answer**: We evaluate along the **RAG Triad**:
1. **Faithfulness**: Token overlap and citation presence against context chunks.
2. **Answer Relevance**: Semantic similarity between answer and question entities.
3. **Context Relevance**: Proportion of retrieved chunks containing query keywords.
We support both automated deterministic scoring and `gpt-4o-mini` LLM-as-a-Judge evaluation.

### Q9: How does the sliding-window rate limiter work?
**Answer**: The `InMemoryRateLimiter` maintains a timestamp list per client IP. On each incoming request, timestamps older than 60 seconds are purged; if the count of valid timestamps reaches `rate_limit_requests_per_minute` (default 60), the request is rejected with HTTP 429 Too Many Requests and a `Retry-After: 60` header.

### Q10: If you had another 2 weeks on this project, what would you build next?
**Answer**:
1. Integrate a local Cross-Encoder reranker (such as `bge-reranker-large`) on the top-15 RRF candidates.
2. Deploy a distributed Redis vector cache (`RediSearch`) for horizontal multi-instance cache sharing.
3. Add asynchronous streaming responses (`StreamingResponse`) via Server-Sent Events (SSE) for lower perceived time-to-first-token.
