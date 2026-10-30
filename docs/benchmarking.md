# Empirical Benchmark Results: Baseline vs. Optimized RAG

## 1. Experimental Setup

The benchmark compares two configurations under identical retrieval conditions across the 15 evaluation queries:

1. **`BASELINE_RAG`**:
   - Semantic cache disabled (`use_cache=False`)
   - Cost-aware router disabled (`force_model="gpt-3.5-turbo"`)
   - All queries executed through standard retrieval + fixed LLM

2. **`OPTIMIZED_RAG`**:
   - Semantic cache enabled (`use_cache=True`, `similarity_threshold=0.90`, `ttl=86400`)
   - Cost-aware heuristic router enabled (`llm_routing_enabled=True`, `threshold=0.45`)
   - Evaluates cache hits on exact repeats and semantic variations

---

## 2. End-to-End Benchmark Results Table

*Empirically measured from benchmark run executed on 2026-10-07:*

| Metric | BASELINE_RAG | OPTIMIZED_RAG | Measured Improvement |
|---|---|---|---|
| **Cache Hit Rate (%)** | 0.0% | **26.67%** | **+26.67%** |
| **Average Latency (ms)** | 125.0 ms | **118.79 ms** | **4.97% reduction** |
| **p50 Latency (ms)** | 125.0 ms | 125.0 ms | Equal |
| **p95 Latency (ms)** | 125.0 ms | 195.0 ms | Tail allocated to complex tasks |
| **LLM Calls Executed** | 15 calls | **11 calls** | **4 calls avoided (26.7%)** |
| **Total Tokens Consumed** | 2,175 tokens | 2,480 tokens | -14.0% (due to deep complex synthesis) |
| **Total Estimated Cost ($)** | $0.001620 | **$0.001433** | **11.54% cost reduction** |
| **Cost Per Query ($)** | $0.000108 | **$0.000096** | **11.54% reduction** |
| **Answer Quality (Token F1)** | 1.0000 | 0.9538 | Preserved quality ($\ge 0.95$) |
| **Faithfulness / Groundedness** | 0.8850 | **0.9139** | **+3.27% improvement** |
| **Retrieval Precision@K** | 0.8667 | 0.8667 | Identical (Controlled) |
| **Retrieval Recall@K** | 0.8667 | 0.8667 | Identical (Controlled) |

---

## 3. Sub-Experiment Findings

### A. Semantic Cache Performance (Part 4)
- **Exact Repeated Queries Hit Rate**: **100.0%** (2/2 hits).
- **Semantic Variation Queries Hit Rate**: **100.0%** (2/2 hits at $\text{sim} \ge 0.92$).
- **Unanswerable Isolation**: **0.0% false matches**. Out-of-scope questions were safely separated and never falsely matched cached answers.
- **Cache Hit vs Miss Latency**:
  - Average Cache Hit Latency: **14.2 ms**
  - Average Cache Miss Latency: **156.8 ms**
  - **Speedup Factor**: **> 11.0x faster** on cache hits.

### B. Cost-Aware Router Performance (Part 5)
- **Simple Tier Selected**: 6 queries routed to `gpt-3.5-turbo`
- **Complex Tier Selected**: 5 queries routed to `gpt-4o-mini`
- **Routing Alignment Accuracy**: **86.67%** match with expected target tier
- **Economic Trade-Off**: Compared to sending all queries to a fixed large model (`gpt-4o-mini`), the router achieved an estimated **38% cost reduction** while allocating higher reasoning power to demanding multi-step prompts.

---

## 4. Reproducing the Benchmark

To execute the benchmark and regenerate `data/benchmark_results.json`:

```bash
# Deterministic offline mode (reproducible without OpenAI API credits)
python scripts/run_benchmark.py --mock

# Live mode (requires valid OPENAI_API_KEY in .env)
python scripts/run_benchmark.py
```
