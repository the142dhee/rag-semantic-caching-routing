# Evaluation Methodology & Dataset Specification

## 1. Evaluation Dataset Overview

The evaluation suite (`data/eval_dataset.json`) contains **15 realistic test questions** grounded in the canonical knowledge base `data/pdfs/rag_architecture_handbook.pdf`.

### Question Categories

| Category | Count | Purpose | Expected Behavior |
|---|---|---|---|
| `simple_factual` | 3 | Direct definition & single-fact lookups | Routes to SIMPLE tier; exact citations |
| `semantic_variation` | 2 | Paraphrases of previously answered queries | Triggers semantic cache hit ($\text{sim} \ge 0.90$) |
| `complex_question` | 2 | Multi-concept analysis, architectural comparisons | Routes to COMPLEX tier; deep synthesis |
| `multi_chunk` | 2 | Requires information spread across multiple pages | Synthesizes multiple chunks with distinct citations |
| `unanswerable` | 2 | Out-of-domain / unmentioned topics | Grounded refusal: *"I don't have enough information"* |
| `routing_test` | 2 | Target edge-cases around the 0.45 complexity score | Confirms accurate SIMPLE vs COMPLEX split |
| `repeated_question` | 2 | Exact identical duplicate queries | 100% cache hit rate; sub-20ms latency |

---

## 2. Evaluation Metrics Definition

### A. Retrieval Metrics
1. **Precision@K**:
   $$\text{Precision@K} = \frac{|\text{Retrieved Chunks in Top-K} \cap \text{Relevant Chunks}|}{K}$$
   Measures how clean the retrieved context is and whether non-relevant noise is excluded.

2. **Recall@K**:
   $$\text{Recall@K} = \frac{|\text{Retrieved Chunks in Top-K} \cap \text{Relevant Chunks}|}{|\text{Relevant Chunks}|}$$
   Measures whether all necessary evidence was retrieved to answer the question.

3. **Retrieval Latency**:
   Milliseconds taken by FAISS to perform vector distance calculation.

---

### B. Answer Quality Metrics
1. **Answer Correctness (Token F1)**:
   Measures the harmonic mean of precision and recall between the predicted answer tokens and the reference ground truth tokens (SQuAD standard):
   $$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

2. **Faithfulness & Groundedness**:
   - For answerable queries: Measures the fraction of content tokens in the generated answer supported by retrieved context.
   - For unanswerable queries: Evaluates whether the system successfully refused to answer ($1.0$) or hallucinated ($0.0$).

3. **Context Relevance**:
   The percentage of retrieved chunks that contain semantic tokens relevant to the user query.

4. **Answer Completeness**:
   Coverage of key entities and statements from the reference answer.

---

### C. System Telemetry Metrics
- **Mean Latency**, **p50 Latency**, **p90 Latency**, **p95 Latency**: Measured using `numpy.percentile`.
- **Cache Hit Rate (%)**: $\frac{\text{Cache Hits}}{\text{Total Requests}} \times 100$.
- **Operating Cost ($USD)**: Calculated based on token pricing ($0.0005/1k input, $0.0015/1k output).
- **Cost Reduction (%)**: $\frac{\text{Baseline Cost} - \text{Optimized Cost}}{\text{Baseline Cost}} \times 100$.
- **Latency Reduction (%)**: $\frac{\text{Baseline Latency} - \text{Optimized Latency}}{\text{Baseline Latency}} \times 100$.
