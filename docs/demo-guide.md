# Technical Demonstration Guide & Walkthrough Script

**Project**: Semantic Caching and Cost-Aware Routing for a RAG System  
**Audience**: Recruiters, Hiring Managers, Technical Interviewers, and AI Engineering Evaluators  
**Estimated Walkthrough Time**: 5–7 minutes  
**Sample Document Used**: `data/pdfs/rag_architecture_handbook.pdf` (Published in repository, safe for public demo)

---

## Demonstration Setup

1. **Launch the Application**:
   ```bash
   # Terminal 1: Backend API
   uvicorn app.main:app --reload --port 8000

   # Terminal 2: Streamlit Dashboard
   streamlit run streamlit_app.py --server.port 8501
   ```
2. **Open Browser**: Navigate to `http://localhost:8501`.
3. **Verify Status**: The sidebar should display `🟢 Online | FAISS: X chunks | Cache: Y items`.

---

## Step-by-Step Live Demo Script

### Step 1: Upload and Ingest Sample PDF
- **Action**: In the left sidebar under **"📤 Document Upload"**, upload `data/pdfs/rag_architecture_handbook.pdf` and click **"Index PDF"**.
- **What Happens**:
  - The PDF is parsed page-by-page by `pypdf`.
  - The text is split into $800$-character chunks with $100$-character overlap.
  - Chunks are embedded and indexed in both FAISS and BM25Okapi.
- **Expected Observation**: Success alert appears: `Indexed rag_architecture_handbook.pdf (12 chunks)!`.

---

### Step 2: First Query — Cache Miss & Simple LLM Routing
- **Action**: In the chat box, type:
  ```text
  What is the semantic cache similarity threshold?
  ```
- **What Happens**:
  1. Semantic Cache checks for matching embeddings $\ge 0.90$ (None exist yet $\to$ **Cache Miss**).
  2. Hybrid retrieval fetches relevant chunks from FAISS and BM25 using Reciprocal Rank Fusion.
  3. Cost-Aware Router evaluates query complexity (Word count: 7, no complex keywords $\to$ Score: ~0.15 $\to$ **SIMPLE Tier**).
  4. Generative model `gpt-3.5-turbo` synthesizes the response with citations.
  5. Answer is saved to the semantic cache with collection fingerprint hash.
- **Expected Observation**:
  - Assistant responds with: *"The semantic cache similarity threshold is 0.90..."*
  - Provenance badge: `📄 rag_architecture_handbook.pdf (Page 2)`
  - Telemetry badges:
    - `<span class="llm-badge">🤖 LLM GENERATED</span>`
    - `<span class="tier-badge">🧭 SIMPLE TIER</span>`
    - `<span class="metric-pill">🧠 gpt-3.5-turbo</span>`
    - Latency: ~100–160 ms

---

### Step 3: Exact Repeat — Instant Semantic Cache Hit ($0.00 Cost)
- **Action**: Re-enter the exact same query:
  ```text
  What is the semantic cache similarity threshold?
  ```
- **What Happens**:
  - Query vector is compared against cached queries.
  - Cosine similarity is **1.0000** ($\ge 0.90$).
  - System bypasses FAISS retrieval and bypasses the LLM entirely!
- **Expected Observation**:
  - Instant response: `<span class="cache-hit-badge">⚡ CACHE HIT (Score: 1.0)</span>`
  - Latency: **< 20 ms** (Over $10\times$ speedup)
  - Cost avoided badge: `<span class="metric-pill">💰 Saved $0.0003</span>`

---

### Step 4: Semantic Paraphrase — Near-Identical Query Cache Hit
- **Action**: Enter a semantically identical question phrased differently:
  ```text
  What cosine similarity is required for the semantic cache to hit?
  ```
- **What Happens**:
  - Words differ, but the embedding cosine similarity exceeds **0.92**.
  - Cache hit is triggered without invoking vector search or generative LLM.
- **Expected Observation**:
  - `<span class="cache-hit-badge">⚡ CACHE HIT (Score: 0.93+)</span>`
  - Matched Query displayed: `"What is the semantic cache similarity threshold?"`
  - Zero token expenditure.

---

### Step 5: Complex Synthesis Query — Heuristic Router to Advanced Tier
- **Action**: Submit an analytical, multi-part synthesis question:
  ```text
  Compare the latency and cost trade-offs between cache hits and cache misses in detail, and analyze the architectural impact.
  ```
- **What Happens**:
  1. Semantic cache misses (different semantic meaning).
  2. Router identifies complex keywords (`compare`, `analyze`, `trade-offs`, `impact`) and high word count ($> 15$ words).
  3. Score $\ge 0.45$ $\to$ Routed to **COMPLEX Tier** (`gpt-4o-mini`).
  4. Model synthesizes deep comparative analysis citing multiple pages.
- **Expected Observation**:
  - `<span class="tier-badge">🧭 COMPLEX TIER</span>`
  - `<span class="metric-pill">🧠 gpt-4o-mini</span>`
  - Rich multi-point comparative markdown answer.

---

### Step 6: Multi-Turn Conversation — Context-Aware Query Rewriting
- **Action**: Submit a pronoun-dependent follow-up question:
  ```text
  When does it expire?
  ```
- **What Happens**:
  - `QueryRewriter` inspects session conversation history.
  - Resolves *"it"* to the semantic cache TTL discussed in previous turns.
  - Reformulates query to: *"When does the semantic cache TTL expire?"*.
- **Expected Observation**:
  - Badge appears: `<span class="metric-pill">🔄 Rewritten: "When does the semantic cache TTL expire?"</span>`
  - Accurate, grounded answer explaining the 86,400s (24h) TTL without confusion.

---

### Step 7: Inspect Production Observability & Audit Traces
- **Action**: Click on **"📈 Observability & Metrics"** tab, then **"🔍 Request Traces"** tab.
- **Expected Observations**:
  - Real-time KPI gauges: Total requests, Cache Hit Rate (~50% during demo), LLM calls avoided, cumulative dollar savings.
  - Full audit trail table displaying Request UUID, timestamp, latency breakdown, tokens used, and model assigned.
  - Click any trace to view full request metadata with credentials automatically redacted.

---

### Step 8: Scientific Benchmark & Evaluation Panel
- **Action**: Click on **"🔬 Benchmark & Evaluation"** tab.
- **Expected Observations**:
  - Side-by-side comparison between `BASELINE_RAG` and `OPTIMIZED_RAG`.
  - Verified empirical metrics: **26.67% hit rate**, **11.54% cost savings**, **14.2ms vs 156.8ms latency**.
  - Bar charts and latency distribution curves.
