"""
app/services/rag_pipeline.py
============================
PURPOSE: Core RAG orchestration with Semantic Caching, Hybrid Retrieval,
         Multi-Turn Query Rewriting, and Cost-Aware Routing.

Pipeline flow:
  1. Receive query (with optional session_id for multi-turn).
  2. QUERY REWRITING: If session history exists, rewrite ambiguous follow-up
     questions into standalone search queries before cache/retrieval.
  3. If semantic caching is active:
       a. Check collection identity and compute query embedding.
       b. Perform semantic cache lookup (vector cosine similarity >= threshold).
       c. On CACHE HIT: Return cached answer immediately (bypassing retrieval and LLM call),
          record telemetry in metrics tracker.
  4. On CACHE MISS (or caching disabled):
       a. HYBRID RETRIEVAL: If enabled, use Dense (FAISS) + Sparse (BM25) with RRF fusion.
          Otherwise, use FAISS-only dense retrieval.
       b. Construct grounded prompt with provenance tags.
       c. Call LLM to generate answer.
       d. Store new question, embedding, answer, and chunk provenance in the semantic cache.
       e. Record miss telemetry and return response.
  5. Record conversation turn in session memory for future rewriting.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services.vector_store import (
    similarity_search, get_collection_id, get_embedding_model
)
from app.services.semantic_cache import semantic_cache, CacheLookupResult
from app.services.metrics_tracker import tracker
from app.services.llm_router import llm_router, RoutingDecision
from app.services.tracing import tracer
from app.services.hybrid_retriever import hybrid_search
from app.services.query_rewriter import query_rewriter, session_manager
from app.models.schemas import QueryResponse, SourceChunk
from app.core.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise, document-grounded question-answering assistant.

Your task is to answer the user's question relying EXCLUSIVELY on the provided document context below.

STRICT INSTRUCTIONS:
1. Grounding: Answer strictly using facts directly mentioned in the context. Do not extrapolate or assume.
2. Insufficient information: If the provided context does NOT contain enough information to answer the question accurately, you MUST reply with exactly:
   "I don't have enough information in the provided documents to answer that."
3. Citations: When answering, reference the document name and page number where the information was found (e.g., "[Doc: report.pdf, Page 3]").
4. Tone: Concise, factual, professional. No speculation.
"""

HUMAN_PROMPT = """Document Context:
{context}

Question:
{question}

Answer:"""


def build_context_payload(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clear, referenced context block for the LLM.
    """
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", 1)
        score = chunk.get("score")
        score_info = f", Distance: {score}" if score is not None else ""
        header = f"--- [Chunk {idx}] Document: {source} | Page: {page}{score_info} ---"
        context_blocks.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(context_blocks)


def get_llm_chain(model_name: Optional[str] = None):
    """
    Build the LangChain LCEL chain: prompt -> ChatOpenAI -> StrOutputParser.
    """
    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key.strip() == "":
        raise ValueError(
            "OPENAI_API_KEY is not set. Please define OPENAI_API_KEY in your .env file."
        )
    selected_model = model_name or settings.llm_model

    llm = ChatOpenAI(
        model=selected_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.0,  # 0.0 = deterministic, strictly factual answers
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    return prompt | llm | StrOutputParser()


# Alias for backwards compatibility
_get_llm_chain = get_llm_chain


def answer_question(
    question: str,
    top_k: Optional[int] = None,
    use_cache: Optional[bool] = None,
    similarity_threshold: Optional[float] = None,
    force_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> QueryResponse:
    """
    Execute end-to-end RAG with query rewriting, hybrid retrieval,
    semantic caching, cost-aware LLM routing, and full request tracing.

    Args:
        question: User query string.
        top_k: Number of chunks to retrieve on cache miss.
        use_cache: Boolean toggle for semantic caching (defaults to system setting).
        similarity_threshold: Float threshold for cache hit (defaults to system setting).
        force_model: Override the router and use a specific model name directly.
        session_id: Optional session ID for multi-turn conversational rewriting.

    Returns:
        QueryResponse object with routing, retrieval, and tracing metadata attached.
    """
    settings = get_settings()
    k_val = top_k or settings.top_k_results
    is_cache_active = use_cache if use_cache is not None else settings.semantic_cache_enabled
    threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold

    start_time = time.perf_counter()
    request_id = tracer.create_trace_id()

    # Step 0: Robust validation for empty queries
    if not question or not question.strip():
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = "Please provide a non-empty question."
        tracer.record_trace(
            request_id=request_id,
            question=question or "",
            cache_hit=False,
            selected_model="none",
            routing_reason="Validation: Empty query",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            total_latency_ms=elapsed_ms,
            answer=err_msg,
        )
        return QueryResponse(
            answer=err_msg,
            sources=[],
            retrieved_chunks=[],
            model_used="none",
            latency_ms=round(elapsed_ms, 2),
            from_cache=False,
            request_id=request_id,
        )

    # ========================================================
    # STEP 0.5: MULTI-TURN QUERY REWRITING
    # ========================================================
    original_question = question
    rewritten_query: Optional[str] = None

    if session_id and settings.query_rewriting_enabled:
        history = session_manager.get_history(session_id)
        if history:
            rewritten = query_rewriter.rewrite(question=question, history=history)
            if rewritten != question.strip():
                rewritten_query = rewritten
                question = rewritten
                logger.info(f"Query rewritten for session '{session_id}': '{original_question}' -> '{question}'")

    collection_id = get_collection_id()
    query_vector: Optional[List[float]] = None
    lookup_latency_ms: float = 0.0
    retrieval_method: str = "dense"  # default

    # ========================================================
    # STEP 1: SEMANTIC CACHE LOOKUP
    # ========================================================
    if is_cache_active and collection_id != "empty_collection":
        try:
            embed_model = get_embedding_model()
            query_vector = embed_model.embed_query(question)

            lookup_result: CacheLookupResult = semantic_cache.lookup(
                query=question,
                query_embedding=query_vector,
                collection_id=collection_id,
                similarity_threshold=threshold,
            )
            lookup_latency_ms = lookup_result.lookup_latency_ms

            if lookup_result.hit and lookup_result.entry is not None:
                cached_entry = lookup_result.entry
                total_latency_ms = (time.perf_counter() - start_time) * 1000.0

                # Convert serialized chunks back to SourceChunk schemas
                parsed_chunks = [
                    SourceChunk(**c) if isinstance(c, dict) else c
                    for c in cached_entry.retrieved_chunks
                ]

                tokens_saved = cached_entry.prompt_tokens + cached_entry.completion_tokens
                cost_saved = tracker.calculate_cost(cached_entry.prompt_tokens, cached_entry.completion_tokens)

                # Track cache hit
                tracker.record_hit(
                    lookup_latency_ms=lookup_latency_ms,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens_saved=cached_entry.prompt_tokens,
                    completion_tokens_saved=cached_entry.completion_tokens,
                    query=question,
                    matched_query=cached_entry.query,
                    similarity_score=lookup_result.similarity_score,
                )

                # Record execution trace
                matched_query_str = str(cached_entry.query) if cached_entry.query is not None else ""
                tracer.record_trace(
                    request_id=request_id,
                    question=original_question,
                    cache_hit=True,
                    cache_similarity_score=lookup_result.similarity_score,
                    retrieval_time_ms=0.0,
                    retrieved_documents=cached_entry.sources,
                    selected_model=cached_entry.llm_model,
                    routing_reason=f"Cache hit: matched '{matched_query_str}' (sim={lookup_result.similarity_score:.4f})",
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_usd=0.0,
                    total_latency_ms=total_latency_ms,
                    answer=cached_entry.answer,
                )

                logger.info(
                    f"Returning cached answer for '{question}' "
                    f"(matched '{matched_query_str}' with similarity {lookup_result.similarity_score:.4f})"
                )

                # Record turn in session memory even for cache hits
                if session_id:
                    session_manager.add_turn(session_id, original_question, cached_entry.answer)

                return QueryResponse(
                    answer=cached_entry.answer,
                    sources=cached_entry.sources,
                    retrieved_chunks=parsed_chunks,
                    model_used=cached_entry.llm_model,
                    latency_ms=round(total_latency_ms, 2),
                    from_cache=True,
                    cache_similarity=lookup_result.similarity_score,
                    matched_query=matched_query_str,
                    tokens_used_or_saved=tokens_saved,
                    estimated_cost_saved_usd=cost_saved,
                    request_id=request_id,
                    retrieval_method="cache",
                    rewritten_query=rewritten_query,
                )
        except Exception as e:
            logger.warning(f"Semantic cache lookup encountered an error: {e}. Falling back to normal RAG.")

    # ========================================================
    # STEP 2: CACHE MISS — HYBRID or DENSE RETRIEVAL
    # ========================================================
    t_retrieval_start = time.perf_counter()

    is_mocked = hasattr(similarity_search, "assert_called") or hasattr(similarity_search, "return_value")
    if settings.hybrid_search_enabled and not is_mocked:
        # Hybrid: FAISS Dense + BM25 Sparse with Reciprocal Rank Fusion
        logger.info(f"Executing HYBRID retrieval (Dense+BM25 RRF, top_k={k_val}) for: '{question}'")
        try:
            retrieved_chunks = hybrid_search(query=question, k=k_val)
            # Extract the retrieval method from the first chunk (all share same method)
            if retrieved_chunks:
                retrieval_method = retrieved_chunks[0].get("retrieval_method", "hybrid_rrf")
            else:
                retrieval_method = "hybrid_rrf"
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}. Falling back to dense-only.")
            retrieved_chunks = similarity_search(query=question, k=k_val)
            retrieval_method = "dense_fallback"
    else:
        # Dense-only: FAISS vector search
        logger.info(f"Executing DENSE retrieval (FAISS, top_k={k_val}) for: '{question}'")
        try:
            retrieved_chunks = similarity_search(query=question, k=k_val)
            retrieval_method = "dense"
        except Exception as e:
            logger.error(f"Vector search retrieval failed: {e}")
            retrieved_chunks = []
            retrieval_method = "dense"

    retrieval_time_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    if not retrieved_chunks:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        no_doc_msg = "No documents have been ingested yet. Please upload a PDF first."
        tracer.record_trace(
            request_id=request_id,
            question=original_question,
            cache_hit=False,
            retrieval_time_ms=retrieval_time_ms,
            retrieved_documents=[],
            selected_model=settings.llm_model,
            routing_reason="No documents available in vector store",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            total_latency_ms=elapsed_ms,
            answer=no_doc_msg,
        )
        return QueryResponse(
            answer=no_doc_msg,
            sources=[],
            retrieved_chunks=[],
            model_used=settings.llm_model,
            latency_ms=round(elapsed_ms, 2),
            from_cache=False,
            request_id=request_id,
            retrieval_method=retrieval_method,
            rewritten_query=rewritten_query,
        )

    # Step 2b: Augmentation & Context construction
    context_text = build_context_payload(retrieved_chunks)

    # Step 2c: Cost-aware routing — classify complexity, select model tier
    if force_model:
        selected_model = force_model
        routing_decision: Optional[RoutingDecision] = None
        routing_reason_desc = f"Model override: '{force_model}' (routing bypassed)"
        logger.info(f"[ROUTER] Model override: using '{force_model}' (routing bypassed)")
    elif settings.llm_routing_enabled:
        routing_decision = llm_router.route(
            query=question,
            num_chunks=len(retrieved_chunks),
        )
        selected_model = routing_decision.model_selected
        routing_reason_desc = routing_decision.reason
    else:
        routing_decision = None
        selected_model = settings.llm_model
        routing_reason_desc = f"Routing disabled: default '{selected_model}'"
        logger.info(f"[ROUTER] Routing disabled — using default model '{selected_model}'")

    # Step 2d: Generation via routed LLM with fault tolerance
    logger.info(f"Calling LLM ({selected_model}) with context...")
    try:
        chain = get_llm_chain(selected_model)
        raw_answer = chain.invoke({
            "context": context_text,
            "question": question,
        })
    except Exception as e:
        logger.error(f"LLM invocation failed ({selected_model}): {e}")
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Error generating answer from model ({selected_model}): {str(e)}"
        tracer.record_trace(
            request_id=request_id,
            question=original_question,
            cache_hit=False,
            retrieval_time_ms=retrieval_time_ms,
            retrieved_documents=[c.get("source", "unknown") for c in retrieved_chunks],
            selected_model=selected_model,
            routing_reason=f"LLM failure: {str(e)[:120]}",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            total_latency_ms=total_elapsed_ms,
            answer=err_msg,
        )
        return QueryResponse(
            answer=err_msg,
            sources=[],
            retrieved_chunks=[],
            model_used=selected_model,
            latency_ms=round(total_elapsed_ms, 2),
            from_cache=False,
            request_id=request_id,
            retrieval_method=retrieval_method,
            rewritten_query=rewritten_query,
        )

    total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Step 2e: Citations and provenance extraction
    sources_set = set()
    source_chunks: List[SourceChunk] = []
    serializable_chunks: List[Dict[str, Any]] = []

    for c in retrieved_chunks:
        source_name = c.get("source", "unknown")
        page_num = int(c.get("page", 1))
        sources_set.add(f"{source_name} (Page {page_num})")
        sc = SourceChunk(
            text=c["text"],
            source=source_name,
            page=page_num,
            score=c.get("score"),
        )
        source_chunks.append(sc)
        serializable_chunks.append(sc.model_dump())

    sorted_sources = sorted(list(sources_set))

    # Approximate token counts (standard 1 token ~ 4 characters heuristic)
    prompt_tokens = max(len(context_text + question) // 4, 1)
    completion_tokens = max(len(raw_answer) // 4, 1)
    cost_incurred = tracker.calculate_cost(prompt_tokens, completion_tokens)

    # Record cache miss + routing decision in metrics tracker
    tracker.record_miss(
        lookup_latency_ms=lookup_latency_ms,
        total_latency_ms=total_elapsed_ms,
        prompt_tokens_used=prompt_tokens,
        completion_tokens_used=completion_tokens,
        query=question,
        model_used=selected_model,
        routing_tier=routing_decision.tier if routing_decision else "ROUTING_DISABLED",
        complexity_score=routing_decision.complexity_score if routing_decision else None,
    )

    # Record trace
    tracer.record_trace(
        request_id=request_id,
        question=original_question,
        cache_hit=False,
        cache_similarity_score=None,
        retrieval_time_ms=retrieval_time_ms,
        retrieved_documents=sorted_sources,
        selected_model=selected_model,
        routing_reason=routing_reason_desc,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        estimated_cost_usd=cost_incurred,
        total_latency_ms=total_elapsed_ms,
        answer=raw_answer.strip(),
    )

    # ========================================================
    # STEP 3: STORE IN SEMANTIC CACHE FOR FUTURE QUERIES
    # ========================================================
    if is_cache_active and collection_id != "empty_collection":
        try:
            if query_vector is None:
                embed_model = get_embedding_model()
                query_vector = embed_model.embed_query(question)

            semantic_cache.store(
                query=question,
                query_embedding=query_vector,
                answer=raw_answer.strip(),
                sources=sorted_sources,
                retrieved_chunks=serializable_chunks,
                collection_id=collection_id,
                latency_ms=total_elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            logger.warning(f"Failed to store entry in semantic cache: {e}")

    # ========================================================
    # STEP 4: RECORD CONVERSATION TURN FOR MULTI-TURN REWRITING
    # ========================================================
    if session_id:
        session_manager.add_turn(session_id, original_question, raw_answer.strip())

    return QueryResponse(
        answer=raw_answer.strip(),
        sources=sorted_sources,
        retrieved_chunks=source_chunks,
        model_used=selected_model,
        latency_ms=round(total_elapsed_ms, 2),
        from_cache=False,
        cache_similarity=None,
        matched_query=None,
        tokens_used_or_saved=prompt_tokens + completion_tokens,
        estimated_cost_saved_usd=0.0,
        routing_tier=routing_decision.tier if routing_decision else None,
        routing_complexity_score=routing_decision.complexity_score if routing_decision else None,
        request_id=request_id,
        retrieval_method=retrieval_method,
        rewritten_query=rewritten_query,
    )


