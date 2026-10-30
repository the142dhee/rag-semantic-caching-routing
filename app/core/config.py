"""
app/core/config.py
==================
PURPOSE: Central configuration hub for the entire application.

Settings are loaded automatically from .env or environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings for RAG system and Semantic Caching.
    """

    # --- API Keys ---
    openai_api_key: str = ""

    # --- Model Selection ---
    llm_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"

    # --- Cost-Aware Routing (Stage 4) ---
    # Models selected per routing tier (override via env vars)
    llm_model_simple: str = "gpt-3.5-turbo"   # fast & cheap — simple factual queries
    llm_model_complex: str = "gpt-4o-mini"    # capable — complex reasoning & synthesis
    # Complexity score threshold: 0.0=always simple, 1.0=always complex
    routing_threshold: float = 0.45
    # Set to False to disable routing and always use llm_model
    llm_routing_enabled: bool = True

    # --- Vector DB (FAISS) ---
    faiss_index_dir: str = "./faiss_index"

    # --- Semantic Cache Settings ---
    semantic_cache_enabled: bool = True
    similarity_threshold: float = 0.90
    cache_ttl_seconds: int = 86400  # 24 hours
    cache_persist_dir: str = "./cache_data"

    # --- Model Pricing (GPT-3.5 Turbo standard reference) ---
    # Cost per 1,000 tokens in USD
    cost_per_1k_input_tokens: float = 0.0005   # $0.50 / 1M tokens
    cost_per_1k_output_tokens: float = 0.0015  # $1.50 / 1M tokens

    # --- Legacy / Optional Chroma settings ---
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    # --- RAG Parameters ---
    top_k_results: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Stage 6: Advanced Retrieval & Multi-Turn Intelligence ---
    # Enable Hybrid Retrieval (FAISS Dense + BM25 Sparse with Reciprocal Rank Fusion)
    hybrid_search_enabled: bool = True
    hybrid_dense_weight: float = 0.5
    hybrid_sparse_weight: float = 0.5
    rrf_k: int = 60  # Standard RRF constant (Cormack et al.)

    # Multi-turn conversational query rewriting
    query_rewriting_enabled: bool = True

    # LLM-as-a-Judge model
    llm_judge_model: str = "gpt-4o-mini"

    # --- Stage 7: Production Security, Database & Deployment ---
    # API Key Authentication (header: X-API-Key)
    api_key_auth_enabled: bool = False
    api_auth_key: str = "rag-secret-api-key"

    # Rate Limiting (per-client requests per minute)
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60

    # Vector store backend: 'faiss' (default, zero-dependency) or 'pgvector' (production PostgreSQL)
    vector_store_backend: str = "faiss"
    postgres_connection_url: str = "postgresql://postgres:postgres@localhost:5432/rag_db"

    # Look for .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")



@lru_cache()
def get_settings() -> Settings:
    """Returns a cached Settings instance."""
    return Settings()
