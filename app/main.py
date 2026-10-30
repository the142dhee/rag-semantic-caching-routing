"""
app/main.py
===========
PURPOSE: Application entry point — creates the FastAPI app and configures it.

THIS FILE:
  1. Creates the FastAPI app instance
  2. Registers our routes
  3. Configures CORS (so a browser frontend can call this API)
  4. Sets up logging
  5. Validates configuration at startup

FASTAPI LIFESPAN:
  The @asynccontextmanager lifespan function runs code:
    - BEFORE the server accepts requests (startup)
    - AFTER the server shuts down (cleanup)
  This is the right place to validate the API key exists, connect to DBs, etc.

RUNNING THE APP:
  From the project root directory, run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

  --reload: auto-restarts when you change a .py file (great for development)
  Then visit: http://localhost:8000/docs for the interactive API explorer
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings
from app.core.security import rate_limit_middleware

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
# Configure logging once at startup. This formats every log line as:
#   2026-09-28 14:30:00 | INFO     | app.services.rag_pipeline | Message here
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.

    WHAT THIS DOES AT STARTUP:
      - Validates the OpenAI API key exists (fails fast if missing)
      - Logs the configuration so you can verify settings at a glance

    yield separates startup (before) from shutdown (after) logic.
    """
    # --- STARTUP ---
    logger.info("=" * 60)
    logger.info("Starting RAG System API")
    logger.info("=" * 60)

    try:
        settings = get_settings()
        logger.info(f"LLM Model       : {settings.llm_model}")
        logger.info(f"Embedding Model : {settings.embedding_model}")
        logger.info(f"Vector DB Path  : {settings.chroma_persist_dir}")
        logger.info(f"Top-K Results   : {settings.top_k_results}")
        logger.info(f"Chunk Size      : {settings.chunk_size} chars")
        logger.info("API Key         : [set]" if settings.openai_api_key else "API Key: [MISSING!]")
        logger.info("=" * 60)
    except Exception as e:
        logger.critical(f"Configuration error at startup: {e}")
        logger.critical("Make sure you have a .env file with OPENAI_API_KEY set.")
        sys.exit(1)

    yield  # Server runs here, handling requests

    # --- SHUTDOWN ---
    logger.info("RAG System API shutting down.")


# ---------------------------------------------------------------------------
# CREATE FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG System API",
    description="""
A PDF-based Retrieval-Augmented Generation (RAG) chatbot with:
- Semantic caching (Stage 2)
- Cost-aware LLM routing (Stage 3)
- Metrics tracking (Stage 4)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",   # Swagger UI lives here
    redoc_url="/redoc", # ReDoc alternative docs
)

# ---------------------------------------------------------------------------
# CORS MIDDLEWARE
# ---------------------------------------------------------------------------
# Allows browser-based frontends (like our Streamlit dashboard in Stage 5)
# to call this API. In production, replace "*" with your actual domain.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request throttling & rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# ---------------------------------------------------------------------------
# INCLUDE ROUTES
# ---------------------------------------------------------------------------
# The prefix "/api/v1" means all our routes become:
#   /api/v1/health, /api/v1/ingest, /api/v1/query
# Versioning the API lets you add /api/v2 later without breaking existing clients
# ---------------------------------------------------------------------------
app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# ROOT ENDPOINTS & PROBES
# ---------------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    """Landing page — confirms the API is running."""
    return {
        "message": "RAG System API is running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "ready": "/ready",
    }


@app.get("/health", tags=["System"])
async def root_health():
    """Root health probe alias for container orchestration (Kubernetes/Docker)."""
    return {"status": "ok"}


@app.get("/ready", tags=["System"])
async def root_readiness():
    """Readiness probe checking backend components."""
    from app.services.vector_store import get_document_count
    try:
        get_document_count()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        return {"status": "not_ready", "error": str(e)}
