# =============================================================================
# Dockerfile — FastAPI Backend for Semantic Caching & Cost-Aware Routing RAG
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and build wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime Stage ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application source code
COPY . /app

# Ensure runtime directories exist with proper permissions
RUN mkdir -p /app/data/pdfs /app/faiss_index /app/cache_data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
