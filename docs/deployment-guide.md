# Production Deployment Guide

This guide details deployment options, persistent storage architecture, security configurations, and operational verification for the **Semantic Caching & Cost-Aware Routing RAG System**.

---

## 1. Architecture & Hosting Evaluation

### Runtime Characteristics
- **FastAPI Core Gateway**: Lightweight ASGI process listening on port 8000.
- **Streamlit Frontend**: Web dashboard listening on port 8501.
- **Vector Storage**:
  - *Option A (Default)*: Local FAISS files (`faiss_index/`) persisted to a block storage volume.
  - *Option B (Enterprise)*: PostgreSQL 16 with `pgvector` extension.
- **Semantic Cache Storage**: Local JSON vector store (`cache_data/`) persisted to disk volume.
- **External Dependencies**: OpenAI API (outbound HTTPS calls to `api.openai.com`).

### Platform Recommendations

| Hosting Option | Suitability | Persistent Storage Support | Estimated Cost | Setup Complexity |
|---|---|---|---|---|
| **Docker Compose (VPS / EC2 / Droplet)** | **Recommended (Portfolio & Production)** | Native Docker volumes mapped to persistent host disk | $5 - $12 / mo | Low (One command) |
| **Railway / Render** | **Recommended (PaaS)** | Persistent disk volumes attachable to services | $7 - $15 / mo | Very Low (Git push) |
| **AWS ECS / Fargate with EFS** | Enterprise Multi-AZ | AWS EFS volume mount for FAISS / RDS for pgvector | $30+ / mo | Moderate |
| **Google Cloud Run** | Serverless Microservices | Cloud Storage bucket / Cloud SQL pgvector | Pay-per-use | Moderate |

---

## 2. Docker Compose Deployment (Self-Hosted VPS)

The fastest and most reliable deployment for single-node production uses Docker Compose.

### Step 1: Provision Server & Clone Repository
```bash
git clone https://github.com/your-username/rag-system.git
cd rag-system
```

### Step 2: Configure Production Environment Variables
Create `.env` based on `.env.example`:
```bash
cp .env.example .env
```
Ensure the following variables are configured:
```ini
OPENAI_API_KEY=sk-proj-your-production-key-here
API_KEY_AUTH_ENABLED=true
API_AUTH_KEY=generate-a-strong-32-char-random-key
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
VECTOR_STORE_BACKEND=faiss
```

### Step 3: Launch Services
```bash
docker compose up -d --build
```

### Step 4: Verify Running Services
```bash
docker compose ps
```
Both `rag_backend`, `rag_frontend`, and `rag_postgres` should show `healthy` status.

---

## 3. Reverse Proxy & SSL Setup (Nginx + Let's Encrypt)

In production, never expose FastAPI or Streamlit directly over unencrypted HTTP. Place an Nginx reverse proxy in front with automatic HTTPS.

### Sample Nginx Configuration (`/etc/nginx/sites-available/rag-system`)
```nginx
server {
    listen 80;
    server_name rag.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name rag.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/rag.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rag.yourdomain.com/privkey.pem;

    # FastAPI REST API & Documentation
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs {
        proxy_pass http://localhost:8000;
    }

    # Streamlit Interactive Web Dashboard
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for Streamlit real-time updates
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

---

## 4. Production Security Checklist

- [ ] **API Key Header Enforced**: Set `API_KEY_AUTH_ENABLED=true` to require `X-API-Key` on mutating endpoints (`/query`, `/upload`, `/cache`).
- [ ] **Rate Limiting Active**: `RATE_LIMIT_ENABLED=true` throttles abusive IP addresses (returns HTTP 429).
- [ ] **Non-Root User**: Verified Docker containers execute under `appuser` (UID 1000).
- [ ] **Secrets Isolated**: Verified `.env` is listed in `.gitignore` and never committed.
- [ ] **Readiness Probes Active**: Container health monitors ping `/health` and `/ready`.
- [ ] **Persistent Volumes Bound**: Verified Docker named volumes `faiss_data` and `cache_data` persist across container restarts.

---

## 5. Automated Verification Checklist

Execute the smoke-test sequence against your deployed instance:
```bash
# 1. Check Root Health
curl -I https://rag.yourdomain.com/api/v1/health

# 2. Check Readiness Probe
curl -s https://rag.yourdomain.com/api/v1/ready | jq .

# 3. Test Authorized Query (if auth enabled)
curl -X POST https://rag.yourdomain.com/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-auth-key" \
  -d '{"question": "What is the architecture of this RAG system?"}' | jq .
```
