# Contributing to Semantic Caching & Cost-Aware Routing RAG

Thank you for your interest in contributing to this project! We welcome community contributions, bug reports, performance enhancements, and documentation improvements.

---

## Code of Conduct

Please maintain a respectful, welcoming, and inclusive environment. Constructive criticism and technical collaboration are valued.

---

## Development Workflow

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `uv`)
- Git

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/your-username/rag-system.git
cd rag-system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and supply your OPENAI_API_KEY
```

### 3. Running the Development Servers
```bash
# Terminal 1: FastAPI Backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit Dashboard
streamlit run streamlit_app.py --server.port 8501
```

### 4. Running Tests
We enforce 100% pass rates across unit and integration suites:
```bash
pytest -v
```

---

## Coding Standards

1. **Type Hints**: All functions and service signatures must include Python type annotations.
2. **Docstrings**: Maintain concise docstrings explaining the purpose, arguments, and return types.
3. **No Hardcoded Secrets**: Never commit API keys, tokens, or personal identifiers.
4. **Error Handling**: Gracefully handle missing files, empty inputs, and service timeouts. Return appropriate HTTP status codes (400, 401, 404, 429, 500).

---

## Pull Request Process

1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/my-feature-name
   ```
2. Write unit tests for new functionality in `tests/`.
3. Verify all tests pass locally:
   ```bash
   pytest -q
   ```
4. Commit your changes with clear, imperative messages:
   ```bash
   git commit -m "feat(retrieval): add cross-encoder reranking support"
   ```
5. Push to your fork and submit a Pull Request targeting `main`.
