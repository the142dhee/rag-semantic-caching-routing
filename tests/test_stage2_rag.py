"""
tests/test_stage2_rag.py
========================
PURPOSE: Comprehensive test suite for Stage 2 Basic RAG Pipeline Implementation.

Covers:
  - Document ingestion (valid, empty, corrupt, zero-text)
  - Text chunking (configurable chunk_size, overlap, metadata preservation)
  - FAISS vector store (embedding, indexing, disk save/load persistence)
  - Top-k retrieval with similarity distance
  - Grounded answer generation (present context vs. absent context)
  - Citation accuracy (document names, page numbers)
  - FastAPI endpoints (upload, query, documents, health)
"""

import io
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from pypdf import PdfWriter


# ============================================================
# FIXTURES & HELPERS
# ============================================================

def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a real minimal in-memory PDF using pypdf."""
    from pypdf import PageObject
    writer = PdfWriter()
    for text in pages_text:
        # Create a blank page and add text
        page = PageObject.create_blank_page(width=300, height=300)
        # Note: blank page without text stream is fine for mock/extract tests,
        # but to test actual text extraction, pypdf writes standard text operators:
        writer.add_page(page)

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    """Set up clean temporary directories and test environment variables."""
    test_index_dir = str(tmp_path / "test_faiss_index")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testkey-stage2")
    monkeypatch.setenv("LLM_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    monkeypatch.setenv("FAISS_INDEX_DIR", test_index_dir)
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    monkeypatch.setenv("TOP_K_RESULTS", "3")

    from app.core.config import get_settings
    get_settings.cache_clear()

    # Reset vector store singleton
    import app.services.vector_store as vs_module
    vs_module._vector_store = None
    vs_module._embedding_model = None

    yield test_index_dir

    get_settings.cache_clear()
    vs_module._vector_store = None
    vs_module._embedding_model = None


@pytest.fixture
def sample_chunks():
    """Sample chunk documents with explicit metadata for testing."""
    return [
        Document(
            page_content="Antigravity RAG system uses FAISS for vector similarity search.",
            metadata={"source": "rag_guide.pdf", "page": 1},
        ),
        Document(
            page_content="Cost-aware routing balances queries between smaller and larger LLMs.",
            metadata={"source": "rag_guide.pdf", "page": 2},
        ),
        Document(
            page_content="Semantic caching stores embeddings of prior questions to avoid re-generating answers.",
            metadata={"source": "caching_whitepaper.pdf", "page": 4},
        ),
    ]


# ============================================================
# 1. DOCUMENT INGESTION & VALIDATION TESTS
# ============================================================

class TestDocumentIngestion:
    """Tests for PDF text extraction and error handling in pdf_loader.py."""

    def test_extract_text_preserves_page_numbers_and_source(self, mock_env):
        """Verify page-by-page extraction preserves 1-based page numbers and filename."""
        from app.services.pdf_loader import extract_text_from_pdf_stream

        with patch("app.services.pdf_loader.PdfReader") as MockPdfReader:
            mock_page1 = MagicMock()
            mock_page1.extract_text.return_value = "Content from page one."
            mock_page2 = MagicMock()
            mock_page2.extract_text.return_value = "Content from page two."

            mock_reader = MagicMock()
            mock_reader.pages = [mock_page1, mock_page2]
            MockPdfReader.return_value = mock_reader

            fake_stream = io.BytesIO(b"%PDF-1.4 simulated bytes")
            docs, total_pages = extract_text_from_pdf_stream(fake_stream, "manual.pdf")

            assert total_pages == 2
            assert len(docs) == 2
            assert docs[0].metadata["source"] == "manual.pdf"
            assert docs[0].metadata["page"] == 1
            assert docs[1].metadata["page"] == 2
            assert docs[0].page_content == "Content from page one."

    def test_empty_pdf_stream_raises_error(self, mock_env):
        """Empty byte streams must raise a clear ValueError."""
        from app.services.pdf_loader import extract_text_from_pdf_stream

        empty_stream = io.BytesIO(b"")
        with pytest.raises(ValueError, match="is empty"):
            extract_text_from_pdf_stream(empty_stream, "empty.pdf")

    def test_corrupt_pdf_raises_error(self, mock_env):
        """Invalid or corrupt PDF headers must raise ValueError."""
        from app.services.pdf_loader import extract_text_from_pdf_stream

        corrupt_stream = io.BytesIO(b"not a real pdf content header")
        with pytest.raises(ValueError, match="Invalid or corrupted"):
            extract_text_from_pdf_stream(corrupt_stream, "corrupt.pdf")

    def test_scanned_image_pdf_with_no_text_raises_error(self, mock_env):
        """PDFs with pages but 0 extractable text must notify user about scanned/image document."""
        from app.services.pdf_loader import extract_text_from_pdf_stream

        with patch("app.services.pdf_loader.PdfReader") as MockPdfReader:
            blank_page = MagicMock()
            blank_page.extract_text.return_value = "   "  # whitespace only
            mock_reader = MagicMock()
            mock_reader.pages = [blank_page]
            MockPdfReader.return_value = mock_reader

            fake_stream = io.BytesIO(b"%PDF-1.4 simulated")
            with pytest.raises(ValueError, match="no extractable text"):
                extract_text_from_pdf_stream(fake_stream, "scanned.pdf")


# ============================================================
# 2. TEXT CHUNKING TESTS
# ============================================================

class TestTextChunking:
    """Tests for chunking logic and metadata propagation."""

    def test_chunking_preserves_metadata(self, mock_env):
        """Chunks must retain the source document and page number of their parent page."""
        from app.services.pdf_loader import chunk_documents

        page_doc = Document(
            page_content="Sentence one. " * 50,  # ~700 characters
            metadata={"source": "report.pdf", "page": 5},
        )

        chunks = chunk_documents([page_doc], chunk_size=300, chunk_overlap=30)
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata["source"] == "report.pdf"
            assert c.metadata["page"] == 5
            assert len(c.page_content) <= 350  # respects size constraint


# ============================================================
# 3. FAISS VECTOR STORE & PERSISTENCE TESTS
# ============================================================

class TestFAISSVectorStore:
    """Tests for FAISS indexing, similarity retrieval, and disk persistence."""

    def test_add_and_retrieve_from_faiss(self, mock_env, sample_chunks):
        """Test adding documents to FAISS and retrieving top-k chunks with scores."""
        from langchain_core.embeddings import FakeEmbeddings
        import app.services.vector_store as vs

        fake_embed = FakeEmbeddings(size=1536)
        with patch.object(vs, "get_embedding_model", return_value=fake_embed):
            # Add documents
            added = vs.add_documents(sample_chunks)
            assert added == len(sample_chunks)
            assert vs.get_document_count() == len(sample_chunks)

            # Similarity search
            results = vs.similarity_search("How does routing work?", k=2)
            assert len(results) == 2
            assert "text" in results[0]
            assert "source" in results[0]
            assert "page" in results[0]
            assert "score" in results[0]

    def test_faiss_disk_persistence(self, mock_env, sample_chunks):
        """Verify FAISS index saves to disk and successfully reloads without re-embedding."""
        from langchain_core.embeddings import FakeEmbeddings
        import app.services.vector_store as vs

        fake_embed = FakeEmbeddings(size=1536)
        with patch.object(vs, "get_embedding_model", return_value=fake_embed):
            # Ingest and save
            vs.add_documents(sample_chunks)
            assert os.path.exists(os.path.join(mock_env, "index.faiss"))
            assert os.path.exists(os.path.join(mock_env, "index.pkl"))

            # Simulate application restart: clear in-memory singleton
            vs._vector_store = None

            # Reload from disk
            reloaded_vs = vs.get_vector_store()
            assert reloaded_vs is not None
            assert vs.get_document_count() == len(sample_chunks)

            # Check document summary
            summary = vs.get_indexed_documents()
            assert len(summary) == 2  # rag_guide.pdf (2 chunks) and caching_whitepaper.pdf (1 chunk)


# ============================================================
# 4. RAG PIPELINE & GROUNDING TESTS
# ============================================================

class TestRAGPipeline:
    """Tests for answer generation with evidence grounding and citation formatting."""

    def test_answer_when_info_is_present(self, mock_env):
        """When evidence is present, return answer with citations."""
        import app.services.rag_pipeline as rag

        fake_chunks = [
            {
                "text": "FAISS provides fast similarity search for dense vectors.",
                "source": "faiss_overview.pdf",
                "page": 3,
                "score": 0.15,
            }
        ]

        with patch("app.services.rag_pipeline.similarity_search", return_value=fake_chunks), \
             patch("app.services.rag_pipeline.get_llm_chain") as mock_chain_getter:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "FAISS is used for dense vector similarity search [Doc: faiss_overview.pdf, Page 3]."
            mock_chain_getter.return_value = mock_chain

            response = rag.answer_question("What does FAISS do?", top_k=2)

            assert "FAISS" in response.answer
            assert "faiss_overview.pdf (Page 3)" in response.sources
            assert len(response.retrieved_chunks) == 1
            assert response.retrieved_chunks[0].page == 3
            assert response.retrieved_chunks[0].source == "faiss_overview.pdf"
            assert response.latency_ms > 0

    def test_answer_when_info_is_absent(self, mock_env):
        """When evidence is absent from documents, model states it cannot find the answer."""
        import app.services.rag_pipeline as rag

        fake_chunks = [
            {
                "text": "This document discusses gardening techniques.",
                "source": "garden.pdf",
                "page": 1,
                "score": 0.85,
            }
        ]

        with patch("app.services.rag_pipeline.similarity_search", return_value=fake_chunks), \
             patch("app.services.rag_pipeline.get_llm_chain") as mock_chain_getter:

            mock_chain = MagicMock()
            # Strict fallback output from system prompt
            mock_chain.invoke.return_value = "I don't have enough information in the provided documents to answer that."
            mock_chain_getter.return_value = mock_chain

            response = rag.answer_question("What is quantum entanglement?")

            assert "don't have enough information" in response.answer.lower()


# ============================================================
# 5. FASTAPI ENDPOINT TESTS
# ============================================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints in routes.py."""

    @pytest.fixture
    def client(self, mock_env):
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client

    def test_health_endpoint(self, client):
        """GET /api/v1/health returns status and FAISS stats."""
        with patch("app.api.routes.get_document_count", return_value=8), \
             patch("app.api.routes.get_indexed_documents", return_value=[{"filename": "doc1.pdf", "chunk_count": 8}]):
            res = client.get("/api/v1/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ok"
            assert data["vector_db_type"] == "FAISS"
            assert data["documents_in_db"] == 8
            assert "doc1.pdf" in data["indexed_files"]

    def test_documents_list_endpoint(self, client):
        """GET /api/v1/documents returns document breakdown."""
        with patch("app.api.routes.get_indexed_documents", return_value=[{"filename": "guide.pdf", "chunk_count": 4}]):
            res = client.get("/api/v1/documents")
            assert res.status_code == 200
            data = res.json()
            assert data["total_chunks"] == 4
            assert data["documents"][0]["filename"] == "guide.pdf"

    def test_upload_pdf_endpoint(self, client):
        """POST /api/v1/upload accepts PDF file and returns chunk count."""
        fake_pdf_content = b"%PDF-1.4 minimal test"
        fake_docs = [
            Document(page_content="Chunk 1", metadata={"source": "upload_test.pdf", "page": 1}),
            Document(page_content="Chunk 2", metadata={"source": "upload_test.pdf", "page": 2}),
        ]

        with patch("app.api.routes.extract_text_from_pdf_stream", return_value=(fake_docs, 2)), \
             patch("app.api.routes.add_documents", return_value=2):

            res = client.post(
                "/api/v1/upload",
                files={"file": ("upload_test.pdf", fake_pdf_content, "application/pdf")},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["filename"] == "upload_test.pdf"
            assert data["chunks_added"] == 2
            assert data["pages_read"] == 2

    def test_upload_non_pdf_rejected(self, client):
        """Uploading non-PDF files must return 400 Bad Request."""
        res = client.post(
            "/api/v1/upload",
            files={"file": ("image.png", b"fake image bytes", "image/png")},
        )
        assert res.status_code == 400
        assert "Only PDF files are supported" in res.json()["detail"]

    def test_query_endpoint(self, client):
        """POST /api/v1/query returns answer, sources, and chunks."""
        from app.models.schemas import QueryResponse, SourceChunk

        fake_resp = QueryResponse(
            answer="FAISS handles vector search.",
            sources=["faiss.pdf (Page 1)"],
            retrieved_chunks=[
                SourceChunk(text="Chunk 1", source="faiss.pdf", page=1, score=0.12)
            ],
            model_used="gpt-3.5-turbo",
            latency_ms=150.0,
            from_cache=False,
        )

        with patch("app.api.routes.answer_question", return_value=fake_resp):
            res = client.post(
                "/api/v1/query",
                json={"question": "What is FAISS?", "top_k": 3},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["answer"] == "FAISS handles vector search."
            assert len(data["sources"]) == 1
            assert data["sources"][0] == "faiss.pdf (Page 1)"
            assert len(data["retrieved_chunks"]) == 1
            assert data["retrieved_chunks"][0]["score"] == 0.12
