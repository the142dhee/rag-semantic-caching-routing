"""
tests/test_stage1_rag.py
========================
PURPOSE: Baseline tests for the RAG pipeline components.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_documents():
    """A small set of Document chunks for testing."""
    return [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "test.pdf", "page": 1},
        ),
        Document(
            page_content="FastAPI is a modern web framework for building APIs with Python.",
            metadata={"source": "test.pdf", "page": 2},
        ),
        Document(
            page_content="FAISS is an efficient vector similarity search library.",
            metadata={"source": "test.pdf", "page": 3},
        ),
    ]


@pytest.fixture
def mock_settings(monkeypatch, tmp_path):
    """Override settings to avoid needing a real .env file during tests."""
    temp_dir = str(tmp_path / "faiss_test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("FAISS_INDEX_DIR", temp_dir)

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.services.vector_store as vs
    vs._vector_store = None
    vs._embedding_model = None

    yield

    get_settings.cache_clear()
    vs._vector_store = None
    vs._embedding_model = None


# ============================================================
# UNIT TESTS — PDF Loader
# ============================================================

class TestPDFLoader:
    """Tests for chunking and loader utilities."""

    def test_chunk_documents_returns_chunks(self, mock_settings, sample_documents):
        """Verify chunk_documents returns Document objects with correct structure."""
        from app.services.pdf_loader import chunk_documents
        result = chunk_documents(sample_documents, chunk_size=200, chunk_overlap=20)

        assert isinstance(result, list)
        assert len(result) >= len(sample_documents)
        assert all(isinstance(doc, Document) for doc in result)

    def test_chunking_respects_chunk_size(self, mock_settings):
        """Verify that no chunk exceeds the configured chunk_size."""
        from app.services.pdf_loader import chunk_documents
        from app.core.config import get_settings
        settings = get_settings()

        long_text = "This is a sentence. " * 100
        long_doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})

        result = chunk_documents([long_doc], chunk_size=300, chunk_overlap=30)
        for chunk in result:
            assert len(chunk.page_content) <= 350


# ============================================================
# UNIT TESTS — Vector Store
# ============================================================

class TestVectorStore:
    """Tests for vector store operations."""

    def test_similarity_search_returns_list(self, mock_settings, sample_documents):
        """Verify similarity_search returns a list of dicts with expected keys."""
        import app.services.vector_store as vs
        fake_embed = FakeEmbeddings(size=1536)

        with patch.object(vs, "get_embedding_model", return_value=fake_embed):
            vs.add_documents(sample_documents)
            results = vs.similarity_search("What is Python?", k=2)

            assert isinstance(results, list)
            assert len(results) == 2
            assert "text" in results[0]
            assert "source" in results[0]
            assert "page" in results[0]
            assert "score" in results[0]

    def test_add_documents_calls_embed(self, mock_settings, sample_documents):
        """Verify add_documents indexes all documents and increases count."""
        import app.services.vector_store as vs
        fake_embed = FakeEmbeddings(size=1536)

        with patch.object(vs, "get_embedding_model", return_value=fake_embed):
            count = vs.add_documents(sample_documents)
            assert count == len(sample_documents)
            assert vs.get_document_count() == len(sample_documents)


# ============================================================
# UNIT TESTS — RAG Pipeline
# ============================================================

class TestRAGPipeline:
    """Tests for app/services/rag_pipeline.py"""

    def test_answer_question_returns_response(self, mock_settings):
        """Verify answer_question returns a QueryResponse with expected fields."""
        fake_chunks = [
            {"text": "Python was created by Guido van Rossum.", "source": "test.pdf", "page": 1, "score": 0.1},
        ]
        fake_answer = "Python was created by Guido van Rossum in 1991 [Doc: test.pdf, Page 1]."

        with patch("app.services.rag_pipeline.similarity_search", return_value=fake_chunks), \
             patch("app.services.rag_pipeline.get_llm_chain") as mock_chain_getter:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = fake_answer
            mock_chain_getter.return_value = mock_chain

            from app.services.rag_pipeline import answer_question
            response = answer_question("Who created Python?")

            assert response.answer == fake_answer
            assert len(response.sources) == 1
            assert response.sources[0] == "test.pdf (Page 1)"
            assert response.from_cache is False
            assert response.latency_ms is not None

    def test_answer_question_no_documents(self, mock_settings):
        """When no documents are in the DB, return a helpful message."""
        with patch("app.services.rag_pipeline.similarity_search", return_value=[]):
            from app.services.rag_pipeline import answer_question
            response = answer_question("Any question")

            assert "No documents" in response.answer or "ingested" in response.answer
