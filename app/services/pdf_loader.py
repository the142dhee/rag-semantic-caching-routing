"""
app/services/pdf_loader.py
==========================
PURPOSE: Load, validate, and chunk PDF documents with robust error handling.

Features:
  - Supports loading from file path or raw bytes (for web uploads).
  - Page-by-page extraction preserving 1-based page numbers and filenames.
  - Gracefully handles empty files, corrupted streams, and scanned/image-only PDFs.
  - Configurable chunk size and overlap via settings.
"""

import io
import os
import logging
from typing import List, Optional, Tuple
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def extract_text_from_pdf_stream(
    stream: io.BytesIO,
    filename: str,
) -> Tuple[List[Document], int]:
    """
    Extract text page-by-page from an in-memory PDF stream.

    Args:
        stream: BytesIO stream of the PDF file.
        filename: Name of the original PDF document.

    Returns:
        Tuple of (list of page-level Documents, total page count).

    Raises:
        ValueError: If file is empty, corrupted, unreadable, or contains no text.
    """
    if stream.getbuffer().nbytes == 0:
        raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

    try:
        reader = PdfReader(stream)
    except (PdfReadError, Exception) as e:
        logger.error(f"Failed to read PDF '{filename}': {e}")
        raise ValueError(f"Invalid or corrupted PDF file '{filename}': {str(e)}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF file '{filename}' has no pages.")

    page_documents: List[Document] = []
    total_extracted_chars = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed for human readability
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num} of '{filename}': {e}")
            page_text = ""

        page_text = page_text.strip()
        if page_text:
            total_extracted_chars += len(page_text)
            page_documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": os.path.basename(filename),
                        "page": page_num,
                    },
                )
            )

    if total_extracted_chars == 0:
        raise ValueError(
            f"PDF '{filename}' contains {total_pages} page(s) but no extractable text. "
            "It may be a scanned or image-only document."
        )

    logger.info(
        f"Extracted text from '{filename}': {len(page_documents)}/{total_pages} "
        f"pages have readable text ({total_extracted_chars} characters)."
    )
    return page_documents, total_pages


def load_pdf(pdf_path: str) -> Tuple[List[Document], int]:
    """
    Load and validate a PDF file from a local filesystem path.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (list of page-level Documents, total page count).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    stream = io.BytesIO(file_bytes)
    return extract_text_from_pdf_stream(stream, filename)


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split a list of page Documents into smaller chunks while preserving metadata.

    Args:
        documents: List of Document objects (with source and page metadata).
        chunk_size: Max characters per chunk (defaults to settings.chunk_size).
        chunk_overlap: Overlap between consecutive chunks (defaults to settings.chunk_overlap).

    Returns:
        List of chunked Document objects.
    """
    settings = get_settings()
    c_size = chunk_size or settings.chunk_size
    c_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} page documents into {len(chunks)} chunks "
        f"(chunk_size={c_size}, chunk_overlap={c_overlap})."
    )
    return chunks


def load_and_split_pdf(
    pdf_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Convenience function: load a PDF from disk and return its chunks.
    """
    page_docs, _ = load_pdf(pdf_path)
    return chunk_documents(page_docs, chunk_size, chunk_overlap)
