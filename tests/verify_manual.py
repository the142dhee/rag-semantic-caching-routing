"""
Verify loading and chunking the generated sample PDF.
"""
from app.services.pdf_loader import load_and_split_pdf

chunks = load_and_split_pdf("data/pdfs/sample_rag_guide.pdf")
print(f"Total chunks extracted: {len(chunks)}")
for i, c in enumerate(chunks, 1):
    source = c.metadata.get("source")
    page = c.metadata.get("page")
    print(f"Chunk {i} -> [Doc: {source}, Page: {page}]:\n  {c.page_content}\n")
