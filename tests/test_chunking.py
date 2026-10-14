"""Tests for chunking: sizes, overlap, page metadata, edge cases."""

from src.ingestion import chunk_page_text, ingest_pdf


def test_chunks_respect_max_chars():
    text = " ".join(f"Sentence number {i} about vacation policy." for i in range(40))
    chunks = chunk_page_text(text, max_chars=200, overlap_chars=40)
    assert len(chunks) > 1
    for c in chunks:
        # Allow slight overflow from one long sentence, but nothing crazy.
        assert len(c) <= 400, f"chunk too long: {len(c)}"


def test_overlap_carries_context():
    text = " ".join(f"Unique sentence alpha {i}." for i in range(30))
    chunks = chunk_page_text(text, max_chars=150, overlap_chars=60)
    assert len(chunks) >= 2
    # The tail of chunk 0 should reappear at the start of chunk 1.
    tail_words = set(chunks[0].split()[-6:])
    head_words = set(chunks[1].split()[:12])
    assert tail_words & head_words, "no overlap between consecutive chunks"


def test_empty_text_yields_no_chunks():
    assert chunk_page_text("") == []
    assert chunk_page_text("   \n  ") == []


def test_ingest_pdf_carries_page_metadata(sample_pdfs):
    # sample_pdfs fixture -> list of pdf paths (see conftest.py)
    chunks = ingest_pdf("doc1", "vacation.pdf", str(sample_pdfs[0]))
    assert chunks, "expected chunks from sample PDF"
    pages = {c.page for c in chunks}
    assert pages == {1, 2}, f"expected pages 1 and 2, got {pages}"
    for c in chunks:
        assert c.doc_id == "doc1"
        assert c.filename == "vacation.pdf"
        assert c.text.strip()
        assert c.tokens, "chunks should be tokenized"
