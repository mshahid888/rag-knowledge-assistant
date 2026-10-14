"""Tests for retrieval: ranking quality, score ranges, edge cases."""

import pytest

from src.ingestion import Chunk, ingest_pdf
from src.retriever import HybridRetriever


def _chunks_from_pdfs(sample_pdfs):
    chunks = []
    for i, pdf in enumerate(sample_pdfs):
        chunks.extend(ingest_pdf(f"doc{i}", pdf.name, str(pdf)))
    return chunks


def test_relevant_doc_ranks_first(sample_pdfs):
    chunks = _chunks_from_pdfs(sample_pdfs)
    r = HybridRetriever()
    r.index(chunks)

    hits = r.retrieve("How many vacation days do employees get per year?", k=3)
    assert hits, "expected hits for an on-topic question"
    assert "vacation" in hits[0].chunk.filename, (
        f"expected vacation doc first, got {hits[0].chunk.filename}"
    )

    hits = r.retrieve("What is the minimum password length?", k=3)
    assert "it-security" in hits[0].chunk.filename

    hits = r.retrieve("How long does the WidgetPro battery last?", k=3)
    assert "widgetpro" in hits[0].chunk.filename


def test_scores_in_unit_range(sample_pdfs):
    r = HybridRetriever()
    r.index(_chunks_from_pdfs(sample_pdfs))
    hits = r.retrieve("vacation days policy", k=5)
    assert hits
    for h in hits:
        assert 0.0 < h.score <= 1.0, f"score out of range: {h.score}"
    # Sorted best-first.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_query_returns_nothing_or_weak(sample_pdfs):
    r = HybridRetriever()
    r.index(_chunks_from_pdfs(sample_pdfs))
    hits = r.retrieve("quantum entanglement photon spin", k=5)
    # Nothing in the corpus is about quantum physics: either no hits at all,
    # or only very weak ones that the answerer would refuse on.
    assert all(h.score < 0.10 for h in hits)


def test_empty_query_returns_no_hits(sample_pdfs):
    r = HybridRetriever()
    r.index(_chunks_from_pdfs(sample_pdfs))
    assert r.retrieve("", k=5) == []
    assert r.retrieve("the and or", k=5) == []  # stopwords only


def test_retriever_accepts_custom_embedder():
    from src.embeddings import TfidfEmbedder

    chunks = [
        Chunk(doc_id="d", filename="f.pdf", page=1, chunk_index=0,
              text="The cat sat on the mat."),
        Chunk(doc_id="d", filename="f.pdf", page=1, chunk_index=1,
              text="Dogs love running in the park."),
    ]
    r = HybridRetriever(embedder=TfidfEmbedder())
    r.index(chunks)
    hits = r.retrieve("cat mat", k=2)
    assert hits and hits[0].chunk.chunk_index == 0
