"""Document ingestion: PDF text extraction + chunking with overlap.

Every chunk carries its source metadata (document id, filename, 1-based page
number) so answers can cite the exact origin of each quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from .preprocessing import tokenize


@dataclass
class Chunk:
    doc_id: str
    filename: str
    page: int  # 1-based page number
    chunk_index: int  # 0-based index within the document
    text: str
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(self.text)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for every page of a PDF (1-based)."""
    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        # Collapse stray whitespace but keep sentence structure.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append((i, text))
    return pages


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_page_text(
    text: str,
    max_chars: int = 600,
    overlap_chars: int = 120,
) -> list[str]:
    """Split page text into overlapping chunks, preferring sentence boundaries.

    Falls back to a hard character split for pathological pages without
    sentence punctuation (e.g. tables).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))

    for sent in sentences:
        # Over-long single sentence: hard-split it.
        while len(sent) > max_chars:
            if current:
                flush()
                current, current_len = [], 0
            chunks.append(sent[:max_chars])
            sent = sent[max_chars:]
        if current_len + len(sent) + 1 > max_chars and current:
            flush()
            # Overlap: carry the tail of the previous chunk forward,
            # re-split at a sentence boundary where possible.
            tail = chunks[-1][-overlap_chars:]
            tail_sents = _split_sentences(tail)
            current = tail_sents[-1:] if tail_sents else []
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1
    flush()
    return [c for c in chunks if c.strip()]


def ingest_pdf(doc_id: str, filename: str, pdf_path: str) -> list[Chunk]:
    """Extract, chunk and tokenize a PDF. Returns chunks with full metadata."""
    chunks: list[Chunk] = []
    idx = 0
    for page_num, page_text in extract_pages(pdf_path):
        if not page_text:
            continue
        for piece in chunk_page_text(page_text):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    page=page_num,
                    chunk_index=idx,
                    text=piece,
                )
            )
            idx += 1
    return chunks
