"""Answering with citations: clean interface + extractive implementation.

The default `ExtractiveAnswerer` NEVER generates new claims: it quotes the
most relevant retrieved chunks verbatim, attaches `[source: file, p. N]`
citations, and refuses to answer when nothing relevant was retrieved.

Upgrade path: implement `Answerer` with an LLM (local or API-based) that
receives the retrieved chunks as context and is instructed to cite them.
The refusal contract ("never answer beyond the documents") must be kept.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .retriever import RetrievedChunk

REFUSAL_TEXT = (
    "I couldn't find anything about this in the uploaded documents. "
    "Try rephrasing your question, or upload a document that covers this topic."
)


@dataclass
class Citation:
    filename: str
    page: int
    text: str  # verbatim quote from the document
    score: float


@dataclass
class Answer:
    text: str  # full answer incl. inline citation markers
    citations: list[Citation]
    refused: bool


def format_citation(filename: str, page: int) -> str:
    """Canonical citation marker: [source: filename, p. N]."""
    return f"[source: {filename}, p. {page}]"


class Answerer(ABC):
    @abstractmethod
    def answer(
        self, question: str, hits: list[RetrievedChunk]
    ) -> Answer:
        """Compose an answer from retrieved chunks (may refuse)."""


class ExtractiveAnswerer(Answerer):
    """Quotes retrieved chunks verbatim. Refuses below `min_score`.

    Args:
        min_score: minimum hybrid retrieval score for the top hit;
            anything below triggers the "not in the documents" refusal.
        max_quotes: how many chunks to quote at most.
        max_quote_chars: quotes longer than this are trimmed with [...].
    """

    def __init__(
        self,
        min_score: float = 0.10,
        max_quotes: int = 3,
        max_quote_chars: int = 450,
    ) -> None:
        self.min_score = min_score
        self.max_quotes = max_quotes
        self.max_quote_chars = max_quote_chars

    def _trim(self, text: str) -> str:
        text = " ".join(text.split())
        if len(text) > self.max_quote_chars:
            return text[: self.max_quote_chars].rstrip() + " [...]"
        return text

    def answer(
        self, question: str, hits: list[RetrievedChunk]
    ) -> Answer:
        if not hits or hits[0].score < self.min_score:
            return Answer(text=REFUSAL_TEXT, citations=[], refused=True)

        # Only quote hits that clear the bar -- weak tail hits are dropped.
        usable = [h for h in hits[: self.max_quotes] if h.score >= self.min_score]
        if not usable:
            return Answer(text=REFUSAL_TEXT, citations=[], refused=True)

        citations = [
            Citation(
                filename=h.chunk.filename,
                page=h.chunk.page,
                text=self._trim(h.chunk.text),
                score=round(h.score, 3),
            )
            for h in usable
        ]

        lines = ["Based on your uploaded documents:\n"]
        for c in citations:
            lines.append(f'"{c.text}" {format_citation(c.filename, c.page)}\n')
        return Answer(text="\n".join(lines).strip(), citations=citations, refused=False)
