"""Retrieval: clean interface + hybrid TF-IDF / BM25 ranking.

`HybridRetriever` blends two complementary signals:
  * TF-IDF cosine similarity (vector-space relevance)
  * BM25 (probabilistic term weighting, strong on keyword queries)
Both are implemented from scratch on the standard library -- no API keys,
no downloads, fully offline.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

from .embeddings import Embedder, TfidfEmbedder
from .ingestion import Chunk
from .preprocessing import tokenize


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # blended hybrid score in [0, 1]
    tfidf_score: float
    bm25_score: float


class Retriever(ABC):
    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """(Re)build the retrieval index from chunks."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, best first."""


class HybridRetriever(Retriever):
    """Weighted blend of TF-IDF cosine and BM25, normalized to [0, 1]."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        tfidf_weight: float = 0.5,
        bm25_weight: float = 0.5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_freq: Counter[str] = Counter()
        self._doc_lens: list[int] = []
        self._avg_len = 0.0
        self._tfidf_vecs: list[dict[str, float]] = []

    # -- indexing ------------------------------------------------------
    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        token_lists = [c.tokens for c in chunks]
        self.embedder.fit(token_lists)
        self._tfidf_vecs = [self.embedder.transform(t) for t in token_lists]
        self._doc_freq = Counter()
        for tokens in token_lists:
            self._doc_freq.update(set(tokens))
        self._doc_lens = [len(t) for t in token_lists]
        self._avg_len = (
            sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        )

    # -- BM25 ----------------------------------------------------------
    def _bm25(self, query_tokens: list[str], idx: int) -> float:
        n = len(self._chunks)
        if n == 0:
            return 0.0
        tf = Counter(self._chunks[idx].tokens)
        dl = self._doc_lens[idx] or 1
        score = 0.0
        for term in set(query_tokens):
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            f = tf.get(term, 0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    # -- retrieval -----------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        q_tokens = tokenize(query)
        if not q_tokens or not self._chunks:
            return []
        q_vec = self.embedder.transform(q_tokens)

        ranked: list[RetrievedChunk] = []
        for i, chunk in enumerate(self._chunks):
            # TF-IDF cosine is naturally in [0, 1] -- absolute, comparable
            # across queries. BM25 is unbounded, so it gets a saturating
            # normalization b/(b+1) into [0, 1). Deliberately NOT
            # normalized by the max of this result set: the top hit must be
            # able to score low in absolute terms, otherwise a refusal
            # threshold could never fire.
            t = self.embedder.similarity(q_vec, self._tfidf_vecs[i])
            b_raw = self._bm25(q_tokens, i)
            b = b_raw / (b_raw + 1.0) if b_raw > 0 else 0.0
            blended = self.tfidf_weight * t + self.bm25_weight * b
            if blended > 0.0:
                ranked.append(
                    RetrievedChunk(
                        chunk=chunk, score=blended, tfidf_score=t, bm25_score=b
                    )
                )
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:k]
