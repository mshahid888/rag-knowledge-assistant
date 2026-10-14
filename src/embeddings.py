"""Embeddings: clean interface + a dependency-free TF-IDF implementation.

The default `TfidfEmbedder` runs fully offline on the standard library.
To upgrade to neural embeddings later, implement `Embedder` with
sentence-transformers (or any vector model) and pass it to HybridRetriever --
see `SentenceTransformerEmbedder` below, which activates automatically once
the package is installed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter


class Embedder(ABC):
    """Maps token lists to sparse/dense vectors and compares them."""

    @abstractmethod
    def fit(self, corpus: list[list[str]]) -> None:
        """Learn corpus statistics (e.g. IDF) from tokenized documents."""

    @abstractmethod
    def transform(self, tokens: list[str]) -> dict[str, float]:
        """Vectorize one token list."""

    def similarity(self, a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors in [0, 1]."""
        if not a or not b:
            return 0.0
        dot = sum(w * b.get(t, 0.0) for t, w in a.items())
        na = math.sqrt(sum(w * w for w in a.values()))
        nb = math.sqrt(sum(w * w for w in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


class TfidfEmbedder(Embedder):
    """TF-IDF vectors, pure stdlib. Good default for small corpora."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self._n_docs = 0

    def fit(self, corpus: list[list[str]]) -> None:
        self._n_docs = len(corpus)
        df: Counter[str] = Counter()
        for tokens in corpus:
            df.update(set(tokens))
        # Smoothed IDF so unseen terms don't explode.
        self.idf = {
            t: math.log(1.0 + self._n_docs / (1.0 + c)) for t, c in df.items()
        }

    def transform(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        n = len(tokens)
        vec: dict[str, float] = {}
        for t, c in tf.items():
            idf = self.idf.get(t)
            if idf is None:
                continue  # term never seen in corpus -> contributes nothing
            vec[t] = (c / n) * idf
        return vec


class SentenceTransformerEmbedder(Embedder):
    """Optional neural upgrade path. Requires `pip install sentence-transformers`.

    Drop-in replacement: HybridRetriever accepts any Embedder, so swapping
    `TfidfEmbedder()` for `SentenceTransformerEmbedder("all-MiniLM-L6-v2")`
    upgrades retrieval quality with no other code changes.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run: "
                "pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def fit(self, corpus: list[list[str]]) -> None:
        pass  # neural models need no corpus fitting

    def transform(self, tokens: list[str]) -> dict[str, float]:
        vec = self.model.encode(" ".join(tokens), normalize_embeddings=True)
        return {str(i): float(v) for i, v in enumerate(vec)}
