"""DocumentStore: owns PDFs on disk, chunk ingestion, and the retrieval index.

Simple and predictable: every mutation (upload/delete) triggers a full
re-index, which is instant at portfolio scale and can never go stale.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .answerer import Answer, Answerer, ExtractiveAnswerer
from .ingestion import Chunk, ingest_pdf
from .retriever import HybridRetriever, RetrievedChunk, Retriever


@dataclass
class DocumentMeta:
    doc_id: str
    filename: str
    pages: int
    chunks: int
    path: str = ""


@dataclass
class AskResult:
    answer: Answer
    hits: list[RetrievedChunk] = field(default_factory=list)


class DocumentStore:
    def __init__(
        self,
        data_dir: str | Path,
        retriever: Retriever | None = None,
        answerer: Answerer | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.retriever: Retriever = retriever or HybridRetriever()
        self.answerer: Answerer = answerer or ExtractiveAnswerer()
        self.documents: dict[str, DocumentMeta] = {}
        self.reindex()

    # -- mutations -----------------------------------------------------
    def add_pdf(self, filename: str, pdf_bytes: bytes) -> DocumentMeta:
        safe_name = Path(filename).name
        if not safe_name.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")
        dest = self.upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
        dest.write_bytes(pdf_bytes)
        self.reindex()
        # Find the meta entry for the file we just added.
        for meta in self.documents.values():
            if meta.filename == safe_name and meta.path == str(dest):
                return meta
        raise RuntimeError("Re-index did not pick up the uploaded file.")

    def delete(self, doc_id: str) -> bool:
        meta = self.documents.get(doc_id)
        if meta is None:
            return False
        Path(meta.path).unlink(missing_ok=True)
        self.reindex()
        return True

    def list_documents(self) -> list[DocumentMeta]:
        return sorted(self.documents.values(), key=lambda d: d.filename)

    # -- indexing ------------------------------------------------------
    def reindex(self) -> None:
        self.documents = {}
        all_chunks: list[Chunk] = []
        for pdf_path in sorted(self.upload_dir.glob("*.pdf")):
            # Original filename is stored after the uuid prefix.
            original = pdf_path.name.split("_", 1)[1] if "_" in pdf_path.name else pdf_path.name
            doc_id = uuid.uuid5(uuid.NAMESPACE_URL, str(pdf_path)).hex
            chunks = ingest_pdf(doc_id, original, str(pdf_path))
            pages = {c.page for c in chunks}
            meta = DocumentMeta(
                doc_id=doc_id,
                filename=original,
                pages=len(pages),
                chunks=len(chunks),
                path=str(pdf_path),
            )
            self.documents[doc_id] = meta
            all_chunks.extend(chunks)
        self.retriever.index(all_chunks)

    # -- Q&A -----------------------------------------------------------
    def ask(self, question: str, k: int = 5) -> AskResult:
        hits = self.retriever.retrieve(question, k=k)
        return AskResult(answer=self.answerer.answer(question, hits), hits=hits)

    def clear(self) -> None:
        shutil.rmtree(self.upload_dir, ignore_errors=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.reindex()
