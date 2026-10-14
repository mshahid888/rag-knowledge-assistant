"""FastAPI backend for the Business Knowledge Assistant (RAG).

Endpoints:
    GET  /                     -> minimal HTML frontend
    GET  /health               -> liveness check
    POST /api/upload           -> upload a PDF (multipart, field "file")
    GET  /api/documents        -> list ingested documents
    DELETE /api/documents/{id} -> delete a document
    POST /api/ask               -> {"question": str, "k": int} ->
                                   {"answer", "refused", "citations", "hits"}

Everything runs locally: no API keys, no network calls, no paid services.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.answerer import format_citation
from src.store import DocumentStore

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="Business Knowledge Assistant (RAG)")
store = DocumentStore(DATA_DIR)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(store.documents)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(pdf_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 25 MB).")
    try:
        meta = store.add_pdf(file.filename, pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {
            "doc_id": meta.doc_id,
            "filename": meta.filename,
            "pages": meta.pages,
            "chunks": meta.chunks,
        }
    )


@app.get("/api/documents")
def list_documents() -> JSONResponse:
    return JSONResponse(
        [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "pages": d.pages,
                "chunks": d.chunks,
            }
            for d in store.list_documents()
        ]
    )


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> JSONResponse:
    if not store.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return JSONResponse({"deleted": doc_id})


@app.post("/api/ask")
def ask(req: AskRequest) -> JSONResponse:
    if not store.documents:
        raise HTTPException(
            status_code=400, detail="No documents uploaded yet."
        )
    result = store.ask(req.question, k=req.k)
    return JSONResponse(
        {
            "answer": result.answer.text,
            "refused": result.answer.refused,
            "citations": [
                {
                    "filename": c.filename,
                    "page": c.page,
                    "text": c.text,
                    "score": c.score,
                    "marker": format_citation(c.filename, c.page),
                }
                for c in result.answer.citations
            ],
            "hits": [
                {
                    "filename": h.chunk.filename,
                    "page": h.chunk.page,
                    "score": round(h.score, 3),
                    "text": h.chunk.text[:300],
                }
                for h in result.hits
            ],
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
