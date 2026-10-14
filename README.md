# Business Knowledge Assistant (RAG)

Upload company PDFs → ask questions in plain language → get answers that **quote
the documents verbatim** with `[source: filename, p. N]` citations. When the
answer isn't in the documents, it says so — it never invents content.

100% local. No API keys, no network calls, no paid services. Runs fully offline.

> **Demo data:** the bundled sample PDFs describe the fictional company
> "Muster GmbH" (German for "Sample Ltd.") and its fictional "WidgetPro 3000"
> product. All content is invented for demonstration — no real company,
> product, or policy is referenced anywhere in this repo.

## Architecture

```
                        ┌─────────────┐
                        │  index.html │  minimal frontend (upload / ask / cites)
                        └──────┬──────┘
                               │  REST/JSON
┌──────────┐  upload   ┌────────▼────────┐  ask   ┌──────────────────┐
│ PDF file │ ────────► │  FastAPI (app)  │ ──────► │  DocumentStore   │
└──────────┘           └────────┬────────┘        └────────┬─────────┘
                               │                          │
                    ┌──────────▼──────────┐     ┌─────────▼──────────┐
                    │ ingestion.py        │     │ HybridRetriever    │
                    │ pypdf extract →     │     │  ├─ TF-IDF cosine  │
                    │ sentence-aware      │     │  └─ BM25           │
                    │ chunking + overlap  │     │  (both stdlib-only)│
                    │ + page metadata     │     └─────────┬──────────┘
                    └─────────────────────┘               │ top-k chunks
                                              ┌───────────▼────────────┐
                                              │ ExtractiveAnswerer     │
                                              │ quotes verbatim +      │
                                              │ [source: file, p. N]   │
                                              │ refuses if score < bar │
                                              └────────────────────────┘
```

**Clean upgrade interfaces** (see module docstrings):

| Interface    | Default (offline)        | Plug-in upgrade path                          |
|--------------|--------------------------|-----------------------------------------------|
| `Embedder`   | `TfidfEmbedder` (stdlib) | `SentenceTransformerEmbedder` (one-line swap) |
| `Retriever`  | `HybridRetriever`        | any `Retriever` implementation                |
| `Answerer`   | `ExtractiveAnswerer`     | LLM answerer using chunks as context (keep the refusal contract) |

## Setup

```bash
cd rag-knowledge-assistant
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Regenerate the sample PDFs any time (they are committed, this is optional):

```bash
.venv/bin/python sample_docs/generate_samples.py
```

Run the tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

Start the server:

```bash
.venv/bin/python -m uvicorn app:app --port 8000
# open http://127.0.0.1:8000
```

## Demo in 3 steps

1. **Upload** — drag the 3 PDFs from `sample_docs/` into the upload box
   (or `curl -X POST localhost:8000/api/upload -F "file=@sample_docs/muster-gmbh-vacation-policy.pdf"`).
2. **Ask** — type a question, e.g. *"How many vacation days do employees get per year?"*
3. **Verify** — the answer quotes the document verbatim with a highlighted
   `[source: muster-gmbh-vacation-policy.pdf, p. 1]` citation; expand the cited
   chunk cards below to see filename, page, and retrieval score.

## Recording guide

Record in this order (total ~2 minutes):

1. **Uploading docs (0:00–0:30).** Browser at `http://127.0.0.1:8000`.
   Drag all 3 sample PDFs into the upload box. Point at the document list
   showing *"2 pages · N chunks"* per file.
2. **Question 1 — factual (0:30–1:00).** Ask *"What is the minimum password
   length?"* Show the quoted answer and the highlighted
   `[source: muster-gmbh-it-security-policy.pdf, p. 1]` citation.
3. **Question 2 — cross-document (1:00–1:30).** Ask *"How long does the
   WidgetPro 3000 battery last in standard mode?"* Show the citation pointing
   at the product manual.
4. **Question 3 — the refusal (1:30–2:00).** Ask *"Who won the football World
   Cup?"* Show the *"I couldn't find anything about this in the uploaded
   documents"* response. Narrate: "It quotes or it refuses — it never
   hallucinates."

## API reference

| Method | Path                      | Description                                    |
|--------|---------------------------|------------------------------------------------|
| GET    | `/`                       | HTML frontend                                  |
| GET    | `/health`                 | liveness + document count                      |
| POST   | `/api/upload`             | multipart `file` (PDF, max 25 MB) → doc meta   |
| GET    | `/api/documents`          | list documents (id, filename, pages, chunks)   |
| DELETE | `/api/documents/{doc_id}` | delete a document (re-indexes)                 |
| POST   | `/api/ask`                | `{"question": str, "k": 5}` → answer + citations + raw hits |

## How the "never hallucinate" guarantee works

1. **Extractive only.** `ExtractiveAnswerer` composes answers exclusively from
   verbatim quotes of retrieved chunks. A test (`test_answer_never_invents_content`)
   asserts the answer contains nothing beyond the intro line, quotes, and
   citation markers.
2. **Absolute scoring.** Retrieval scores are absolute (TF-IDF cosine ∈ [0,1],
   BM25 via saturating normalization) — deliberately *not* normalized per
   query, so a weak top hit stays weak and trips the refusal threshold
   (`min_score`, default 0.10).
3. **Refusal.** Below the bar → *"I couldn't find anything about this in the
   uploaded documents."* No fallback generation, ever.

**Known limitation (documented, not hidden):** a query that shares generic
vocabulary with the docs (e.g. "refund policy" when the docs mention "policy"
and "software" in other contexts) can retrieve weakly-related chunks. The
quotes are still verbatim and cited — nothing is invented — but relevance is
imperfect. The production upgrade path is a cross-encoder reranker or an LLM
relevance judge in front of the answerer; the `Answerer` interface is built
for exactly that swap.

## Project layout

```
rag-knowledge-assistant/
├── app.py                 # FastAPI: upload / list / ask / delete / frontend
├── src/
│   ├── preprocessing.py   # tokenize + stopwords + light stemmer
│   ├── ingestion.py       # PDF extract → sentence-aware chunking + page meta
│   ├── embeddings.py      # Embedder interface + TF-IDF (+ ST stub)
│   ├── retriever.py       # Retriever interface + hybrid TF-IDF/BM25
│   ├── answerer.py        # Answerer interface + extractive quoter + refusal
│   └── store.py           # DocumentStore: uploads, index, ask()
├── static/index.html      # minimal frontend (vanilla JS, no build step)
├── sample_docs/           # 3 FICTIONAL demo PDFs + generator script
├── tests/                 # chunking, retrieval, citation/refusal tests
├── data/uploads/          # runtime uploads (gitignored)
└── requirements.txt
```
