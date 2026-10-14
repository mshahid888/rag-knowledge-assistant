"""Tests for citations: format, verbatim quotes, and the refusal contract."""

from src.answerer import ExtractiveAnswerer, format_citation
from src.ingestion import Chunk
from src.retriever import RetrievedChunk


def _hit(filename, page, text, score):
    chunk = Chunk(doc_id="d", filename=filename, page=page,
                  chunk_index=0, text=text)
    return RetrievedChunk(chunk=chunk, score=score,
                          tfidf_score=score, bm25_score=score)


def test_citation_format():
    assert format_citation("policy.pdf", 2) == "[source: policy.pdf, p. 2]"
    assert format_citation("a b.pdf", 1) == "[source: a b.pdf, p. 1]"


def test_answer_quotes_verbatim_with_citations():
    quote = "Employees receive 25 vacation days per calendar year."
    hits = [_hit("vacation.pdf", 1, quote + " Extra context here.", 0.9)]
    ans = ExtractiveAnswerer().answer("How many vacation days?", hits)
    assert not ans.refused
    assert quote in ans.text, "answer must contain the verbatim quote"
    assert "[source: vacation.pdf, p. 1]" in ans.text
    assert len(ans.citations) == 1
    assert ans.citations[0].filename == "vacation.pdf"
    assert ans.citations[0].page == 1


def test_refusal_on_no_hits():
    ans = ExtractiveAnswerer().answer("What is the refund policy?", [])
    assert ans.refused
    assert ans.citations == []
    assert "couldn't find" in ans.text.lower()


def test_refusal_on_weak_hits():
    hits = [_hit("vacation.pdf", 1, "Something vaguely unrelated.", 0.02)]
    ans = ExtractiveAnswerer(min_score=0.10).answer("quantum physics?", hits)
    assert ans.refused, "weak hits must trigger refusal, never hallucination"


def test_answer_never_invents_content():
    # The answer text may only consist of: intro line + verbatim quotes +
    # citation markers. Strip those and nothing claim-like may remain.
    quote = "Passwords must be at least 12 characters long."
    hits = [_hit("it.pdf", 1, quote, 0.8)]
    ans = ExtractiveAnswerer().answer("password length?", hits)
    body = ans.text
    body = body.replace("Based on your uploaded documents:", "")
    body = body.replace(f'"{quote}"', "")
    body = body.replace("[source: it.pdf, p. 1]", "")
    assert body.strip() == "", f"unexpected generated content: {body!r}"
