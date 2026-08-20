from docsense.indexing.store import upsert_chunks
from docsense.llm.base import FakeProvider
from docsense.rag.chain import ask, build_context, build_prompt
from docsense.retrieval.hybrid import Hit, invalidate_bm25_cache

from .conftest import make_corpus


def _hits(n: int = 2) -> list[Hit]:
    return [Hit(chunk=c, score=1.0 - 0.1 * i) for i, c in enumerate(make_corpus()[:n])]


def test_build_context_includes_citation_headers():
    ctx = build_context(_hits(), max_chars=5000)
    assert "[acme-10k, p.1]" in ctx
    assert "12.5 million" in ctx


def test_build_context_respects_char_budget():
    ctx = build_context(_hits(3), max_chars=120)
    assert len(ctx) <= 120


def test_build_prompt_contains_question_and_context():
    prompt = build_prompt("What was revenue?", _hits())
    assert "What was revenue?" in prompt
    assert "[acme-10k, p.1]" in prompt
    assert "cite" in prompt.lower()


def test_ask_end_to_end_with_fake_provider():
    upsert_chunks(make_corpus())
    invalidate_bm25_cache()
    provider = FakeProvider(canned="Revenue was 12.5 million dollars [acme-10k, p.1].")
    result = ask("What was the total revenue?", provider=provider)
    assert result.provider == "fake"
    assert "12.5 million" in result.answer
    assert result.hits, "retrieval should surface chunks"
    # The provider must have received the grounded prompt, not the bare question.
    sent_prompt = provider.calls[0]["prompt"]
    assert "Context excerpts" in sent_prompt
    assert "What was the total revenue?" in sent_prompt
