from fastapi.testclient import TestClient

from docsense.api.main import create_app
from docsense.indexing.store import upsert_chunks
from docsense.retrieval.hybrid import invalidate_bm25_cache

from .conftest import make_corpus


def _client() -> TestClient:
    return TestClient(create_app())


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents_lists_indexed_docs():
    upsert_chunks(make_corpus())
    invalidate_bm25_cache()
    r = _client().get("/documents")
    assert r.status_code == 200
    assert r.json() == {"acme-10k": 3, "globex-10k": 3}


def test_ask_streams_sources_then_tokens():
    upsert_chunks(make_corpus())
    invalidate_bm25_cache()
    with _client().stream(
        "POST", "/ask", json={"question": "What was the revenue?", "provider": "fake"}
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "event: sources" in body
    assert "acme-10k" in body
    assert "event: token" in body
    assert "event: done" in body


def test_ask_rejects_unknown_provider():
    r = _client().post("/ask", json={"question": "What was the revenue?", "provider": "nope"})
    assert r.status_code == 422


def test_ask_rejects_short_question():
    r = _client().post("/ask", json={"question": "hi", "provider": "fake"})
    assert r.status_code == 422


def test_upload_rejects_non_pdf():
    r = _client().post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 422
