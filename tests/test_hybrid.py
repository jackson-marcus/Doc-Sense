from docsense.indexing.store import list_documents, upsert_chunks
from docsense.retrieval.hybrid import invalidate_bm25_cache, retrieve

from .conftest import make_corpus


def _index_corpus():
    upsert_chunks(make_corpus())
    invalidate_bm25_cache()


def test_upsert_and_list_documents():
    _index_corpus()
    docs = list_documents()
    assert docs == {"acme-10k": 3, "globex-10k": 3}


def test_retrieve_finds_topical_chunk():
    _index_corpus()
    hits = retrieve("What was the total revenue in fiscal year 2025?", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0].chunk.doc_id == "acme-10k"
    assert "revenue" in hits[0].chunk.text.lower()


def test_retrieve_distinguishes_documents():
    _index_corpus()
    hits = retrieve("Where did Globex open a research facility?", top_k=3)
    assert hits[0].chunk.doc_id == "globex-10k"
    assert "berlin" in hits[0].chunk.text.lower()


def test_retrieve_respects_top_k():
    _index_corpus()
    assert len(retrieve("dollars", top_k=2)) <= 2


def test_retrieve_empty_index_returns_nothing():
    assert retrieve("anything", top_k=3) == []
