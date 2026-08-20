"""Hybrid retrieval: dense (Chroma) + lexical (BM25), fused with RRF.

Reciprocal-rank fusion is rank-based, so the two retrievers' incomparable
scores never need calibrating against each other.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from docsense.indexing.chunker import Chunk
from docsense.indexing.embedder import embed_query
from docsense.indexing.store import all_chunks, get_collection
from docsense.settings import get_config


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@functools.lru_cache(maxsize=1)
def _bm25_index(collection_name: str | None = None) -> tuple[BM25Okapi, list[Chunk]]:
    chunks = all_chunks(collection_name)
    corpus = [_tokenize(c.text) for c in chunks]
    return BM25Okapi(corpus or [["empty"]]), chunks


def invalidate_bm25_cache() -> None:
    """Call after upserting new documents."""
    _bm25_index.cache_clear()


def dense_search(query: str, k: int, collection_name: str | None = None) -> list[tuple[str, Chunk]]:
    collection = get_collection(collection_name)
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(k, max(collection.count(), 1)),
        include=["documents", "metadatas"],
    )
    hits = []
    for cid, text, meta in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], strict=True
    ):
        hits.append(
            (
                cid,
                Chunk(cid, meta["doc_id"], int(meta["page"]), text, meta.get("source", "digital")),
            )
        )
    return hits


def bm25_search(query: str, k: int, collection_name: str | None = None) -> list[tuple[str, Chunk]]:
    bm25, chunks = _bm25_index(collection_name)
    if not chunks:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i].chunk_id, chunks[i]) for i in ranked if scores[i] > 0]


def retrieve(query: str, top_k: int | None = None, collection_name: str | None = None) -> list[Hit]:
    cfg = get_config()["retrieval"]
    top_k = top_k or cfg["top_k"]
    candidates = cfg["candidates"]
    rrf_k = cfg["rrf_k"]

    ranked_lists = [dense_search(query, candidates, collection_name)]
    if cfg["hybrid"]:
        ranked_lists.append(bm25_search(query, candidates, collection_name))

    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, (cid, chunk) in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            by_id[cid] = chunk

    best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [Hit(chunk=by_id[cid], score=round(score, 6)) for cid, score in best]
