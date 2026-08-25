"""ChromaDB persistence: collection management and chunk upserts."""

from __future__ import annotations

import functools
import logging

from docsense.indexing.chunker import Chunk
from docsense.indexing.embedder import embed_texts
from docsense.settings import get_config, resolve_path

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _client():
    import chromadb
    from chromadb.config import Settings

    persist_dir = str(resolve_path(get_config()["indexing"]["chroma_dir"]))
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            chroma_server_api_default="chromadb.api.segment.SegmentAPI",
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        ),
    )


def get_collection(name: str | None = None):
    name = name or get_config()["indexing"]["collection"]
    return _client().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(chunks: list[Chunk], collection_name: str | None = None) -> int:
    if not chunks:
        return 0
    collection = get_collection(collection_name)
    batch = get_config()["indexing"]["batch_size"]
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        collection.upsert(
            ids=[c.chunk_id for c in part],
            documents=[c.text for c in part],
            embeddings=embed_texts([c.text for c in part]),
            metadatas=[{"doc_id": c.doc_id, "page": c.page, "source": c.source} for c in part],
        )
    logger.info("Upserted %d chunks into %r", len(chunks), collection.name)
    return len(chunks)


def list_documents(collection_name: str | None = None) -> dict[str, int]:
    """Return {doc_id: chunk_count} for everything indexed."""
    collection = get_collection(collection_name)
    result = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in result["metadatas"]:
        counts[meta["doc_id"]] = counts.get(meta["doc_id"], 0) + 1
    return counts


def all_chunks(collection_name: str | None = None) -> list[Chunk]:
    """Load every chunk back (used by BM25 to build its index)."""
    collection = get_collection(collection_name)
    result = collection.get(include=["documents", "metadatas"])
    return [
        Chunk(
            chunk_id=cid,
            doc_id=meta["doc_id"],
            page=int(meta["page"]),
            text=text,
            source=meta.get("source", "digital"),
        )
        for cid, text, meta in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
    ]
