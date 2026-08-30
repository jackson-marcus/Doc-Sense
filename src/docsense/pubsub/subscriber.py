"""Ingestion as an observable, incrementally durable pipeline.

`ingest_pdf` is one call that returns when everything is done. For a short
digital PDF that is fine. For a long scanned one it is not: OCR runs per page
and dominates the wall clock, so the caller sits on a silent connection with no
idea whether the job is progressing or wedged.

Worse, it is all-or-nothing in the way that matters least. If the store rejects
a batch two thirds of the way through, the exception discards the report and the
caller learns nothing about how far it got - even though the batches that
already landed are still sitting in the collection, indexed and searchable.

This runs the same work as a sequence of observable steps: each batch is
upserted on its own and reported as it lands, and a failure names the batch it
died on and how much survived it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from docsense.indexing.chunker import chunk_document
from docsense.indexing.store import upsert_chunks
from docsense.ingestion.loader import load_document
from docsense.pubsub.broker import CHANNEL, PubSubBroker
from docsense.retrieval.hybrid import invalidate_bm25_cache
from docsense.settings import get_config

STAGES: tuple[str, ...] = ("load", "chunk", "index")


@dataclass(frozen=True, slots=True)
class IngestEvent:
    """One observable step of an ingest."""

    stage: str
    payload: dict[str, Any]
    elapsed_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "elapsed_ms": self.elapsed_ms, **self.payload}


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingest achieved, including a partial one."""

    doc_id: str
    pages: int
    ocr_pages: int
    chunks_total: int
    chunks_indexed: int
    batches_indexed: int
    failed_batch: int | None = None
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.failed_batch is None and self.chunks_indexed == self.chunks_total

    @property
    def indexed_fraction(self) -> float:
        if self.chunks_total == 0:
            return 0.0
        return round(self.chunks_indexed / self.chunks_total, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "pages": self.pages,
            "ocr_pages": self.ocr_pages,
            "chunks_total": self.chunks_total,
            "chunks_indexed": self.chunks_indexed,
            "batches_indexed": self.batches_indexed,
            "indexed_fraction": self.indexed_fraction,
            "complete": self.complete,
            "failed_batch": self.failed_batch,
            "error": self.error,
        }


class IngestSubscriber:
    """Ingests every document published to the channel, reporting as it goes."""

    def __init__(self, broker: PubSubBroker, channel: str = CHANNEL) -> None:
        self.channel = channel
        self.journal: list[IngestReport] = []
        self.rejected: list[dict[str, Any]] = []
        broker.subscribe(channel, self.on_message)

    def on_message(self, message: dict[str, Any]) -> None:
        try:
            events = list(self.iter_events(message))
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            self.rejected.append({"frame": dict(message), "error": str(exc)})
            return
        data = events[-1].payload["report"]
        # as_dict() also carries derived keys, so rebuild from declared fields only.
        names = {field.name for field in fields(IngestReport)}
        self.journal.append(IngestReport(**{k: v for k, v in data.items() if k in names}))

    def iter_events(self, message: dict[str, Any]) -> Iterator[IngestEvent]:
        """Yield each step as it completes. A generator, so progress is real."""
        path = Path(str(message["path"]))
        if not path.exists():
            raise FileNotFoundError(f"no such document: {path}")

        started = time.perf_counter()
        document = load_document(path)
        yield IngestEvent(
            "load",
            {
                "doc_id": document.doc_id,
                "pages": len(document.pages),
                "ocr_pages": document.n_ocr_pages,
            },
            _ms_since(started),
        )

        started = time.perf_counter()
        chunks = chunk_document(document)
        yield IngestEvent(
            "chunk", {"doc_id": document.doc_id, "chunks": len(chunks)}, _ms_since(started)
        )

        batch_size = int(get_config()["indexing"]["batch_size"])
        indexed = batches = 0
        failed_batch: int | None = None
        error: str | None = None

        for number, start in enumerate(range(0, len(chunks), batch_size)):
            part = chunks[start : start + batch_size]
            started = time.perf_counter()
            try:
                # Upserted one batch at a time on purpose: whatever lands stays
                # landed, so a later failure cannot un-index earlier pages.
                indexed += upsert_chunks(part)
            except Exception as exc:
                failed_batch, error = number, f"{type(exc).__name__}: {exc}"
                yield IngestEvent(
                    "failed",
                    {"batch": number, "chunks_indexed": indexed, "error": error},
                    _ms_since(started),
                )
                break
            batches += 1
            yield IngestEvent(
                "index",
                {
                    "batch": number,
                    "chunks_indexed": indexed,
                    "chunks_total": len(chunks),
                    "progress": round(indexed / len(chunks), 4) if chunks else 0.0,
                },
                _ms_since(started),
            )

        if indexed:
            # The BM25 side is rebuilt from the store, so it only needs telling
            # once, and only if something actually landed.
            invalidate_bm25_cache()

        report = IngestReport(
            doc_id=document.doc_id,
            pages=len(document.pages),
            ocr_pages=document.n_ocr_pages,
            chunks_total=len(chunks),
            chunks_indexed=indexed,
            batches_indexed=batches,
            failed_batch=failed_batch,
            error=error,
        )
        yield IngestEvent("complete", {"report": report.as_dict()}, 0.0)


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
