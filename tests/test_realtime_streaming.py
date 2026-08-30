"""Ingestion as observable, incrementally durable work.

The property under test is not "events are emitted". It is that a batch which
lands stays landed: if the store fails partway through a document, the pages
already indexed are still searchable and the report says exactly how far it got.
"""

from __future__ import annotations

import json

import pytest

import docsense.pubsub.subscriber as subscriber_module
from docsense.gateway.sse_handler import SseHandler
from docsense.gateway.ws_handler import IngestRejectedError, WebSocketHandler
from docsense.indexing.store import list_documents
from docsense.pubsub.broker import CHANNEL, PubSubBroker
from docsense.pubsub.subscriber import IngestSubscriber
from docsense.settings import get_config


@pytest.fixture()
def one_chunk_batches():
    """Force one chunk per batch so batch boundaries are observable."""
    indexing = get_config()["indexing"]
    original = indexing["batch_size"]
    indexing["batch_size"] = 1
    yield
    indexing["batch_size"] = original


def test_events_arrive_in_order_and_end_with_a_report(digital_pdf):
    events = list(IngestSubscriber(PubSubBroker()).iter_events({"path": str(digital_pdf)}))
    names = [e.stage for e in events]

    assert names[0] == "load"
    assert names[1] == "chunk"
    assert set(names[2:-1]) == {"index"}
    assert names[-1] == "complete"


def test_load_reports_pages_and_ocr_usage(digital_pdf):
    events = list(IngestSubscriber(PubSubBroker()).iter_events({"path": str(digital_pdf)}))
    load = events[0].payload

    assert load["pages"] == 2
    assert load["ocr_pages"] == 0, "a digital PDF needs no OCR"
    assert load["doc_id"]


def test_progress_climbs_to_everything_indexed(digital_pdf, one_chunk_batches):
    events = list(IngestSubscriber(PubSubBroker()).iter_events({"path": str(digital_pdf)}))
    indexed = [e.payload for e in events if e.stage == "index"]
    total = events[1].payload["chunks"]

    assert len(indexed) == total, "one batch per chunk was configured"
    assert [p["chunks_indexed"] for p in indexed] == list(range(1, total + 1))
    assert indexed[-1]["progress"] == 1.0

    report = events[-1].payload["report"]
    assert report["complete"] is True
    assert report["chunks_indexed"] == report["chunks_total"] == total


def test_nothing_is_indexed_until_the_stream_is_consumed(digital_pdf):
    """The generator must be lazy, or the progress it reports is retrospective."""
    events = IngestSubscriber(PubSubBroker()).iter_events({"path": str(digital_pdf)})

    assert next(events).stage == "load"
    assert list_documents() == {}, "loading a document must not index it"

    assert next(events).stage == "chunk"
    assert list_documents() == {}, "chunking must not index it either"

    list(events)
    assert list_documents(), "draining the stream does index it"


def test_a_failed_batch_keeps_everything_that_already_landed(
    digital_pdf, one_chunk_batches, monkeypatch
):
    real_upsert = subscriber_module.upsert_chunks
    calls = {"n": 0}

    def flaky(chunks, collection_name=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("chroma went away")
        return real_upsert(chunks, collection_name)

    monkeypatch.setattr(subscriber_module, "upsert_chunks", flaky)
    events = list(IngestSubscriber(PubSubBroker()).iter_events({"path": str(digital_pdf)}))
    report = events[-1].payload["report"]

    assert report["failed_batch"] == 1, "batches are zero-indexed; the second one died"
    assert "chroma went away" in report["error"]
    assert report["complete"] is False
    assert report["chunks_indexed"] == 1
    assert 0.0 < report["indexed_fraction"] < 1.0

    # The point of upserting per batch: the survivor is really in the store,
    # queryable, even though the ingest as a whole failed.
    assert sum(list_documents().values()) == 1


def test_the_failure_event_names_the_batch(digital_pdf, one_chunk_batches, monkeypatch):
    def always_fails(chunks, collection_name=None):
        raise RuntimeError("disk full")

    monkeypatch.setattr(subscriber_module, "upsert_chunks", always_fails)
    events = list(IngestSubscriber(PubSubBroker()).iter_events({"path": str(digital_pdf)}))

    failed = [e for e in events if e.stage == "failed"]
    assert len(failed) == 1, "it must stop at the first failure, not keep hammering"
    assert failed[0].payload["batch"] == 0
    assert failed[0].payload["chunks_indexed"] == 0
    assert list_documents() == {}


def test_broker_delivers_documents_to_the_subscriber(digital_pdf):
    broker = PubSubBroker()
    subscriber = IngestSubscriber(broker)

    broker.publish(CHANNEL, {"path": str(digital_pdf)})
    assert len(subscriber.journal) == 1
    assert subscriber.journal[0].complete
    assert subscriber.rejected == []


def test_a_missing_document_is_rejected_without_killing_the_subscriber(digital_pdf, tmp_path):
    broker = PubSubBroker()
    subscriber = IngestSubscriber(broker)

    broker.publish(CHANNEL, {"path": str(tmp_path / "nope.pdf")})
    broker.publish(CHANNEL, {})
    assert len(subscriber.rejected) == 2
    assert subscriber.journal == []

    broker.publish(CHANNEL, {"path": str(digital_pdf)})
    assert len(subscriber.journal) == 1, "a good document after bad ones must still ingest"


def test_websocket_raises_on_a_frame_it_cannot_ingest(tmp_path):
    with pytest.raises(IngestRejectedError):
        WebSocketHandler().handle_frame({"path": str(tmp_path / "missing.pdf")})


def test_sse_streams_the_stages_and_the_report(digital_pdf):
    events = list(SseHandler().stream({"path": str(digital_pdf)}))

    assert events[0]["event"] == "load"
    assert events[-1]["event"] == "complete"
    assert events[-1]["data"]["report"]["complete"] is True


def test_sse_rejection_is_streamed_not_raised(tmp_path):
    events = list(SseHandler().stream({"path": str(tmp_path / "missing.pdf")}))
    assert len(events) == 1 and events[0]["event"] == "rejected"
    assert "missing.pdf" in events[0]["data"]["reason"]


def test_sse_wire_format_is_parseable(digital_pdf):
    frames = list(SseHandler().as_wire_format({"path": str(digital_pdf)}))

    assert frames[0].startswith("event: load\ndata: {")
    assert frames[0].endswith("\n\n")
    assert json.loads(frames[0].split("data: ", 1)[1].strip())["pages"] == 2


def test_upload_stream_endpoint_reports_progress(digital_pdf, tmp_path):
    from fastapi.testclient import TestClient

    from docsense.api.main import create_app

    cfg = get_config()["ingestion"]
    original = cfg["raw_dir"]
    cfg["raw_dir"] = str(tmp_path / "raw")
    try:
        client = TestClient(create_app())
        with open(digital_pdf, "rb") as handle:
            response = client.post(
                "/upload/stream",
                files={"file": ("acme-report.pdf", handle, "application/pdf")},
            )
        assert response.status_code == 200
        names = [line[7:] for line in response.text.splitlines() if line.startswith("event: ")]
        assert names[0] == "load" and names[-1] == "complete"
        assert "index" in names
    finally:
        cfg["raw_dir"] = original


def test_upload_stream_rejects_non_pdf(tmp_path):
    from fastapi.testclient import TestClient

    from docsense.api.main import create_app

    client = TestClient(create_app())
    response = client.post(
        "/upload/stream", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422
