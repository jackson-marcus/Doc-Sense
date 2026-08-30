"""WebSocket-shaped handler: one document frame in, one ingest report out."""

from __future__ import annotations

from typing import Any

from docsense.pubsub.broker import CHANNEL, PubSubBroker
from docsense.pubsub.subscriber import IngestReport, IngestSubscriber


class IngestRejectedError(ValueError):
    """The frame did not name a document that could be ingested."""


class WebSocketHandler:
    """Publishes a document to the broker and returns what the ingest achieved."""

    def __init__(self, broker: PubSubBroker | None = None) -> None:
        self.broker = broker or PubSubBroker()
        self.subscriber = IngestSubscriber(self.broker)

    def handle_frame(self, frame: dict[str, Any]) -> IngestReport:
        before = len(self.subscriber.journal)
        self.broker.publish(CHANNEL, frame)
        if len(self.subscriber.journal) == before:
            rejected = self.subscriber.rejected
            raise IngestRejectedError(rejected[-1]["error"] if rejected else "unknown")
        return self.subscriber.journal[-1]
