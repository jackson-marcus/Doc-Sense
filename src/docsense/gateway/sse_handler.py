"""SSE-shaped handler: report ingestion progress while it is still happening."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from docsense.gateway.ws_handler import WebSocketHandler
from docsense.pubsub.broker import PubSubBroker


class SseHandler:
    """Projects an ingest into the event stream an uploader can watch."""

    def __init__(self, broker: PubSubBroker | None = None) -> None:
        self.ws = WebSocketHandler(broker=broker)

    def stream(self, frame: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            for event in self.ws.subscriber.iter_events(frame):
                yield {"event": event.stage, "data": event.as_dict()}
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            yield {"event": "rejected", "data": {"reason": str(exc)}}

    def as_wire_format(self, frame: dict[str, Any]) -> Iterator[str]:
        for item in self.stream(frame):
            payload = json.dumps(item["data"], default=str)
            yield f"event: {item['event']}\ndata: {payload}\n\n"
