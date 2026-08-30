"""Fan-out broker for the docs.pages channel."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

CHANNEL = "docs.pages"


class PubSubBroker:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, channel: str, message: dict[str, Any]) -> int:
        self.published.append((channel, dict(message)))
        for callback in list(self._subs[channel]):
            callback(message)
        return len(self._subs[channel])

    def subscribe(self, channel: str, callback: Callable[[dict[str, Any]], None]) -> None:
        self._subs[channel].append(callback)
