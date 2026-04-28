"""Tiny synchronous pub/sub for pipeline lifecycle events.

Used for observability (logging, metrics, debug overlays). Not part of the
data flow — the orchestrator drives that explicitly.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._subs[topic].append(handler)

    def emit(self, topic: str, payload: Any = None) -> None:
        for handler in self._subs.get(topic, []):
            try:
                handler(payload)
            except Exception:
                # A misbehaving subscriber must never break the pipeline.
                pass
