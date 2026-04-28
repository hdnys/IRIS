"""Tiny synchronous pub/sub for pipeline lifecycle events.

Used for observability (logging, metrics, debug overlays). NOT part of the
data flow — the orchestrator drives that explicitly through DataPool calls.
Subscribers should treat events as advisory: missing one must never break
the pipeline.

Why synchronous: the pipeline is already split into clear stages, and we
want stage-end events to fire in deterministic order (e.g. for a debug UI
that paints state in real time). Async would buy us nothing here and would
complicate test fixtures.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class EventBus:
    """Topic-based pub/sub with exception isolation.

    Topics are plain strings agreed on by the orchestrator and subscribers
    (e.g. ``"static_updated"``, ``"dynamic_complete"``, ``"validation_error"``).
    No registration of topics is needed — emit() with an unknown topic is
    just a no-op, which is exactly the behavior we want for optional
    observers.
    """

    def __init__(self) -> None:
        # defaultdict means we never have to check "is this topic known?";
        # subscribe() and emit() can both treat any string as a valid topic.
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        """Register a handler for a topic. Order of registration = order of invocation."""
        self._subs[topic].append(handler)

    def emit(self, topic: str, payload: Any = None) -> None:
        """Fire ``payload`` to every subscriber of ``topic``, swallowing exceptions.

        We deliberately suppress all exceptions from subscribers. Observability
        code (logging, metrics) failing must NEVER take the pipeline down with
        it — that would mean a bug in a debug overlay could blind the user.
        Subscribers are expected to handle their own errors; if they don't,
        the event silently drops.
        """
        for handler in self._subs.get(topic, []):
            try:
                handler(payload)
            except Exception:
                pass
