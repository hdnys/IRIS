"""Adapter contract.

Every model implementation lives behind this interface. The orchestrator
never imports a concrete model — it looks adapters up by role string in
the registry and calls .run() on whatever the YAML configured.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    version: str = "0.0.0"
    writes: list[str] = []
    reads: list[str] = []

    def __init__(self, role: str, **kwargs: Any) -> None:
        self.role = role
        self.config = kwargs

    def load(self) -> None:
        """Optional: load weights, warm caches. Called once at registry build time."""
        return None

    def unload(self) -> None:
        """Optional: release resources. Called on orchestrator shutdown."""
        return None

    @abstractmethod
    def run(self, frame: Any, pool_snapshot: dict, **kwargs: Any) -> Any:
        """Pure function over (frame, read-only snapshot) → this adapter's slice.

        - Dynamic adapters (objdet, face, ocr, ...): return the payload that
          will be written under ``dynamic.<role>``. Must conform to the schema.
        - VLM adapter: ``kwargs['mode']`` is either 'static' (return a partial
          dict of static fields) or 'describe' (return the description string).
        """
        ...
