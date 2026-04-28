"""Plugin registry: maps role strings to adapter instances based on config.

Swapping a model is a one-line YAML change — point the role at a different
``module``/``class``. No orchestrator or pool code changes.
"""
from __future__ import annotations

import importlib
from typing import Optional

from Moteur.adapters.base import ModelAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, role: str, adapter: ModelAdapter) -> None:
        self._adapters[role] = adapter

    def get(self, role: str) -> Optional[ModelAdapter]:
        return self._adapters.get(role)

    def has(self, role: str) -> bool:
        return role in self._adapters

    def all(self) -> dict[str, ModelAdapter]:
        return dict(self._adapters)

    @classmethod
    def from_config(cls, config: dict) -> "AdapterRegistry":
        """Build a registry from an ``adapters:`` block.

        Expected shape::

            adapters:
              vlm:
                module: Moteur.adapters.vlm_stub
                class:  VLMStub
                params: {}
              object_detection:
                module: Moteur.adapters.objdet_stub
                class:  ObjDetStub
                params: {}
        """
        registry = cls()
        for role, spec in config.get("adapters", {}).items():
            module = importlib.import_module(spec["module"])
            adapter_cls = getattr(module, spec["class"])
            adapter = adapter_cls(role=role, **spec.get("params", {}))
            adapter.load()
            registry.register(role, adapter)
        return registry
