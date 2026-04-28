"""Plugin registry: maps role strings to adapter instances based on config.

This is what makes IRIS model-agnostic. The orchestrator never imports a
concrete model — it asks the registry for whatever adapter is registered
under a given role. Swapping a real model in or out is a one-line YAML
change; no orchestrator or pool code is touched.

Roles are short strings agreed on across the codebase: ``"vlm"``,
``"object_detection"``, ``"face_recognition"``, ``"emotion_detection"``,
``"ocr"``, ``"depth_estimation"``. The orchestrator's DISPATCH_MAP defines
exactly which roles it knows how to dispatch on.
"""
from __future__ import annotations

import importlib
from typing import Optional

from Moteur.adapters.base import ModelAdapter


class AdapterRegistry:
    """In-memory map of role → adapter instance.

    Built once at startup from a YAML config and held for the lifetime of
    the orchestrator. Adapters are loaded eagerly (their ``load()`` runs
    during from_config) so that any model-loading failure surfaces at
    startup, not on the first frame the user sees.
    """

    def __init__(self) -> None:
        # Plain dict — registry has no concurrency requirements because it
        # is built once before the orchestrator starts and never mutated
        # after. If hot-swap is ever needed, add a lock at that point.
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, role: str, adapter: ModelAdapter) -> None:
        """Insert an adapter under ``role``. Replaces any existing entry."""
        self._adapters[role] = adapter

    def get(self, role: str) -> Optional[ModelAdapter]:
        """Look up an adapter. Returns None if no adapter is configured for the role,
        which the orchestrator treats as 'skip this stage' rather than an error.
        """
        return self._adapters.get(role)

    def has(self, role: str) -> bool:
        return role in self._adapters

    def all(self) -> dict[str, ModelAdapter]:
        """Defensive copy — callers (e.g. shutdown loop) should not mutate the live map."""
        return dict(self._adapters)

    @classmethod
    def from_config(cls, config: dict) -> "AdapterRegistry":
        """Build a registry from the parsed YAML config.

        Expected shape::

            adapters:
              vlm:
                module: Moteur.adapters.vlm_stub
                class:  VLMStub
                params: {}
              object_detection:
                module: Moteur.adapters.objdet_stub
                class:  ObjDetStub
                params: {model_path: "..."}

        ``params`` is forwarded to the adapter constructor as kwargs. Each
        adapter's ``load()`` runs immediately after construction so that
        model weights are warm before the first frame arrives.
        """
        registry = cls()
        for role, spec in config.get("adapters", {}).items():
            # Dynamic import lets us add new adapters without touching this
            # file. The module path in YAML drives discovery entirely.
            module = importlib.import_module(spec["module"])
            adapter_cls = getattr(module, spec["class"])

            # All adapters take ``role`` as their first positional kwarg
            # (see ModelAdapter.__init__) so they know what slice of the
            # pool they own.
            adapter = adapter_cls(role=role, **spec.get("params", {}))

            # Eager load: fail fast on bad weights / missing files / GPU issues
            # rather than crashing on the first frame in front of the user.
            adapter.load()

            registry.register(role, adapter)
        return registry
