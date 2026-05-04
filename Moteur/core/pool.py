"""DataPool: the single shared state for the IRIS pipeline.

The pool is the only piece of mutable state in the system. Adapters do not
read or write it directly — the orchestrator mediates every update so that
stale results from a previous frame cannot leak into the current context.

The pool holds three schema-defined sections (static / dynamic / user_profile)
plus a non-schema ``model_meta`` extension used purely for observability.
"""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

# Resolve the schema path relative to this file: parents[0] = core/, parents[1] = Moteur/.
# Doing it once at import time means tests and runtime both pick up the same schema
# regardless of the current working directory.
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema.json"


class DataPool:
    """Thread-safe, schema-validated context store shared across the pipeline.

    Adapters never construct or mutate this directly. The orchestrator owns
    the lifecycle: it builds one DataPool per session, hands a read-only
    ``snapshot()`` to each adapter, and routes adapter outputs back through
    ``update_static`` / ``update_dynamic``.
    """

    def __init__(self, user_profile: dict, schema_path: Path = SCHEMA_PATH):
        # Load the JSON schema once. We re-use it on every validate() call.
        with open(schema_path) as f:
            self._schema = json.load(f)

        # RLock (re-entrant) rather than Lock: lets the same thread call
        # multiple pool methods inside one critical section if we ever need to
        # (e.g. snapshot() inside a larger update). A plain Lock would deadlock.
        self._lock = threading.RLock()

        # Initial state. ``user_profile`` is provided by the caller and stays
        # constant across the session; ``static`` and ``dynamic`` get reset on
        # every new scene. ``model_meta`` is our own per-role observability
        # extension — see validate() for why it is excluded from schema checks.
        self._state: dict[str, Any] = {
            "static": {},
            "dynamic": {},
            "user_profile": user_profile,
            "model_meta": {},
        }

        # Populate ``static`` with safe defaults so the very first VLM pass has
        # something coherent to read instead of a blank dict.
        self._reset_static()

    def _reset_static(self) -> None:
        """Reset static + dynamic to the defaults used at session start or new scene.

        Called both from __init__ (initial state) and from update_static() when
        the VLM signals ``is_new_scene: true``. Defaults are chosen so every
        ``required`` field in the schema is populated — that way the pool
        validates even before any adapter has run.
        """
        self._state["static"] = {
            "frame_id": "",
            "timestamp": 0,
            "is_new_scene": True,
            "scene_type": "unknown",
            "lighting_condition": "unknown",
            "motion_level": "unknown",
            "run_object_detection": False,
            "run_face_recognition": False,
            "run_emotion_detection": False,
            "run_ocr": False,
            "run_depth_estimation": False,
            "people_present": False,
            "people_count": 0,
            "text_visible": False,
            "hazard_detected": False,
            "priority_context": "general",
        }
        # Dynamic is wiped on scene change: previous-scene face encodings or
        # OCR text would just confuse the LLM pass. Better to start empty.
        self._state["dynamic"] = {}

    def snapshot(self) -> dict:
        """Return a deep copy of the current state.

        Adapters get this snapshot as input. The deep copy is critical:
        without it, a parallel adapter could mutate a nested dict and corrupt
        another adapter's view of the world mid-frame.
        """
        with self._lock:
            return deepcopy(self._state)

    def current_frame_id(self) -> str:
        """Frame currently being processed. Adapters use this to detect cancellation."""
        with self._lock:
            return self._state["static"].get("frame_id", "")

    def begin_frame(self, frame_id: str, timestamp: float) -> None:
        """Stamp the pool for a new frame and clear stale dynamic state.

        Called by the orchestrator at the top of every frame so all parallel
        adapter writes (gated on frame_id by ``update_dynamic``) target the
        same frame. Static fields are left at their session defaults — the
        new pipeline does not derive any static state from the VLM, and
        scene-change detection lives in the capture layer.
        """
        with self._lock:
            self._state["dynamic"] = {}
            self._state["static"]["frame_id"] = frame_id
            self._state["static"]["timestamp"] = timestamp

    def update_static(self, static_partial: dict, frame_id: str, timestamp: float) -> None:
        """Apply the VLM's static-pass output. Triggers a scene reset if requested.

        ``static_partial`` is merged on top of the existing static fields, so
        the VLM can return only the keys that actually changed. The orchestrator
        always passes ``frame_id`` and ``timestamp`` so we can stamp them
        consistently here rather than trusting the VLM to set them.
        """
        with self._lock:
            # Reset BEFORE merging so the partial's own values win and aren't
            # immediately wiped by _reset_static's defaults.
            if static_partial.get("is_new_scene"):
                self._reset_static()
            self._state["static"].update(static_partial)
            # Always overwrite frame_id / timestamp from the orchestrator's
            # authoritative values — the VLM should not be picking these.
            self._state["static"]["frame_id"] = frame_id
            self._state["static"]["timestamp"] = timestamp

    def update_dynamic(self, key: str, payload: Any, frame_id: str) -> bool:
        """Write an adapter's output into ``dynamic[key]``, gated on frame_id.

        Returns True on success, False if the write was rejected because the
        pool has already moved on to a newer frame. This is the core defense
        against stale results: a slow OCR call for frame N must NOT pollute
        frame N+1's context. The orchestrator records the rejection in
        ``model_meta`` as ``status='stale'`` so we can spot adapters that are
        consistently too slow.
        """
        with self._lock:
            if frame_id != self._state["static"].get("frame_id"):
                return False
            self._state["dynamic"][key] = payload
            return True

    def update_meta(self, role: str, adapter: str, latency_ms: float,
                    status: str, version: str = "") -> None:
        """Record observability info for a role's last run.

        Not part of the schema — purely for debugging and metrics. ``status``
        is ``"ok"``, ``"stale"``, or ``"error:<msg>"``.
        """
        with self._lock:
            self._state["model_meta"][role] = {
                "adapter": adapter,
                "version": version,
                "latency_ms": latency_ms,
                "status": status,
            }

    def validate(self) -> None:
        """Validate the schema-covered subset of the pool.

        ``model_meta`` is intentionally excluded: it is our own extension and
        does not appear in schema.json. Validation failures raise; the
        orchestrator catches them and continues into the LLM pass anyway —
        a partially-valid context is better than no description at all.
        """
        with self._lock:
            schema_subset = {
                k: v for k, v in self._state.items()
                if k in {"static", "dynamic", "user_profile"}
            }
            jsonschema.validate(schema_subset, self._schema)
