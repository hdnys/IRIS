"""DataPool: the single shared state for the IRIS pipeline.

The pool is the only piece of mutable state in the system. Adapters do not
read or write it directly; the orchestrator mediates every update so that
stale results from a previous frame cannot leak into the current context.
"""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema.json"


class DataPool:
    def __init__(self, user_profile: dict, schema_path: Path = SCHEMA_PATH):
        with open(schema_path) as f:
            self._schema = json.load(f)
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "static": {},
            "dynamic": {},
            "user_profile": user_profile,
            # extension beyond schema: per-role observability
            "model_meta": {},
        }
        self._reset_static()

    def _reset_static(self) -> None:
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
        self._state["dynamic"] = {}

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._state)

    def current_frame_id(self) -> str:
        with self._lock:
            return self._state["static"].get("frame_id", "")

    def update_static(self, static_partial: dict, frame_id: str, timestamp: float) -> None:
        with self._lock:
            if static_partial.get("is_new_scene"):
                self._reset_static()
            self._state["static"].update(static_partial)
            self._state["static"]["frame_id"] = frame_id
            self._state["static"]["timestamp"] = timestamp

    def update_dynamic(self, key: str, payload: Any, frame_id: str) -> bool:
        # Drop writes whose frame_id no longer matches the current frame —
        # a slow adapter must not pollute the next frame's context.
        with self._lock:
            if frame_id != self._state["static"].get("frame_id"):
                return False
            self._state["dynamic"][key] = payload
            return True

    def update_meta(self, role: str, adapter: str, latency_ms: float,
                    status: str, version: str = "") -> None:
        with self._lock:
            self._state["model_meta"][role] = {
                "adapter": adapter,
                "version": version,
                "latency_ms": latency_ms,
                "status": status,
            }

    def validate(self) -> None:
        with self._lock:
            schema_subset = {
                k: v for k, v in self._state.items()
                if k in {"static", "dynamic", "user_profile"}
            }
            jsonschema.validate(schema_subset, self._schema)
