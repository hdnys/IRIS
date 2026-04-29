"""Pipeline state machine.

Sequence per frame:
  1. VLM static pass        — VLM reads prior context, writes static fields.
  2. Parallel dispatch      — for each ``run_*`` flag set true, fire the matching
                              adapter; results are written into ``dynamic`` keyed
                              by role, gated on frame_id to discard stale writes.
  3. Schema validation      — assembled context is validated; failures are
                              logged but do not abort the LLM pass.
  4. VLM describe pass      — same VLM, ``mode='describe'``, receives both the
                              previous-frame snapshot and the freshly assembled
                              one so it can describe what changed; produces the
                              natural-language description for TTS.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from Moteur.core.events import EventBus
from Moteur.core.pool import DataPool
from Moteur.core.registry import AdapterRegistry

# Static flag → adapter role. Single source of truth for dispatch.
DISPATCH_MAP = {
    "run_object_detection": "object_detection",
    "run_face_recognition": "face_recognition",
    "run_emotion_detection": "emotion_detection",
    "run_ocr": "ocr",
    "run_depth_estimation": "depth_estimation",
}


class Orchestrator:
    def __init__(self, pool: DataPool, registry: AdapterRegistry,
                 events: Optional[EventBus] = None, max_workers: int = 4) -> None:
        self.pool = pool
        self.registry = registry
        self.events = events or EventBus()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_frame(self, frame: Any) -> Optional[str]:
        """Run one full pass on a single frame. Returns the description, or None on error."""
        frame_id = uuid.uuid4().hex
        ts = time.time() * 1000.0

        # Capture the pool BEFORE any update so the describe pass can hand both
        # the old and new snapshots to the VLM and let it diff them itself.
        previous_snapshot = self.pool.snapshot()

        if not self._run_vlm_static(frame, frame_id, ts):
            return None

        self._dispatch_dynamic(frame, frame_id)

        try:
            self.pool.validate()
        except Exception as e:
            self.events.emit("validation_error", {"error": str(e)})

        return self._run_vlm_describe(frame_id, previous_snapshot)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        for adapter in self.registry.all().values():
            try:
                adapter.unload()
            except Exception:
                pass

    # --- internal pipeline steps ---

    def _run_vlm_static(self, frame: Any, frame_id: str, ts: float) -> bool:
        vlm = self.registry.get("vlm")
        if vlm is None:
            self.events.emit("error", "no VLM adapter registered")
            return False

        snap = self.pool.snapshot()
        t0 = time.time()
        try:
            static_partial = vlm.run(frame, snap, mode="static")
        except Exception as e:
            self.pool.update_meta("vlm_static", type(vlm).__name__,
                                  (time.time() - t0) * 1000.0, f"error:{e}")
            self.events.emit("vlm_static_error", str(e))
            return False

        # The VLM may include a free-form ``scene_description`` alongside the
        # orchestration flags. That belongs in the dynamic block (it's a model
        # output, not an orchestration field), so peel it off here before
        # update_static merges the rest.
        scene_desc = ""
        if isinstance(static_partial, dict):
            scene_desc = static_partial.pop("scene_description", "") or ""

        # Write the VLM's partial dict on top of the existing static fields.
        # This call may also reset the pool if is_new_scene is true.
        self.pool.update_static(static_partial, frame_id, ts)
        # Now that the pool's frame_id is set to ``frame_id``, update_dynamic
        # will accept this write (it gates on matching frame_id).
        if scene_desc:
            self.pool.update_dynamic("scene_description", scene_desc, frame_id)
        self.pool.update_meta("vlm_static", type(vlm).__name__,
                              (time.time() - t0) * 1000.0, "ok",
                              version=getattr(vlm, "version", ""))
        self.events.emit("static_updated", self.pool.snapshot()["static"])
        return True

    def _dispatch_dynamic(self, frame: Any, frame_id: str) -> None:
        snap = self.pool.snapshot()
        static = snap["static"]
        futures = {}
        for flag, role in DISPATCH_MAP.items():
            if not static.get(flag):
                continue
            adapter = self.registry.get(role)
            if adapter is None:
                self.events.emit("missing_adapter", role)
                continue
            futures[self._executor.submit(
                self._run_adapter, adapter, role, frame, snap, frame_id
            )] = role

        for _ in as_completed(futures):
            pass
        self.events.emit("dynamic_complete", self.pool.snapshot()["dynamic"])

    def _run_adapter(self, adapter: Any, role: str, frame: Any,
                     snapshot: dict, frame_id: str) -> None:
        t0 = time.time()
        status = "ok"
        try:
            payload = adapter.run(frame, snapshot)
            written = self.pool.update_dynamic(role, payload, frame_id)
            if not written:
                status = "stale"
        except Exception as e:
            status = f"error:{e}"
        self.pool.update_meta(role, type(adapter).__name__,
                              (time.time() - t0) * 1000.0, status,
                              version=getattr(adapter, "version", ""))

    def _run_vlm_describe(self, frame_id: str, previous_snapshot: dict) -> Optional[str]:
        vlm = self.registry.get("vlm")
        if vlm is None:
            return None
        snap = self.pool.snapshot()
        if snap["static"]["frame_id"] != frame_id:
            return None
        t0 = time.time()
        try:
            text = vlm.run(None, snap, mode="describe", previous_snapshot=previous_snapshot)
        except Exception as e:
            self.pool.update_meta("vlm_describe", type(vlm).__name__,
                                  (time.time() - t0) * 1000.0, f"error:{e}")
            return None
        self.pool.update_meta("vlm_describe", type(vlm).__name__,
                              (time.time() - t0) * 1000.0, "ok",
                              version=getattr(vlm, "version", ""))
        self.pool.update_dynamic("vlm_description", text, frame_id)
        return text
