"""Pipeline state machine.

This is the only component that knows the order of operations. Per frame:

  1. VLM static pass        — VLM reads prior context, writes static fields
                              (scene type, run_* flags, people_present, etc.).
  2. Parallel dispatch      — for each ``run_*`` flag set true, fire the
                              matching adapter on a thread pool. Results are
                              written into ``dynamic`` keyed by role and gated
                              on frame_id so stale outputs are dropped.
  3. Schema validation      — assembled context is validated; failures are
                              logged via the EventBus but do not abort.
  4. VLM describe pass      — same VLM, ``mode='describe'``, generates the
                              natural-language string for TTS.

Steps 1 and 4 are sequential because step 4 depends on the full pool. Step
2 is embarrassingly parallel: object detection, OCR, depth, etc. don't read
each other's outputs in this version of the pipeline.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from Moteur.core.events import EventBus
from Moteur.core.pool import DataPool
from Moteur.core.registry import AdapterRegistry

# Single source of truth for dispatch: which static flag activates which role.
# Adding a new dynamic adapter means adding one entry here PLUS a corresponding
# ``run_X`` field in schema.json. No other orchestrator changes required.
DISPATCH_MAP = {
    "run_object_detection": "object_detection",
    "run_face_recognition": "face_recognition",
    "run_emotion_detection": "emotion_detection",
    "run_ocr": "ocr",
    "run_depth_estimation": "depth_estimation",
}


class Orchestrator:
    """Drives one frame through the four pipeline stages."""

    def __init__(self, pool: DataPool, registry: AdapterRegistry,
                 events: Optional[EventBus] = None, max_workers: int = 4) -> None:
        self.pool = pool
        self.registry = registry
        # An EventBus is always present — callers can pass None and we make
        # one. This keeps emit() calls below from needing a None-check.
        self.events = events or EventBus()

        # ThreadPoolExecutor is the right choice for adapter parallelism:
        # most ONNX / Torch / OpenCV inference releases the GIL during
        # native compute, so threads achieve real parallelism without the
        # IPC cost of multiprocessing. If a future adapter is pure Python
        # CPU-bound, switch THAT adapter to a process pool — don't change
        # this one.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_frame(self, frame: Any) -> Optional[str]:
        """Run one full pass on a single frame. Returns the description, or None on failure.

        ``frame`` is whatever the capture layer hands us — typically an
        OpenCV BGR ndarray. Adapters are responsible for any conversion they
        need (RGB, resize, tensor packing) — the orchestrator stays format-
        agnostic so we don't have to revisit it when models change.
        """
        # uuid4 hex gives us a 32-char unique id per frame. Used to gate
        # writes against stale adapter results — see DataPool.update_dynamic.
        frame_id = uuid.uuid4().hex
        # Milliseconds since epoch matches the schema's ``timestamp`` field type.
        ts = time.time() * 1000.0

        # Stage 1: VLM static pass. If this fails there is no point
        # dispatching downstream models — they have no flags to act on.
        if not self._run_vlm_static(frame, frame_id, ts):
            return None

        # Stage 2: parallel fan-out. This blocks until all dispatched
        # adapters finish (or raise). No timeout enforced here — add one
        # at the adapter level once we know real-world latencies.
        self._dispatch_dynamic(frame, frame_id)

        # Stage 3: validate. Caught broadly because we deliberately want
        # the LLM pass to run even on partial data — better a degraded
        # description than silence.
        try:
            self.pool.validate()
        except Exception as e:
            self.events.emit("validation_error", {"error": str(e)})

        # Stage 4: describe. Returns the string to hand off to TTS.
        return self._run_vlm_describe(frame_id)

    def shutdown(self) -> None:
        """Release executor and unload every adapter. Called on app exit."""
        # cancel_futures=True so any queued (but not yet running) adapter
        # tasks are abandoned instead of forcing the user to wait through them.
        self._executor.shutdown(wait=False, cancel_futures=True)
        for adapter in self.registry.all().values():
            try:
                adapter.unload()
            except Exception:
                # One bad adapter must not prevent the others from unloading.
                pass

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _run_vlm_static(self, frame: Any, frame_id: str, ts: float) -> bool:
        """Stage 1: VLM examines the frame + prior context, sets static fields.

        Returns True on success. False means we abort the frame — no point
        running downstream adapters if we don't know what to dispatch.
        """
        vlm = self.registry.get("vlm")
        if vlm is None:
            # Fail loudly via the event bus — this is a config error, not
            # something to silently skip.
            self.events.emit("error", "no VLM adapter registered")
            return False

        # Adapter sees a snapshot, not the live pool — see DataPool.snapshot
        # for why deep-copy isolation matters.
        snap = self.pool.snapshot()
        t0 = time.time()
        try:
            static_partial = vlm.run(frame, snap, mode="static")
        except Exception as e:
            # Record the failure in model_meta and emit; do not crash.
            self.pool.update_meta("vlm_static", type(vlm).__name__,
                                  (time.time() - t0) * 1000.0, f"error:{e}")
            self.events.emit("vlm_static_error", str(e))
            return False

        # Write the VLM's partial dict on top of the existing static fields.
        # This call may also reset the pool if is_new_scene is true.
        self.pool.update_static(static_partial, frame_id, ts)
        self.pool.update_meta("vlm_static", type(vlm).__name__,
                              (time.time() - t0) * 1000.0, "ok",
                              version=getattr(vlm, "version", ""))
        # Snapshot the post-update state for subscribers (debug UIs etc.).
        self.events.emit("static_updated", self.pool.snapshot()["static"])
        return True

    def _dispatch_dynamic(self, frame: Any, frame_id: str) -> None:
        """Stage 2: fan out adapters in parallel based on the run_* flags.

        Each adapter gets the same snapshot (post-static-update) and writes
        its result into the live pool via update_dynamic. The pool itself
        gates writes on frame_id, so this method does not need to track
        "did we move on to a newer frame" — DataPool handles that.
        """
        snap = self.pool.snapshot()
        static = snap["static"]

        # Build a {Future: role} map so failure messages can mention the role.
        futures: dict = {}
        for flag, role in DISPATCH_MAP.items():
            # If the VLM did not request this stage, skip — even if an adapter
            # is registered. The VLM is the policy maker for "what is worth
            # running on this frame".
            if not static.get(flag):
                continue
            adapter = self.registry.get(role)
            if adapter is None:
                # Flag was set but no adapter is wired in. Common during dev
                # — emit and move on.
                self.events.emit("missing_adapter", role)
                continue
            futures[self._executor.submit(
                self._run_adapter, adapter, role, frame, snap, frame_id
            )] = role

        # We don't collect return values — _run_adapter writes directly to the
        # pool. We just need to wait for completion. Looping over as_completed
        # would also let us add a per-future timeout in the future.
        for _ in as_completed(futures):
            pass

        self.events.emit("dynamic_complete", self.pool.snapshot()["dynamic"])

    def _run_adapter(self, adapter: Any, role: str, frame: Any,
                     snapshot: dict, frame_id: str) -> None:
        """Run a single adapter on the executor and record result + meta.

        Runs in a worker thread. Must NOT raise: this method is the boundary
        between adapter code (which may fail in any number of ways) and the
        orchestrator (which must keep running). All exceptions are swallowed
        into the model_meta status field.
        """
        t0 = time.time()
        status = "ok"
        try:
            payload = adapter.run(frame, snapshot)
            # update_dynamic returns False if the frame has already advanced —
            # mark this run "stale" so we can detect chronically slow adapters.
            written = self.pool.update_dynamic(role, payload, frame_id)
            if not written:
                status = "stale"
        except Exception as e:
            status = f"error:{e}"
        self.pool.update_meta(role, type(adapter).__name__,
                              (time.time() - t0) * 1000.0, status,
                              version=getattr(adapter, "version", ""))

    def _run_vlm_describe(self, frame_id: str) -> Optional[str]:
        """Stage 4: VLM, now in 'describe' mode, generates the user-facing string.

        Returns None if (a) no VLM is registered, (b) the frame has been
        superseded mid-pipeline, or (c) the VLM call raises. The TTS layer
        treats None as "stay silent on this frame".
        """
        vlm = self.registry.get("vlm")
        if vlm is None:
            return None

        snap = self.pool.snapshot()
        # Frame supersession check: if a newer frame has already started its
        # static pass while we were in dispatch, this description would be
        # built from a mix of old + new state. Drop it.
        if snap["static"]["frame_id"] != frame_id:
            return None

        t0 = time.time()
        try:
            text = vlm.run(None, snap, mode="describe")
        except Exception as e:
            self.pool.update_meta("vlm_describe", type(vlm).__name__,
                                  (time.time() - t0) * 1000.0, f"error:{e}")
            return None

        self.pool.update_meta("vlm_describe", type(vlm).__name__,
                              (time.time() - t0) * 1000.0, "ok",
                              version=getattr(vlm, "version", ""))

        # Persist the description into the pool too — useful for logs, debug
        # overlays, and any subscriber that wants the final output without
        # listening to TTS.
        self.pool.update_dynamic("vlm_description", text, frame_id)
        return text
