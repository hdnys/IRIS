"""Pipeline state machine.

This is the only component that knows the order of operations. Per frame:

  1. Begin frame             — stamp the pool with frame_id + timestamp and
                              clear last frame's dynamic state.
  2. Parallel fan-out        — VLM scene-description + every dynamic adapter
                              are dispatched concurrently to the same thread
                              pool. They write into ``dynamic[role]``;
                              ``update_dynamic`` gates on frame_id so stale
                              outputs are dropped.
  3. Schema validation       — assembled context is validated; failures are
                              logged via the EventBus but do not abort.
  4. Finalization            — the VLM scene output is mirrored as the final
                              narration string.

The VLM is now single-pass per frame: one multimodal call receives the image
plus both the scene and system instructions, and returns the narration text.
Step 2 is embarrassingly parallel: the VLM call and the dynamic adapters write
to disjoint dynamic.* keys.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from Moteur.core.events import EventBus
from Moteur.core.pool import DataPool
from Moteur.core.registry import AdapterRegistry


class Orchestrator:
    """Drives one frame through the pipeline."""

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
        # IPC cost of multiprocessing. The Ollama-based VLM adapter is
        # network-bound — also fine on threads.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_frame(self, frame: Any,
                      objects: Optional[list] = None,
                      faces: Optional[list] = None) -> Optional[str]:
        """Run one full pass on a single frame. Returns the narration string, or None.

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

        # Phase 1: stamp the pool so all parallel writes target this frame.
        self.pool.begin_frame(frame_id, ts)

        # Phase 2: parallel dispatch. Blocks until every submitted adapter
        # finishes (or raises). No timeout enforced here — add one at the
        # adapter level once we know real-world latencies.
        self._dispatch_parallel(frame, frame_id, objects=objects, faces=faces)

        # Phase 3: validate. Caught broadly because we deliberately want
        # final narration to continue even on partial data — better a degraded
        # description than silence.
        try:
            self.pool.validate()
        except Exception as e:
            self.events.emit("validation_error", {"error": str(e)})

        # Phase 4: mirror the VLM output as final narration.
        return self._finalize_vlm_output(frame_id)

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

    def _dispatch_parallel(self, frame: Any, frame_id: str,
                           objects: Optional[list] = None,
                           faces: Optional[list] = None) -> None:
        """Submit VLM scene + every other adapter to the executor concurrently.

        No gating: every configured non-VLM adapter runs every frame. The cost
        of always-running stubs is negligible; when real adapters land they
        should be cheap enough to keep this true (or short-circuit themselves).
        """
        snap = self.pool.snapshot()
        futures: dict = {}

        # VLM scene-description call (uses the image).
        vlm = self.registry.get("vlm")
        if vlm is None:
            self.events.emit("error", "no VLM adapter registered")
        else:
            futures[self._executor.submit(
                self._run_vlm_scene, vlm, frame, snap, frame_id, objects, faces
            )] = "vlm_scene"

        # Every other registered adapter — runs in parallel with the VLM.
        # If pre-computed results were supplied by the caller (live pipeline
        # already ran SFace + YOLO for display), write them straight into the
        # pool and skip re-running the adapter.  This prevents double-running
        # ONNX models that are already busy on the main thread, which was the
        # cause of the 1-second display freeze on inference trigger.
        for role, adapter in self.registry.all().items():
            if role == "vlm":
                continue
            if role == "face_recognition" and faces is not None:
                self.pool.update_dynamic("face_recognition", faces, frame_id)
                self.pool.update_meta(role, type(adapter).__name__, 0.0, "precomputed",
                                      version=getattr(adapter, "version", ""))
                continue
            if role == "object_detection" and objects is not None:
                self.pool.update_dynamic("object_detection", objects, frame_id)
                self.pool.update_meta(role, type(adapter).__name__, 0.0, "precomputed",
                                      version=getattr(adapter, "version", ""))
                continue
            futures[self._executor.submit(
                self._run_adapter, adapter, role, frame, snap, frame_id
            )] = role

        # Wait for everyone. We don't collect return values — each task
        # writes directly into the pool. as_completed is used (vs wait()) so
        # we can later add per-future timeout handling without restructuring.
        for _ in as_completed(futures):
            pass

        self.events.emit("dynamic_complete", self.pool.snapshot()["dynamic"])

    def _run_vlm_scene(self, vlm: Any, frame: Any, snapshot: dict, frame_id: str,
                       objects: Optional[list] = None,
                       faces: Optional[list] = None) -> None:
        """Run the VLM in static mode and route its string output to the pool.

        Static mode now returns the final narration string (see
        ``Moteur.adapters.vlm_ollama``). We write it to
        ``dynamic.scene_description`` so it sits alongside every other
        adapter's output. Same boundary semantics as ``_run_adapter``: must NOT
        raise. All exceptions become ``status="error:..."`` in model_meta.
        """
        t0 = time.time()
        status = "ok"
        try:
            scene_desc = vlm.run(frame, snapshot, mode="static",
                                  objects=objects, faces=faces)
            # Defensive: an adapter that returns a dict (older contract) or
            # None should not break the pool. Coerce to str.
            if not isinstance(scene_desc, str):
                scene_desc = "" if scene_desc is None else str(scene_desc)
            if scene_desc:
                written = self.pool.update_dynamic(
                    "scene_description", scene_desc, frame_id
                )
                if not written:
                    status = "stale"
        except Exception as e:
            status = f"error:{e}"
        self.pool.update_meta("vlm_scene", type(vlm).__name__,
                              (time.time() - t0) * 1000.0, status,
                              version=getattr(vlm, "version", ""))

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

    def _finalize_vlm_output(self, frame_id: str) -> Optional[str]:
        """Phase 4: finalize narration from the single VLM output.

        The VLM already produced narration in static mode (image + scene prompt
        + system prompt). We keep that text under both dynamic keys so existing
        consumers can read either ``scene_description`` or ``vlm_description``.
        """
        snap = self.pool.snapshot()
        # Frame supersession check: if a newer frame has already started while
        # we were dispatching, drop this narration.
        if snap["static"]["frame_id"] != frame_id:
            return None

        text = (snap.get("dynamic", {}).get("scene_description") or "").strip()
        if not text:
            return None

        self.pool.update_dynamic("vlm_description", text, frame_id)
        return text
