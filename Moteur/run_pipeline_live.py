"""Live IRIS pipeline driven by the laptop webcam.

Reads frames from the camera at native FPS, runs the configured adapter
stack (Ollama VLM + SFace + YOLO) on the latest frame in a worker
thread, and speaks the narration through TTS.

Inference is gated by a multi-signal :class:`SceneGate` (see
``Moteur.core.scene_gate``). On every captured frame the gate evaluates:

* dHash distance vs the last committed frame (pixel-level change)
* Face count delta (YuNet)
* Identity-set change (SFace)
* Optical flow magnitude (instantaneous motion)
* MobileNet embedding cosine distance (semantic change)

The signals are combined into a weighted score; if the score exceeds the
configured threshold AND the inference worker is idle AND TTS is silent,
the frame is submitted. The committed reference advances only on successful
submission, so per-frame noise drift cannot accumulate past the threshold.

Gate evaluation runs in its own thread so the camera display stays smooth
even though the embedding + SFace passes total ~50-70 ms per frame.

Usage from project root::

    python -m Moteur.run_pipeline_live
    python -m Moteur.run_pipeline_live --camera 1
    python -m Moteur.run_pipeline_live --no-gate           # always submit
    python -m Moteur.run_pipeline_live --no-tts            # silent

Controls:
    q / ESC — quit
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from Moteur.core.face_learner import FaceLearner
from Moteur.core.orchestrator import Orchestrator
from Moteur.core.pool import DataPool
from Moteur.core.registry import AdapterRegistry
from Moteur.core.scene_gate import GateSignals, SceneGate

CONFIG_PATH = Path(__file__).parent / "config" / "iris.yaml"

# Display colours (BGR)
_GREEN = (0, 200, 0)
_RED   = (0, 0, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_OBJ   = (255, 200, 0)   # cyan-ish for object boxes (BGR)


# ---------------------------------------------------------------------------
# Text-to-speech worker
# ---------------------------------------------------------------------------

class TTSWorker(threading.Thread):
    """Speaks queued narration lines, one at a time, in a background thread.

    ``speak()`` replaces any pending (un-started) line with the new one — only
    the freshest narration is ever queued, so we never read an old line aloud
    after the scene has moved on. While a line is being spoken we expose
    ``is_busy() == True`` so the inference gate can wait us out.
    """

    def __init__(self, rate: int = 175, enabled: bool = True) -> None:
        super().__init__(daemon=True, name="iris-tts")
        self._rate = rate
        self._enabled = enabled
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._busy = threading.Event()
        # Note: cannot use `_stop` — it shadows threading.Thread._stop and
        # crashes on join() with `Event object is not callable`.
        self._stop_evt = threading.Event()
        self._last_spoken = ""

    def speak(self, text: str) -> None:
        if not self._enabled or not text or text == self._last_spoken:
            return
        # Drain any queued-but-not-started lines — keep only the newest.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(text)

    def is_busy(self) -> bool:
        return self._busy.is_set() or not self._queue.empty()

    def stop(self) -> None:
        self._stop_evt.set()
        self._queue.put(None)

    def run(self) -> None:
        if not self._enabled:
            return
        try:
            import pythoncom  # type: ignore
            pythoncom.CoInitialize()
        except Exception:
            pythoncom = None  # type: ignore
        try:
            import pyttsx3  # type: ignore
        except Exception as e:
            print(f"[TTS disabled — pyttsx3 import failed: {e}]")
            return

        try:
            while not self._stop_evt.is_set():
                text = self._queue.get()
                if text is None or self._stop_evt.is_set():
                    break
                self._busy.set()
                self._last_spoken = text
                # Re-init the engine for every utterance. pyttsx3's SAPI5
                # backend gets stuck after the first runAndWait() when called
                # from a non-main thread on Windows — subsequent say() calls
                # silently no-op. A fresh init costs ~150-300 ms but is the
                # only reliable way to keep speech firing on every line.
                engine = None
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self._rate)
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS error] {e}")
                finally:
                    if engine is not None:
                        try:
                            engine.stop()
                        except Exception:
                            pass
                        del engine
                    self._busy.clear()
        finally:
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Scene-gate worker
# ---------------------------------------------------------------------------

class GateWorker(threading.Thread):
    """Runs SceneGate.evaluate() async so the camera display stays smooth.

    Submitting a frame overwrites any pending un-evaluated frame — we only
    care about the freshest signals, never about a backlog. The main loop
    reads ``latest_signals()`` and, when it confirms it submitted the frame
    for inference, calls ``commit_latest()`` to advance the gate's
    "committed reference" to that frame.
    """

    def __init__(self, gate: SceneGate) -> None:
        super().__init__(daemon=True, name="iris-gate")
        self._gate = gate
        # Pending = (frame, faces_or_None, objects_or_None). The main loop
        # pre-computes SFace and reads ObjDetWorker.latest_objects() so both
        # results are forwarded here and never recomputed inside the gate.
        self._pending: Optional[tuple[np.ndarray, Optional[list], Optional[list]]] = None
        self._latest_signals: Optional[GateSignals] = None
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        # See TTSWorker note — never name this `_stop`.
        self._stop_evt = threading.Event()

    def submit(self, frame: np.ndarray, faces: Optional[list] = None,
               objects: Optional[list] = None) -> None:
        with self._lock:
            self._pending = (frame, faces, objects)
        self._wakeup.set()

    def latest_signals(self) -> Optional[GateSignals]:
        with self._lock:
            return self._latest_signals

    def commit_latest(self) -> None:
        """Promote the most recently evaluated frame to the new committed reference."""
        with self._lock:
            sig = self._latest_signals
        if sig is not None:
            self._gate.commit(sig)

    def stop(self) -> None:
        self._stop_evt.set()
        self._wakeup.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop_evt.is_set():
                break
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                continue
            frame, faces, objects = pending
            try:
                signals = self._gate.evaluate(frame, faces=faces, objects=objects)
            except Exception as e:
                print(f"[gate error] {e}")
                continue
            with self._lock:
                self._latest_signals = signals


# ---------------------------------------------------------------------------
# Live object-detection worker
# ---------------------------------------------------------------------------

class ObjDetWorker(threading.Thread):
    """Runs the YOLO adapter async on submitted frames.

    Same submit-latest-only pattern as :class:`GateWorker`: a new submission
    overwrites any pending un-evaluated frame, so the worker always processes
    the freshest pixels and the main display loop never blocks on YOLO.
    """

    def __init__(self, adapter) -> None:
        super().__init__(daemon=True, name="iris-objdet")
        self._adapter = adapter
        self._pending: Optional[np.ndarray] = None
        self._latest: list[dict] = []
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_evt = threading.Event()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame
        self._wakeup.set()

    def latest_objects(self) -> list[dict]:
        with self._lock:
            return list(self._latest)

    def stop(self) -> None:
        self._stop_evt.set()
        self._wakeup.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop_evt.is_set():
                break
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            try:
                objs = self._adapter.run(frame, {}) or []
            except Exception as e:
                print(f"[objdet error] {e}")
                continue
            with self._lock:
                self._latest = objs


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------

class InferenceWorker(threading.Thread):
    """Runs the orchestrator on submitted frames, one at a time."""

    def __init__(self, orch: Orchestrator) -> None:
        super().__init__(daemon=True, name="iris-infer")
        self._orch = orch
        self._pending: Optional[tuple[np.ndarray, Optional[list], Optional[list]]] = None
        self._latest_result: Optional[dict] = None
        self._lock = threading.Lock()
        self._busy = threading.Event()
        # See TTSWorker note — never name this `_stop`.
        self._stop_evt = threading.Event()
        self._wakeup = threading.Event()

    def submit(self, frame: np.ndarray,
               objects: Optional[list] = None,
               faces: Optional[list] = None) -> None:
        with self._lock:
            self._pending = (frame, objects, faces)
        self._wakeup.set()

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def latest_result(self) -> Optional[dict]:
        with self._lock:
            return self._latest_result

    def stop(self) -> None:
        self._stop_evt.set()
        self._wakeup.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop_evt.is_set():
                break
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                continue
            frame, objects, faces = pending

            self._busy.set()
            try:
                t0 = time.perf_counter()
                description = self._orch.process_frame(frame, objects=objects, faces=faces)
                ms = (time.perf_counter() - t0) * 1000
                snap = self._orch.pool.snapshot()
                dyn = snap.get("dynamic", {})
                result = {
                    "description": description or "",
                    "scene":       dyn.get("scene_description", "") or "",
                    "faces":       dyn.get("face_recognition", []) or [],
                    "objects":     dyn.get("object_detection", []) or [],
                    "ms":          ms,
                    # Whole pool snapshot, kept around for the pool-viewer
                    # window. Snapshot is a deep copy so it's safe to ship.
                    "snap":        snap,
                }
                with self._lock:
                    self._latest_result = result
            finally:
                self._busy.clear()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def wrap_text(text: str, max_px: int, scale: float, thick: int) -> list[str]:
    out: list[str] = []
    line = ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw <= max_px:
            line = candidate
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def draw_face_boxes(frame: np.ndarray, faces: Optional[list]) -> np.ndarray:
    out = frame.copy()
    if not faces:
        return out

    for f in faces:
        bb = f.get("bounding_box", {})
        x = int(bb.get("x", 0)); y = int(bb.get("y", 0))
        fw = int(bb.get("width", 0)); fh = int(bb.get("height", 0))
        name = f.get("person_id", "?")
        conf = f.get("confidence", 0.0)
        color = _GREEN if name != "Unknown" else _RED
        cv2.rectangle(out, (x, y), (x + fw, y + fh), color, 2)
        label = f"{name} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, max(th, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2)
    return out


def draw_object_boxes(frame: np.ndarray, objects: Optional[list]) -> np.ndarray:
    """Overlay YOLO detection boxes from the latest inference result.

    Boxes lag the live camera by one inference cycle (the orchestrator runs
    them on the gated frame, not on every captured one). That's an acceptable
    trade against the cost of running YOLO synchronously per frame.
    """
    out = frame.copy()
    if not objects:
        return out

    for o in objects:
        bb = o.get("bounding_box", {})
        x = int(bb.get("x", 0)); y = int(bb.get("y", 0))
        ow = int(bb.get("width", 0)); oh = int(bb.get("height", 0))
        if ow <= 0 or oh <= 0:
            continue
        label = o.get("label", "?")
        conf = o.get("confidence", 0.0)
        cv2.rectangle(out, (x, y), (x + ow, y + oh), _OBJ, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, max(0, y - th - 6)), (x + tw + 4, y), _OBJ, -1)
        cv2.putText(out, text, (x + 2, max(th, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BLACK, 1)
    return out


def draw_overlay(frame: np.ndarray, result: Optional[dict],
                 cam_fps: float, infer_busy: bool, tts_busy: bool,
                 signals: Optional[GateSignals],
                 trigger_score: float,
                 live_faces: Optional[list] = None,
                 live_objects: Optional[list] = None) -> np.ndarray:
    # Draw the freshest face data we have. ``live_faces`` is the SFace
    # output computed on the current frame in the main loop — drawing those
    # boxes keeps them locked to the user's actual head. The previous
    # version drew ``result["faces"]`` which lags by one full inference
    # (~750 ms cold, several seconds during TTS), so the box stuck where
    # the face *was*. Fall back to the inference result only if live data
    # isn't available yet (very first frames before SFace has run once).
    faces_to_draw = live_faces if live_faces is not None else (
        result.get("faces", []) if result else []
    )
    # Prefer the live YOLO worker's freshest output; fall back to the lagged
    # inference result only if the live worker hasn't produced anything yet.
    objects_to_draw = live_objects if live_objects is not None else (
        result.get("objects", []) if result else []
    )
    out = draw_object_boxes(frame, objects_to_draw)
    out = draw_face_boxes(out, faces_to_draw)
    h, w = out.shape[:2]

    state = []
    if infer_busy: state.append("INFER")
    if tts_busy:   state.append("TTS")
    state_str = " ".join(state) if state else "idle"

    hud = [
        f"cam fps : {cam_fps:5.1f}",
        f"state   : {state_str}",
    ]
    if signals is not None:
        hud.append(
            f"score   : {signals.score:.2f} / {trigger_score:.2f}  "
            f"{'TRIG' if signals.triggered else '----'}"
        )
        hud.append(
            f"  dhash : {signals.dhash_distance:3d}b   "
            f"flow  : {signals.flow_magnitude:5.2f}px"
        )
        hud.append(
            f"  faces : {signals.face_count} (Δ {signals.face_count_delta:+d})  "
            f"id-chg: {'Y' if signals.identity_changed else 'N'}"
        )
        hud.append(
            f"  objs  : {signals.object_count} (Δ {signals.object_count_delta:+d})  "
            f"chg: {'Y' if signals.objects_changed else 'N'}"
        )
        hud.append(f"  embed : {signals.embedding_distance:.3f}")
    if result:
        hud.append(f"infer   : {result['ms']:5.0f} ms")
        scene = result["scene"]
        if scene:
            short = scene if len(scene) <= 70 else scene[:70] + "…"
            hud.append(f"scene   : {short}")
        objs = (live_objects if live_objects is not None
                else (result.get("objects", []) or []))
        if objs:
            # Top-3 most-confident detections, condensed.
            top = ", ".join(
                f"{o.get('label','?')}({o.get('confidence',0):.2f},"
                f"{o.get('position','?')},{o.get('size','?')})"
                for o in objs[:3]
            )
            extra = "" if len(objs) <= 3 else f" +{len(objs) - 3}"
            hud.append(f"objs({len(objs)}): {top}{extra}")

    for i, line in enumerate(hud):
        y = 22 + i * 22
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BLACK, 3)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1)

    if result and result["description"]:
        wrap = wrap_text(result["description"], w - 20, 0.65, 1)
        strip_h = 26 * len(wrap) + 16
        cv2.rectangle(out, (0, h - strip_h), (w, h), _BLACK, -1)
        for i, line in enumerate(wrap):
            cv2.putText(out, line, (10, h - strip_h + 26 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 1)
    return out


def draw_learn_overlay(frame: np.ndarray, name: str, message: str) -> np.ndarray:
    """Big banner overlay used while the FaceLearner is active."""
    out = frame.copy()
    w = out.shape[1]
    cv2.rectangle(out, (0, 0), (w, 70), _BLACK, -1)
    cv2.putText(out, f"LEARNING FACE: {name}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 255), 2)
    cv2.putText(out, message, (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1)
    cv2.putText(out, "ESC = abort", (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return out


def render_pool_image(snap: dict, width: int = 760, max_lines: int = 60) -> np.ndarray:
    """Format the pool snapshot as a JSON image suitable for cv2.imshow.

    Long lines are truncated and the dump is capped at ``max_lines`` so the
    window stays a manageable size on smaller screens.
    """
    text = json.dumps(snap, indent=2, default=str, ensure_ascii=False)
    lines = text.split("\n")
    truncated_lines = len(lines) > max_lines
    if truncated_lines:
        lines = lines[:max_lines]

    line_h = 16
    extra = 1 if truncated_lines else 0
    height = max(120, line_h * (len(lines) + extra) + 20)
    img = np.full((height, width, 3), 18, dtype=np.uint8)  # near-black bg

    # Tiny title bar
    cv2.putText(img, "DataPool snapshot (post-inference)",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 200, 255), 1)

    for i, raw in enumerate(lines):
        line = raw if len(raw) <= 100 else raw[:97] + "..."
        # Crude colouring: keys vs values, helps readability.
        color = (210, 210, 210)
        stripped = line.lstrip()
        if stripped.startswith('"') and '":' in stripped:
            color = (180, 220, 160)   # keys = greenish
        elif stripped.startswith(("[", "{", "}", "]")):
            color = (140, 200, 255)   # structural = blue
        cv2.putText(img, line, (8, 32 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    if truncated_lines:
        cv2.putText(img, "... (truncated)", (8, 32 + len(lines) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)
    return img


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_scene_gate(cfg: dict, sface_adapter) -> SceneGate:
    sg_cfg = cfg.get("scene_gate", {}) or {}
    weights = sg_cfg.get("weights", {}) or {}
    thresh  = sg_cfg.get("thresholds", {}) or {}
    return SceneGate(
        sface_adapter=sface_adapter,
        dhash_max_bits   = int(thresh.get("dhash_max_bits", 64)),
        flow_max         = float(thresh.get("flow_max", 5.0)),
        embedding_max    = float(thresh.get("embedding_max", 0.4)),
        w_dhash          = float(weights.get("dhash", 0.30)),
        w_face_count     = float(weights.get("face_count", 0.20)),
        w_identity       = float(weights.get("identity", 0.20)),
        w_flow           = float(weights.get("flow", 0.10)),
        w_embedding      = float(weights.get("embedding", 0.20)),
        w_objects        = float(weights.get("objects", 0.15)),
        trigger_score    = float(sg_cfg.get("trigger_score", 0.20)),
        mobilenet_path   = sg_cfg.get(
            "mobilenet_path",
            "models/image_classification_mobilenetv2_2022apr.onnx",
        ),
        download_if_missing = bool(sg_cfg.get("download_if_missing", True)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-gate", action="store_true",
                        help="disable the scene gate — submit every captured frame")
    parser.add_argument("--no-tts", action="store_true", help="disable TTS")
    parser.add_argument("--tts-rate", type=int, default=175, help="TTS words-per-minute")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pipeline_cfg = cfg.get("pipeline", {})
    annotate_vlm_faces   = bool(pipeline_cfg.get("annotate_vlm_faces",   True))
    annotate_vlm_objects = bool(pipeline_cfg.get("annotate_vlm_objects", True))

    print("Building registry + orchestrator…")
    pool = DataPool(user_profile=cfg["user_profile"])
    registry = AdapterRegistry.from_config(cfg)
    orch = Orchestrator(pool, registry, max_workers=pipeline_cfg.get("max_workers", 4))

    # Share the SFaceAdapter between the orchestrator and the gate so YuNet
    # and the SFace ONNX model are loaded only once.
    sface_adapter = registry.get("face_recognition")
    gate = build_scene_gate(cfg, sface_adapter)
    print("Loading SceneGate (downloading MobileNet on first run)…")
    gate.load()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        orch.shutdown()
        raise SystemExit(f"Could not open camera {args.camera}")

    print(f"Camera {args.camera} open. Warming up (15 frames)…")
    for _ in range(15):
        cap.read()

    tts          = TTSWorker(rate=args.tts_rate, enabled=not args.no_tts)
    gate_worker  = GateWorker(gate)
    infer_worker = InferenceWorker(orch)
    od_worker    = ObjDetWorker(registry.get("object_detection"))
    tts.start(); gate_worker.start(); infer_worker.start(); od_worker.start()

    use_gate = not args.no_gate
    trigger_score = float(cfg.get("scene_gate", {}).get("trigger_score", 0.20))
    gallery_dir = Path(
        cfg.get("adapters", {}).get("face_recognition", {})
        .get("params", {}).get("gallery_dir", "data")
    )
    print(
        f"Live. gate={'on' if use_gate else 'off'}, trigger_score={trigger_score:.2f}, "
        f"tts={'on' if not args.no_tts else 'off'}.\n"
        "Keys: q/ESC = quit   |   l = learn a new face   |   p = toggle pool window\n"
    )


    fps_t0 = time.perf_counter()
    fps_n = 0
    cam_fps = 0.0
    last_announced = ""
    last_pool_img: Optional[np.ndarray] = None
    show_pool = True
    learner: Optional[FaceLearner] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed — exiting.")
                break

            # ------------------------------------------------------------
            # Mode A: face-learning. Skip inference entirely; the learner
            # owns the frame and gives the user spoken / on-screen guidance.
            # ------------------------------------------------------------
            if learner is not None:
                message, done = learner.step(frame)
                display = draw_learn_overlay(frame, learner.name, message)
                cv2.imshow("IRIS — live pipeline", display)
                if done:
                    print(f"[learn] finished: {message}")
                    learner = None
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC aborts learn mode
                    print("[learn] aborted by user")
                    learner = None
                continue

            # ------------------------------------------------------------
            # Mode B: normal pipeline.
            # ------------------------------------------------------------
            # Run SFace synchronously on every captured frame so face boxes
            # track the user's actual head in real time (no stale boxes from
            # a previous inference). ~30 ms per frame, drops display fps to
            # ~25 — fully acceptable, and the only way to keep the boxes
            # locked to live motion. The gate and the inference worker reuse
            # this result so SFace isn't computed twice per frame.
            try:
                live_faces = sface_adapter.run(frame, {}) if sface_adapter else []
                if live_faces is None:
                    live_faces = []
            except Exception as e:
                print(f"[sface error] {e}")
                live_faces = []

            # Snapshot last frame's YOLO result before submitting so the gate
            # always gets objects that correspond to the committed reference
            # (one frame behind at most — acceptable for change detection).
            live_objects = od_worker.latest_objects()
            gate_worker.submit(frame, faces=live_faces, objects=live_objects)
            od_worker.submit(frame)
            signals = gate_worker.latest_signals()

            should_submit = (
                not infer_worker.is_busy()
                and not tts.is_busy()
            )
            if should_submit and use_gate:
                if signals is None or not signals.triggered:
                    should_submit = False

            if should_submit:
                frame_for_vlm = frame
                if annotate_vlm_objects:
                    frame_for_vlm = draw_object_boxes(frame_for_vlm, live_objects)
                if annotate_vlm_faces:
                    frame_for_vlm = draw_face_boxes(frame_for_vlm, live_faces)
                infer_worker.submit(frame_for_vlm, objects=live_objects, faces=live_faces)
                if signals is not None:
                    gate_worker.commit_latest()

            result = infer_worker.latest_result()
            if result and result["description"] and result["description"] != last_announced:
                last_announced = result["description"]
                print(f"[{result['ms']:5.0f} ms] {result['description']}")
                tts.speak(result["description"])
                # Refresh the pool view exactly when a new narration lands —
                # avoids re-rendering the same image every frame.
                snap = result.get("snap")
                if isinstance(snap, dict):
                    last_pool_img = render_pool_image(snap)

            fps_n += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                cam_fps = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now

            display = draw_overlay(
                frame, result, cam_fps,
                infer_worker.is_busy(), tts.is_busy(),
                signals, trigger_score,
                live_faces=live_faces,
                live_objects=live_objects,
            )
            cv2.imshow("IRIS — live pipeline", display)
            if show_pool and last_pool_img is not None:
                cv2.imshow("IRIS — pool", last_pool_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                show_pool = not show_pool
                if not show_pool:
                    try:
                        cv2.destroyWindow("IRIS — pool")
                    except cv2.error:
                        pass
            if key == ord("l") and learner is None:
                # Briefly freezes the camera while the user types — keeping
                # the prompt in the terminal avoids reinventing a text-entry
                # widget on top of cv2.
                print("\n--- Learn a new face ---")
                name = input("Name (no spaces): ").strip()
                if name:
                    learner = FaceLearner(
                        name=name,
                        sface_adapter=sface_adapter,
                        speak=tts.speak,
                        gallery_dir=gallery_dir,
                    )
                else:
                    print("[learn] cancelled — empty name")
    finally:
        print("\nShutting down…")
        infer_worker.stop(); infer_worker.join(timeout=2.0)
        gate_worker.stop();  gate_worker.join(timeout=2.0)
        od_worker.stop();    od_worker.join(timeout=2.0)
        tts.stop();          tts.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        orch.shutdown()


if __name__ == "__main__":
    main()
