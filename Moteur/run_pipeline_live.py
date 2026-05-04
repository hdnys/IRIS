"""Live IRIS pipeline driven by the laptop webcam.

Reads frames from the camera at native FPS, runs the configured adapter
stack (Ollama VLM + SFace + objdet stub) on the latest frame in a worker
thread, and speaks the narration through TTS.

Inference is heavily gated to avoid wasted compute. A new inference run is
queued only when ALL of these are true:

* the inference worker is idle (not running the previous frame)
* the TTS worker is silent (we don't talk over ourselves)
* the current frame's dHash differs from the last *inferred* frame's dHash
  by at least ``pipeline.similarity_threshold`` bits

dHash (difference-hash) on a Gaussian-blurred 16x16 grayscale resize is far
more stable against webcam sensor noise / auto-exposure micro-flicker than
plain average-hash, so a static scene reliably stops re-triggering inference.

Usage from project root::

    python -m Moteur.run_pipeline_live
    python -m Moteur.run_pipeline_live --camera 1
    python -m Moteur.run_pipeline_live --no-similarity   # always submit
    python -m Moteur.run_pipeline_live --no-tts          # silent

Controls:
    q / ESC — quit
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from Moteur.core.orchestrator import Orchestrator
from Moteur.core.pool import DataPool
from Moteur.core.registry import AdapterRegistry

CONFIG_PATH = Path(__file__).parent / "config" / "iris.yaml"

# Display colours (BGR)
_GREEN = (0, 200, 0)
_RED   = (0, 0, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


# ---------------------------------------------------------------------------
# Scene-change detection (dHash on a blurred downsample)
# ---------------------------------------------------------------------------

def scene_hash(frame: np.ndarray, size: int = 16) -> np.ndarray:
    """Difference-hash on Gaussian-blurred grayscale.

    dHash compares neighbouring pixel intensities rather than absolute values,
    so the hash is largely insensitive to overall brightness drift (auto-
    exposure) and small sensor noise. Returns a (size*size)-bit fingerprint
    as a uint8 numpy array of 0/1. For size=16 that's 256 bits.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 5x5 blur kills most per-pixel sensor noise before hashing.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).astype(np.uint8).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


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
        self._stop = threading.Event()
        self._last_spoken = ""
        self._engine = None

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
        # Either currently speaking, or a line is waiting to start.
        return self._busy.is_set() or not self._queue.empty()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)  # unblock the get()

    def run(self) -> None:
        if not self._enabled:
            return
        # Windows SAPI lives on COM; pyttsx3 picks SAPI5 on Win32. Initialize
        # the apartment for this thread before touching any SAPI object.
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
            while not self._stop.is_set():
                text = self._queue.get()
                if text is None or self._stop.is_set():
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
# Inference worker
# ---------------------------------------------------------------------------

class InferenceWorker(threading.Thread):
    """Runs the orchestrator on submitted frames, one at a time.

    The main loop is responsible for *not* submitting while ``is_busy()`` —
    this worker doesn't queue, it just processes whatever's pending. After a
    successful run it stores the result and the dHash of the inferred frame
    so the main loop can compare future frames against the *last described*
    scene rather than the last submitted one (drift-free gating).
    """

    def __init__(self, orch: Orchestrator) -> None:
        super().__init__(daemon=True, name="iris-infer")
        self._orch = orch
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_hash: Optional[np.ndarray] = None
        self._latest_result: Optional[dict] = None
        self._last_inferred_hash: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._wakeup = threading.Event()

    def submit(self, frame: np.ndarray, frame_hash: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame
            self._pending_hash = frame_hash
        self._wakeup.set()

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def latest_result(self) -> Optional[dict]:
        with self._lock:
            return self._latest_result

    def last_inferred_hash(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._last_inferred_hash

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()

    def run(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop.is_set():
                break
            with self._lock:
                frame = self._pending_frame
                fhash = self._pending_hash
                self._pending_frame = None
                self._pending_hash = None
            if frame is None:
                continue

            self._busy.set()
            try:
                t0 = time.perf_counter()
                description = self._orch.process_frame(frame)
                ms = (time.perf_counter() - t0) * 1000
                snap = self._orch.pool.snapshot()
                dyn = snap.get("dynamic", {})
                result = {
                    "description": description or "",
                    "scene":       dyn.get("scene_description", "") or "",
                    "faces":       dyn.get("face_recognition", []) or [],
                    "objects":     dyn.get("object_detection", []) or [],
                    "ms":          ms,
                }
                with self._lock:
                    self._latest_result = result
                    self._last_inferred_hash = fhash
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


def draw_overlay(frame: np.ndarray, result: Optional[dict],
                 cam_fps: float, infer_busy: bool, tts_busy: bool,
                 hamming_dist: Optional[int]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if result:
        for f in result["faces"]:
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

    state = []
    if infer_busy:
        state.append("INFER")
    if tts_busy:
        state.append("TTS")
    state_str = " ".join(state) if state else "idle"

    hud = [
        f"cam fps : {cam_fps:5.1f}",
        f"state   : {state_str}",
        f"hamming : {hamming_dist if hamming_dist is not None else '-'}",
    ]
    if result:
        hud.append(f"infer   : {result['ms']:5.0f} ms")
        hud.append(f"faces   : {len(result['faces'])}  objs : {len(result['objects'])}")
        scene = result["scene"]
        if scene:
            short = scene if len(scene) <= 70 else scene[:70] + "…"
            hud.append(f"scene   : {short}")

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--camera", type=int, default=1, help="webcam index (default 0)")
    parser.add_argument("--no-similarity", action="store_true",
                        help="disable scene-change gate")
    parser.add_argument("--no-tts", action="store_true", help="disable text-to-speech")
    parser.add_argument("--tts-rate", type=int, default=175, help="TTS words-per-minute")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pipeline_cfg = cfg.get("pipeline", {})
    sim_threshold = int(pipeline_cfg.get("similarity_threshold", 25))
    use_similarity = not args.no_similarity

    print("Building registry + orchestrator…")
    pool = DataPool(user_profile=cfg["user_profile"])
    registry = AdapterRegistry.from_config(cfg)
    orch = Orchestrator(pool, registry, max_workers=pipeline_cfg.get("max_workers", 4))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        orch.shutdown()
        raise SystemExit(f"Could not open camera {args.camera}")

    print(f"Camera {args.camera} open. Warming up (15 frames)…")
    for _ in range(15):
        cap.read()

    tts = TTSWorker(rate=args.tts_rate, enabled=not args.no_tts)
    tts.start()

    worker = InferenceWorker(orch)
    worker.start()

    print(
        f"Live. similarity_threshold={sim_threshold} bits, "
        f"tts={'on' if not args.no_tts else 'off'}. q/ESC to quit.\n"
    )

    fps_t0 = time.perf_counter()
    fps_n = 0
    cam_fps = 0.0
    last_announced = ""
    last_hamming: Optional[int] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed — exiting.")
                break

            curr_hash = scene_hash(frame)

            # --- Inference gate -------------------------------------------------
            # Only queue a new inference when worker AND tts are idle AND the
            # scene has actually changed since the last described frame.
            should_submit = (
                not worker.is_busy()
                and not tts.is_busy()
            )
            if should_submit and use_similarity:
                last_hash = worker.last_inferred_hash()
                if last_hash is None:
                    last_hamming = None
                else:
                    last_hamming = hamming(curr_hash, last_hash)
                    if last_hamming < sim_threshold:
                        should_submit = False
            if should_submit:
                worker.submit(frame, curr_hash)

            # --- Pick up freshly-finished narration and hand to TTS ------------
            result = worker.latest_result()
            if result and result["description"] and result["description"] != last_announced:
                last_announced = result["description"]
                print(f"[{result['ms']:5.0f} ms] {result['description']}")
                tts.speak(result["description"])

            # cam fps over a 1s window — purely for the HUD.
            fps_n += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                cam_fps = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now

            display = draw_overlay(
                frame, result, cam_fps,
                worker.is_busy(), tts.is_busy(), last_hamming,
            )
            cv2.imshow("IRIS — live pipeline", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        print("\nShutting down…")
        worker.stop()
        worker.join(timeout=2.0)
        tts.stop()
        tts.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        orch.shutdown()


if __name__ == "__main__":
    main()
