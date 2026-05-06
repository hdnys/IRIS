"""IRIS Web API — FastAPI server that exposes the full pipeline over HTTP + WebSocket.

Endpoints
---------
GET  /                  — serve index.html
GET  /api/status        — running state + pool snapshot
GET  /api/profiles      — available vision profiles
GET  /api/friends       — registered people in data/
POST /api/start         — start the pipeline (returns immediately; init runs in thread)
POST /api/stop          — stop the pipeline
WS   /ws/video          — stream annotated JPEG frames (~30 fps)
WS   /ws/pool           — stream pool snapshot JSON (1 Hz)

Static mounts
-------------
/interface  — interface/ directory (HTML + CSS)
/data       — data/ directory (face image gallery)

Usage::

    python -m Moteur.api_server
    uvicorn Moteur.api_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from Moteur.core.orchestrator import Orchestrator
from Moteur.core.pool import DataPool
from Moteur.core.registry import AdapterRegistry
from Moteur.run_pipeline_live import (
    GateWorker,
    InferenceWorker,
    ObjDetWorker,
    TTSWorker,
    build_scene_gate,
    draw_face_boxes,
    draw_object_boxes,
    draw_overlay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_IRIS_ROOT   = Path(__file__).parent.parent
CONFIG_PATH  = Path(__file__).parent / "config" / "iris.yaml"
CONFIG_FILE  = _IRIS_ROOT / ".iris_config.json"
INTERFACE_DIR = _IRIS_ROOT / "interface"
DATA_DIR      = _IRIS_ROOT / "data"

AVAILABLE_PROFILES = {
    "total_blindness": {"label": "Total Blindness",   "description": "No light perception"},
    "low_vision":      {"label": "Low Vision",        "description": "Some residual vision"},
    "tunnel_vision":   {"label": "Tunnel Vision",     "description": "Central vision only"},
    "color_blindness": {"label": "Color Blindness",   "description": "Cannot distinguish colors"},
    "peripheral_loss": {"label": "Peripheral Loss",   "description": "Reduced side vision"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(config: dict) -> None:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[config] save failed: {e}")


def get_registered_friends() -> list[dict]:
    friends: list[dict] = []
    if DATA_DIR.exists():
        for person_dir in sorted(DATA_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            if images:
                friends.append({
                    "name":   person_dir.name,
                    "count":  len(images),
                    "images": [img.name for img in images],
                })
    return friends

# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------

class PipelineRunner(threading.Thread):
    """Runs the full IRIS pipeline in a background thread.

    Heavy initialisation (loading ONNX models, SceneGate, camera warm-up)
    happens inside ``run()`` so ``start()`` returns immediately.  The
    annotated frame and pool snapshot are exposed via thread-safe accessors
    so the WebSocket handlers can sample them without blocking the event loop.
    """

    def __init__(self, cfg: dict, camera_index: int = 0) -> None:
        super().__init__(daemon=True, name="iris-pipeline")
        self._cfg          = cfg
        self._camera_index = camera_index
        self._stop_evt     = threading.Event()

        # Exposed state — protected by their own locks.
        self._frame_lock:   threading.Lock    = threading.Lock()
        self._latest_jpeg:  Optional[bytes]   = None
        self._pool:         Optional[DataPool] = None  # set once init completes

    # ------------------------------------------------------------------
    # Public accessors (called from async context)
    # ------------------------------------------------------------------

    def latest_frame_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def latest_pool_snap(self) -> dict:
        pool = self._pool
        return pool.snapshot() if pool is not None else {}

    def stop(self) -> None:
        self._stop_evt.set()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg          = self._cfg
        pipeline_cfg = cfg.get("pipeline", {})
        annotate_vlm_faces   = bool(pipeline_cfg.get("annotate_vlm_faces",   True))
        annotate_vlm_objects = bool(pipeline_cfg.get("annotate_vlm_objects", True))
        infer_cooldown_s     = float(pipeline_cfg.get("infer_cooldown_s",    10.0))
        trigger_score        = float(cfg.get("scene_gate", {}).get("trigger_score", 0.40))

        # --- Build pipeline components ---
        pool = DataPool(user_profile=cfg["user_profile"])
        self._pool = pool

        registry = AdapterRegistry.from_config(cfg)
        orch     = Orchestrator(pool, registry,
                                max_workers=pipeline_cfg.get("max_workers", 4))

        sface = registry.get("face_recognition")
        gate  = build_scene_gate(cfg, sface)

        print("[pipeline] Loading SceneGate (may download MobileNet on first run)…")
        gate.load()

        tts         = TTSWorker(enabled=True)
        gate_worker = GateWorker(gate)
        infer       = InferenceWorker(orch)
        od          = ObjDetWorker(registry.get("object_detection"))

        # --- Open camera ---
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print(f"[pipeline] Cannot open camera {self._camera_index}")
            return

        print(f"[pipeline] Camera {self._camera_index} open. Warming up (15 frames)…")
        for _ in range(15):
            cap.read()

        tts.start()
        gate_worker.start()
        infer.start()
        od.start()
        print("[pipeline] All workers running.")

        last_infer_time = 0.0
        last_announced  = ""
        fps_t0 = time.perf_counter()
        fps_n  = 0
        cam_fps = 0.0

        try:
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    print("[pipeline] Camera read failed — stopping.")
                    break

                # SFace on every frame so face boxes track live motion.
                try:
                    live_faces = sface.run(frame, {}) if sface else []
                    if live_faces is None:
                        live_faces = []
                except Exception as e:
                    print(f"[sface] {e}")
                    live_faces = []

                live_objects = od.latest_objects()
                gate_worker.submit(frame, faces=live_faces, objects=live_objects)
                od.submit(frame)
                signals = gate_worker.latest_signals()

                # Inference gating — same logic as run_pipeline_live.
                should_submit = (
                    not infer.is_busy()
                    and not tts.is_busy()
                    and (time.perf_counter() - last_infer_time) >= infer_cooldown_s
                    and signals is not None
                    and signals.triggered
                )

                if should_submit:
                    frame_for_vlm = frame.copy()
                    if annotate_vlm_objects:
                        frame_for_vlm = draw_object_boxes(frame_for_vlm, live_objects)
                    if annotate_vlm_faces:
                        frame_for_vlm = draw_face_boxes(frame_for_vlm, live_faces)
                    infer.submit(frame_for_vlm, objects=live_objects, faces=live_faces)
                    last_infer_time = time.perf_counter()
                    gate_worker.commit_latest()

                result = infer.latest_result()
                if (result
                        and result["description"]
                        and result["description"] != last_announced):
                    last_announced = result["description"]
                    print(f"[{result['ms']:.0f} ms] {result['description']}")
                    tts.speak(result["description"])

                # FPS counter (updated every second).
                fps_n += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    cam_fps = fps_n / (now - fps_t0)
                    fps_n = 0
                    fps_t0 = now

                # Build annotated frame and encode to JPEG.
                display = draw_overlay(
                    frame, result, cam_fps,
                    infer.is_busy(), tts.is_busy(),
                    signals, trigger_score,
                    live_faces=live_faces,
                    live_objects=live_objects,
                )
                _, buf = cv2.imencode(".jpg", display,
                                      [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

        finally:
            cap.release()
            infer.stop();       infer.join(timeout=2.0)
            gate_worker.stop(); gate_worker.join(timeout=2.0)
            od.stop();          od.join(timeout=2.0)
            tts.stop();         tts.join(timeout=2.0)
            orch.shutdown()
            print("[pipeline] Stopped.")


# ---------------------------------------------------------------------------
# Shared runner state (module-level so lifespan + routes share it)
# ---------------------------------------------------------------------------

_runner: Optional[PipelineRunner] = None
_runner_lock = threading.Lock()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] IRIS API started.")
    yield
    # Graceful shutdown — stop any running pipeline.
    global _runner
    with _runner_lock:
        r, _runner = _runner, None
    if r is not None:
        r.stop()
        r.join(timeout=5.0)
    print("[api] IRIS API stopped.")


app = FastAPI(title="IRIS Web API", lifespan=lifespan)

# Static file mounts — must be registered before route definitions so
# FastAPI resolves /interface/... and /data/... before the wildcard routes.
if INTERFACE_DIR.exists():
    app.mount("/interface", StaticFiles(directory=INTERFACE_DIR, html=True),
              name="interface")
else:
    print(f"[api] Warning: interface directory not found at {INTERFACE_DIR}")

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
else:
    print(f"[api] Warning: data directory not found at {DATA_DIR}")


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    index = INTERFACE_DIR / "html" / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "IRIS API running — open /interface/html/index.html"}


@app.get("/api/status")
async def get_status():
    with _runner_lock:
        r = _runner
    running = r is not None and r.is_alive()
    return {
        "running": running,
        "pool":    r.latest_pool_snap() if running else None,
    }


@app.get("/api/profiles")
async def get_profiles():
    return {"profiles": AVAILABLE_PROFILES}


@app.get("/api/friends")
async def get_friends():
    friends = get_registered_friends()
    return {"friends": friends, "count": len(friends)}


@app.post("/api/start")
async def start_pipeline():
    global _runner
    with _runner_lock:
        if _runner is not None and _runner.is_alive():
            return {"error": "Pipeline already running", "status": "failed"}
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            r = PipelineRunner(cfg)
            r.start()
            _runner = r
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    return {"status": "started", "message": "Pipeline initialising in background"}


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge ``override`` into ``base`` in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


@app.get("/api/config")
async def get_config():
    """Return the full iris.yaml config as JSON."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@app.post("/api/config")
async def update_config(request: Request):
    """Deep-merge the posted JSON into iris.yaml and save."""
    try:
        body = await request.json()
    except Exception as e:
        return {"error": f"Invalid JSON: {e}", "status": "failed"}
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        _deep_merge(cfg, body)
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        return {"status": "saved"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/api/stop")
async def stop_pipeline():
    global _runner
    with _runner_lock:
        r, _runner = _runner, None
    if r is not None:
        r.stop()
        # Don't join here — it can take a few seconds; let it wind down.
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# WebSocket routes
# ---------------------------------------------------------------------------

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """Stream annotated JPEG frames from the running pipeline (~30 fps)."""
    await websocket.accept()
    try:
        while True:
            with _runner_lock:
                r = _runner
            if r is not None and r.is_alive():
                jpeg = r.latest_frame_jpeg()
                if jpeg is not None:
                    await websocket.send_bytes(jpeg)
            await asyncio.sleep(0.033)  # ~30 fps cap
    except (WebSocketDisconnect, Exception):
        pass


@app.websocket("/ws/pool")
async def websocket_pool(websocket: WebSocket):
    """Stream pool snapshot JSON once per second."""
    await websocket.accept()
    try:
        while True:
            with _runner_lock:
                r = _runner
            snap = r.latest_pool_snap() if (r is not None and r.is_alive()) else {}
            await websocket.send_text(json.dumps(snap, default=str))
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, Exception):
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
