"""Benchmark the configured VLM adapter on a single image.

Builds the VLM directly from ``config/iris.yaml`` (no orchestrator, no other
adapters), runs the static and describe passes, and reports per-pass latency
plus VRAM peak sampled from ``nvidia-smi``.

Usage from project root::

    python -m Moteur.bench_vlm path/to/image.jpg
    python -m Moteur.bench_vlm path/to/image.jpg --gpu 0 --runs 3
    python -m Moteur.bench_vlm                       # synthetic 640x480 frame

The first run includes weight load into VRAM; pass ``--runs 3`` to also see
the warm latency (Ollama keeps the model resident per ``keep_alive``).
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "iris.yaml"


class VRAMSampler:
    """Polls ``nvidia-smi`` in a daemon thread and tracks the peak MiB used."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.1) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak: Optional[int] = None

    @staticmethod
    def query(gpu_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        """Single shot. Returns (used_mib, total_mib) or (None, None) on failure."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode().strip()
            used, total = (int(x.strip()) for x in out.split(","))
            return used, total
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            return None, None

    def start(self) -> None:
        self._stop.clear()
        self._peak = None

        def loop() -> None:
            while not self._stop.is_set():
                used, _ = self.query(self.gpu_index)
                if used is not None:
                    self._peak = used if self._peak is None else max(self._peak, used)
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self._peak


def build_vlm(cfg: dict):
    """Instantiate the VLM adapter from the YAML's ``adapters.vlm`` block."""
    spec = cfg["adapters"]["vlm"]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["class"])
    vlm = cls(role="vlm", **spec.get("params", {}))
    vlm.load()
    return vlm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="path to test image (default: 640x480 black frame)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index (default 0)")
    parser.add_argument("--runs", type=int, default=1, help="how many static+describe iterations to run")
    args = parser.parse_args()

    # ---- Load config + frame -------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: cannot read image {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print(f"Frame: {frame.shape[1]}x{frame.shape[0]} (HxW={frame.shape[0]}x{frame.shape[1]})")
    print(f"Adapter spec: {cfg['adapters']['vlm']}")

    # ---- Baseline VRAM (before Ollama loads anything) ------------------------
    baseline, total = VRAMSampler.query(args.gpu)
    if baseline is None:
        print(f"GPU {args.gpu}: nvidia-smi unavailable — VRAM tracking disabled")
    else:
        print(f"GPU {args.gpu}: baseline {baseline} / {total} MiB used")

    # ---- Build adapter (verifies Ollama + model is pulled) -------------------
    print("\nBuilding VLM (load() pings Ollama and checks the model tag)...")
    t0 = time.perf_counter()
    vlm = build_vlm(cfg)
    print(f"  load(): {(time.perf_counter() - t0) * 1000:.0f} ms")

    after_load, _ = VRAMSampler.query(args.gpu)
    if after_load is not None and baseline is not None:
        print(f"  VRAM after load(): {after_load} MiB ({after_load - baseline:+d} vs baseline)")

    # ---- Build a minimal pool snapshot the adapter can read ------------------
    user_profile = cfg.get("user_profile", {
        "vision_profile": "low_vision",
        "preferred_language": "en-US",
        "verbosity": "standard",
    })

    snap = {
        "static": {},
        "dynamic": {},
        "user_profile": user_profile,
        "model_meta": {},
    }

    # ---- Run N iterations of static + describe -------------------------------
    for i in range(1, args.runs + 1):
        print(f"\n=== Run {i}/{args.runs} ===")

        # Static pass
        sampler = VRAMSampler(gpu_index=args.gpu)
        sampler.start()
        t0 = time.perf_counter()
        try:
            static_out = vlm.run(frame, snap, mode="static")
        finally:
            peak = sampler.stop()
        elapsed = time.perf_counter() - t0
        peak_str = (
            f"{peak} MiB ({peak - baseline:+d} vs baseline)"
            if (peak is not None and baseline is not None) else (f"{peak} MiB" if peak else "n/a")
        )
        print(f"  static  : {elapsed * 1000:6.0f} ms   |   VRAM peak: {peak_str}")
        if i == 1:
            print(f"  static output: {json.dumps(static_out, indent=2)}")

        # Mirror the orchestrator: route ``scene_description`` from the static
        # return to dynamic.scene_description so the describe pass picks it up.
        static_for_pool = dict(static_out)
        scene_desc = static_for_pool.pop("scene_description", "")

        snap["static"] = {
            "frame_id": f"bench-{i}",
            "timestamp": time.time() * 1000.0,
            **static_for_pool,
        }
        # Inject a couple of fake dynamic entries plus the scene description.
        snap["dynamic"] = {
            "scene_description": scene_desc,
            "objects": [{"label": "chair", "confidence": 0.9}],
            "faces": [],
        }
        if i == 1 and scene_desc:
            print(f"  scene_description: {scene_desc}")

        # Describe pass
        sampler = VRAMSampler(gpu_index=args.gpu)
        sampler.start()
        t0 = time.perf_counter()
        try:
            description = vlm.run(None, snap, mode="describe")
        finally:
            peak = sampler.stop()
        elapsed = time.perf_counter() - t0
        peak_str = (
            f"{peak} MiB ({peak - baseline:+d} vs baseline)"
            if (peak is not None and baseline is not None) else (f"{peak} MiB" if peak else "n/a")
        )
        print(f"  describe: {elapsed * 1000:6.0f} ms   |   VRAM peak: {peak_str}")
        if i == 1:
            print(f"  description: {description}")

    # ---- Final VRAM (model still held by keep_alive) -------------------------
    final, _ = VRAMSampler.query(args.gpu)
    if final is not None and baseline is not None:
        print(f"\nGPU {args.gpu}: final {final} MiB ({final - baseline:+d} vs baseline, "
              f"held until keep_alive expires)")


if __name__ == "__main__":
    main()
