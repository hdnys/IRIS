"""Live webcam test for the SFace face-recognition adapter.

Captures frames from the webcam, runs YuNet + SFace on each one, and draws
bounding boxes with names and confidence scores in a live OpenCV window.

Usage from project root::

    python -m Moteur.bench_face
    python -m Moteur.bench_face --camera 1       # non-default camera index
    python -m Moteur.bench_face --image path/to/photo.jpg   # still image mode

Controls (live mode):
    q / ESC  — quit
    s        — save current frame as data/<name>.jpeg and reload the gallery
               (prompts for the person's name in the terminal)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "iris.yaml"

# Colours
_GREEN = (0, 200, 0)
_RED   = (0, 0, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def build_adapter(cfg: dict):
    import importlib
    spec = cfg["adapters"]["face_recognition"]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["class"])
    adapter = cls(role="face_recognition", **spec.get("params", {}))
    adapter.load()
    return adapter


def draw_faces(frame: "np.ndarray", faces: list[dict]) -> "np.ndarray":
    out = frame.copy()
    for f in faces:
        bb   = f["bounding_box"]
        x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])
        name = f["person_id"]
        conf = f["confidence"]
        color = _GREEN if name != "Unknown" else _RED
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{name} ({conf:.2f})"
        # small filled rect behind text so it's readable on any background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, max(th, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 2)
    return out


def overlay_hud(frame: "np.ndarray", ms: float, n_faces: int, gallery: list[str]) -> None:
    lines = [
        f"latency : {ms:5.1f} ms",
        f"faces   : {n_faces}",
        f"gallery : {', '.join(gallery) if gallery else '(empty)'}",
        "s=save  q=quit",
    ]
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BLACK, 3)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1)


def save_and_reload(adapter, frame: "np.ndarray", gallery_dir: Path) -> None:
    name = input("\nEnter person name (no spaces, no extension): ").strip()
    if not name:
        print("Cancelled.")
        return
    out_path = gallery_dir / f"{name}.jpeg"
    cv2.imwrite(str(out_path), frame)
    print(f"Saved {out_path} — rebuilding gallery...")
    adapter._build_gallery()
    print(f"Gallery now: {list(adapter._gallery)}")


def run_image(adapter, path: str) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: cannot read {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")
    t0 = time.perf_counter()
    faces = adapter.run(frame, {})
    ms = (time.perf_counter() - t0) * 1000

    print(f"Latency : {ms:.1f} ms")
    print(f"Faces   : {len(faces)}")
    for f in faces:
        print(f"  {f}")

    annotated = draw_faces(frame, faces)
    cv2.imshow("SFace — still", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live(adapter, camera: int, gallery_dir: Path) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {camera}", file=sys.stderr)
        sys.exit(1)

    print(f"Camera {camera} opened. Warming up (15 frames)…")
    for _ in range(15):
        cap.read()
    print("Running. Press  q / ESC  to quit,  s  to save current frame.\n")

    last_ms = 0.0
    last_faces: list[dict] = []
    saved_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed — exiting.")
            break

        t0 = time.perf_counter()
        faces = adapter.run(frame, {})
        last_ms = (time.perf_counter() - t0) * 1000
        last_faces = faces
        saved_frame = frame.copy()

        display = draw_faces(frame, faces)
        overlay_hud(display, last_ms, len(last_faces), list(adapter._gallery))
        cv2.imshow("SFace live", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # q or ESC
            break
        if key == ord("s"):
            cv2.destroyWindow("SFace live")
            save_and_reload(adapter, saved_frame, gallery_dir)
            cv2.namedWindow("SFace live")

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--image", default=None, help="still-image path (skips live mode)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    face_cfg = cfg["adapters"]["face_recognition"]
    gallery_dir = Path(face_cfg.get("params", {}).get("gallery_dir", "data"))

    print("Building SFace adapter…")
    adapter = build_adapter(cfg)
    print(f"Gallery: {list(adapter._gallery)}\n")

    if args.image:
        run_image(adapter, args.image)
    else:
        run_live(adapter, args.camera, gallery_dir)


if __name__ == "__main__":
    main()
