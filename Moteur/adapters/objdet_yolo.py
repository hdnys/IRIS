"""YOLO object-detection adapter (Ultralytics).

Wraps an Ultralytics YOLO model (default: yolo12n) and exposes it through the
standard :class:`ModelAdapter` interface so the orchestrator can dispatch it
the same way as any other dynamic adapter.

YAML wiring::

    object_detection:
      module: Moteur.adapters.objdet_yolo
      class:  YoloObjectDetector
      params:
        weights_path:    models/yolo12n.pt
        conf_threshold:  0.35
        iou_threshold:   0.45
        max_image_side:  640        # resize before inference for speed
        keep_classes:    null       # e.g. [0, 2, 56] for person/car/chair
        device:          null       # null = auto (cpu / mps / cuda)
        verbose:         false

Output schema (one entry per detection)::

    {
        "label":           "person",
        "class_id":        0,
        "confidence":      0.87,
        "bounding_box":    {"x": 120, "y": 80, "width": 90, "height": 200},
        "bounding_box_norm": {"x": 0.18, "y": 0.11, "width": 0.14, "height": 0.28},
        "position":        "center-middle",     # 3x3 grid label
        "size":            "medium",            # small | medium | large
        "area_fraction":   0.039,               # box area / frame area
        "distance_m":      null,                # filled in by depth adapter
    }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from Moteur.adapters.base import ModelAdapter

log = logging.getLogger(__name__)


def _position_label(cx: float, cy: float, w: int, h: int) -> str:
    x_zone = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    y_zone = "top"  if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    if x_zone == "center" and y_zone == "middle":
        return "center"
    return f"{y_zone}-{x_zone}"


def _size_label(area_fraction: float) -> str:
    if area_fraction < 0.02:
        return "small"
    if area_fraction < 0.15:
        return "medium"
    return "large"


class YoloObjectDetector(ModelAdapter):
    """Ultralytics YOLO wrapper. Returns rich per-detection metadata."""

    version = "yolo-ultralytics-0.1"
    writes = ["dynamic.object_detection"]

    def __init__(
        self,
        role: str,
        weights_path: str,
        conf_threshold: float,
        iou_threshold: float,
        max_image_side: Optional[int],
        keep_classes: Optional[list[int]],
        exclude_classes: Optional[list[int]],
        device: Optional[str],
        verbose: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(role, **kwargs)
        self.weights_path = str(weights_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_image_side = int(max_image_side) if max_image_side else None
        self.keep_classes = list(keep_classes) if keep_classes else None
        # Post-filter: e.g. [0] to drop persons because SFace owns that class.
        self._exclude = set(exclude_classes or [])
        self.device = device
        self.verbose = bool(verbose)

        self._model = None
        self._names: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e

        weights = Path(self.weights_path)
        # Ultralytics will auto-download standard weights (e.g. yolo12n.pt) into
        # its own cache if the local file is missing — we let it.
        target = str(weights) if weights.exists() else weights.name

        self._model = YOLO(target)
        # `.names` is a dict[int, str] mapping class-id → label.
        self._names = dict(self._model.names)

        # Warm the model with a dummy forward pass so the first real frame
        # doesn't pay the lazy-init cost (esp. on CUDA / MPS).
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self._predict(dummy)
        except Exception as e:
            log.warning("YOLO warm-up failed (non-fatal): %s", e)

        log.info(
            "YoloObjectDetector loaded — %d classes, weights=%s",
            len(self._names), target,
        )

    def unload(self) -> None:
        self._model = None
        self._names = {}

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, frame: Any, pool_snapshot: dict, **_: Any) -> list[dict]:
        if self._model is None or frame is None or not isinstance(frame, np.ndarray):
            return []

        h, w = frame.shape[:2]
        results = self._predict(frame)
        if not results:
            return []

        # Ultralytics returns a list (one Result per input image).
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Pull tensors once and convert to numpy for vectorized iteration.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        frame_area = float(w * h) if w and h else 1.0
        out: list[dict] = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            if int(cls) in self._exclude:
                continue
            bw = max(0.0, float(x2 - x1))
            bh = max(0.0, float(y2 - y1))
            cx = float(x1) + bw / 2.0
            cy = float(y1) + bh / 2.0
            area_frac = (bw * bh) / frame_area

            out.append({
                "label":      self._names.get(int(cls), str(int(cls))),
                "class_id":   int(cls),
                "confidence": round(float(conf), 3),
                "bounding_box": {
                    "x":      int(x1),
                    "y":      int(y1),
                    "width":  int(round(bw)),
                    "height": int(round(bh)),
                },
                "bounding_box_norm": {
                    "x":      round(float(x1) / w, 4) if w else 0.0,
                    "y":      round(float(y1) / h, 4) if h else 0.0,
                    "width":  round(bw / w, 4) if w else 0.0,
                    "height": round(bh / h, 4) if h else 0.0,
                },
                "position":      _position_label(cx, cy, w, h),
                "size":          _size_label(area_frac),
                "area_fraction": round(area_frac, 4),
                "distance_m":    None,
            })

        # Sort most-prominent first: confidence then size. The narrator only
        # has so much attention budget — leading with the high-signal hits
        # keeps the description on-point.
        out.sort(key=lambda d: (d["confidence"], d["area_fraction"]), reverse=True)
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict(self, frame: np.ndarray):
        kwargs: dict[str, Any] = {
            "conf":    self.conf_threshold,
            "iou":     self.iou_threshold,
            "verbose": self.verbose,
        }
        if self.max_image_side:
            kwargs["imgsz"] = self.max_image_side
        if self.keep_classes:
            kwargs["classes"] = self.keep_classes
        if self.device:
            kwargs["device"] = self.device
        return self._model(frame, **kwargs)
