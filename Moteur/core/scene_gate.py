"""Multi-signal scene-change detector for the live pipeline.

Combines five cheap-ish signals on every captured frame to decide whether
to fire a full inference run:

  1. dHash distance       — pixel-level change vs the last *committed* frame
  2. Face count delta     — number of faces detected by YuNet
  3. Identity-set change  — set of recognized names (SFace) — boolean
  4. Optical flow         — instantaneous motion magnitude vs the previous frame
  5. Embedding distance   — semantic change vs committed frame (MobileNet logits)

Each component is normalized to [0, 1] and combined via a configurable
weighted sum. The gate fires when the combined score crosses
``trigger_score``. The "committed" reference is updated only when the main
loop confirms it actually submitted a frame for inference, so noise drift
between non-firing frames cannot accumulate.

Design notes
------------
* The face detector / recognizer is shared with the orchestrator's
  ``face_recognition`` adapter. We pass that same SFaceAdapter instance in
  so we don't load YuNet/SFace twice.
* The MobileNet model is auto-downloaded from opencv_zoo on first load,
  same pattern Mohamed used for YuNet/SFace. Embedding is the
  L2-normalized 1000-dim ImageNet logits — for "did the scene change"
  similarity, the classification head is fine.
* Optical flow runs on a 160x120 downsample of the frame. Farneback at that
  size is ~3-5 ms on a modern CPU; full-res would be 10x more.
* The signals dataclass carries the heavy intermediate computations
  (dHash bits, embedding vector, face list) so commit() can promote them
  without recomputing — saves ~30-40 ms per inference trigger.
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# OpenCV Zoo MobileNetV2 (ImageNet, ~13 MB). Same hosting pattern as
# Mohamed's SFace/YuNet downloads — keeps all model URLs from one source.
MOBILENET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/image_classification_mobilenet/"
    "image_classification_mobilenetv2_2022apr.onnx"
)


@dataclass
class GateSignals:
    """Snapshot of all change signals for one frame, plus the trigger decision.

    Fields with leading ``_`` are heavy intermediates kept around so that
    ``SceneGate.commit()`` doesn't have to recompute them.
    """

    # Raw signal values
    dhash_distance: int = 0          # bits flipped (out of 256)
    face_count: int = 0
    face_count_delta: int = 0
    identity_set: frozenset = field(default_factory=frozenset)
    identity_changed: bool = False
    flow_magnitude: float = 0.0      # mean ||flow||, pixel units
    embedding_distance: float = 0.0  # 1 - cosine_similarity, in [0, 2]
    object_count: int = 0
    object_count_delta: int = 0
    object_labels: frozenset = field(default_factory=frozenset)
    objects_changed: bool = False

    # Combined decision
    score: float = 0.0
    triggered: bool = False

    # Cached intermediates — used by SceneGate.commit() so we don't redo
    # the dHash / embedding / SFace work that evaluate() already did.
    _dhash: Optional[np.ndarray] = None
    _embedding: Optional[np.ndarray] = None
    _faces: list = field(default_factory=list)
    _objects: list = field(default_factory=list)


class SceneGate:
    """Multi-signal scene-change detector. Thread-safe."""

    def __init__(
        self,
        sface_adapter: Optional[Any] = None,
        # Per-signal "max" values used for normalization to [0, 1].
        dhash_max_bits: int = 64,
        flow_max: float = 5.0,
        embedding_max: float = 0.4,
        # Weights — should sum to 1.0 for a well-calibrated trigger score.
        w_dhash: float = 0.30,
        w_face_count: float = 0.20,
        w_identity: float = 0.20,
        w_flow: float = 0.10,
        w_embedding: float = 0.20,
        w_objects: float = 0.15,
        # Final threshold the weighted sum must exceed to fire.
        trigger_score: float = 0.20,
        # MobileNet ONNX model path. Auto-downloads from MOBILENET_URL if
        # missing and ``download_if_missing`` is true.
        mobilenet_path: str | Path = "models/image_classification_mobilenetv2_2022apr.onnx",
        download_if_missing: bool = True,
    ) -> None:
        self._sface = sface_adapter
        self.dhash_max_bits = max(1, dhash_max_bits)
        self.flow_max = max(0.01, flow_max)
        self.embedding_max = max(0.001, embedding_max)
        self.w_dhash = w_dhash
        self.w_face_count = w_face_count
        self.w_identity = w_identity
        self.w_flow = w_flow
        self.w_embedding = w_embedding
        self.w_objects = w_objects
        self.trigger_score = trigger_score
        self._mobilenet_path = Path(mobilenet_path)
        self._download_if_missing = download_if_missing

        # Per-frame state (used for optical flow vs the *previous* frame).
        self._prev_flow_gray: Optional[np.ndarray] = None

        # "Committed" reference state — updated only when the main loop
        # confirms it submitted the gated frame for inference. This is what
        # current frames are compared against; drift between non-firing
        # frames cannot accumulate because we never overwrite this without
        # an explicit commit.
        self._committed_dhash: Optional[np.ndarray] = None
        self._committed_face_count: Optional[int] = None
        self._committed_identity_set: frozenset = frozenset()
        self._committed_embedding: Optional[np.ndarray] = None
        self._committed_object_count: Optional[int] = None
        self._committed_object_labels: frozenset = frozenset()

        self._net: Optional[cv2.dnn.Net] = None
        # Concurrent evaluate() + commit() can race on the committed_*
        # state — guard with a lock.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and load the MobileNet model.

        Failures are logged, not raised — the gate can still run with the
        embedding signal disabled (its weight is implicitly redistributed
        because evaluate() returns 0 for that signal).
        """
        try:
            if self._download_if_missing and not self._mobilenet_path.exists():
                self._mobilenet_path.parent.mkdir(parents=True, exist_ok=True)
                log.info("Downloading MobileNet → %s", self._mobilenet_path)
                urllib.request.urlretrieve(MOBILENET_URL, self._mobilenet_path)
            if self._mobilenet_path.exists():
                self._net = cv2.dnn.readNetFromONNX(str(self._mobilenet_path))
                log.info("MobileNet loaded from %s", self._mobilenet_path)
            else:
                log.warning(
                    "MobileNet not found at %s — embedding signal disabled",
                    self._mobilenet_path,
                )
        except Exception as e:
            log.warning("MobileNet load failed (%s) — embedding signal disabled", e)
            self._net = None

    # ------------------------------------------------------------------
    # Main entrypoints
    # ------------------------------------------------------------------

    def evaluate(
        self,
        frame: np.ndarray,
        faces: Optional[list] = None,
        objects: Optional[list] = None,
    ) -> GateSignals:
        """Compute all signals for a single frame; return the combined decision.

        ``faces`` lets the caller pass pre-computed SFace output to avoid a
        second YuNet/SFace pass per frame.  ``objects`` lets the caller pass
        pre-computed YOLO detections (same reason — the live pipeline runs the
        ObjDetWorker async and forwards its latest result here so YOLO never
        runs twice per frame).
        """
        sig = GateSignals()

        # 1) dHash distance vs committed.
        sig._dhash = self._dhash(frame)
        with self._lock:
            committed_dhash = self._committed_dhash
            committed_face_count = self._committed_face_count
            committed_identity = self._committed_identity_set
            committed_embedding = self._committed_embedding
            committed_object_count = self._committed_object_count
            committed_object_labels = self._committed_object_labels

        if committed_dhash is None:
            # First frame of the session — treat everything as "very different"
            # so the initial scene gets narrated.
            sig.dhash_distance = self.dhash_max_bits
        else:
            sig.dhash_distance = int(np.sum(sig._dhash != committed_dhash))

        # 2) + 3) Face count delta + identity-set change.
        # Prefer caller-supplied faces; fall back to running SFace ourselves
        # so the gate still works in code paths that don't pre-compute (e.g.
        # the standalone test in this module).
        if faces is not None:
            sig._faces = list(faces)
        elif self._sface is not None:
            try:
                sig._faces = self._sface.run(frame, {}) or []
            except Exception as e:
                log.warning("SFace in gate failed: %s", e)
                sig._faces = []

        if sig._faces or self._sface is not None or faces is not None:
            sig.face_count = len(sig._faces)
            sig.identity_set = frozenset(
                f.get("person_id", "") for f in sig._faces
                if isinstance(f, dict)
                and f.get("person_id")
                and f["person_id"] != "Unknown"
            )
            if committed_face_count is None:
                sig.face_count_delta = sig.face_count
                sig.identity_changed = bool(sig.identity_set)
            else:
                sig.face_count_delta = sig.face_count - committed_face_count
                sig.identity_changed = sig.identity_set != committed_identity

        # 3b) YOLO object label-set change vs committed frame.
        if objects is not None:
            sig._objects = list(objects)
        if sig._objects or objects is not None:
            sig.object_count = len(sig._objects)
            sig.object_labels = frozenset(
                o.get("label", "") for o in sig._objects
                if isinstance(o, dict) and o.get("label")
            )
            if committed_object_count is None:
                sig.object_count_delta = sig.object_count
                sig.objects_changed = bool(sig.object_labels)
            else:
                sig.object_count_delta = sig.object_count - committed_object_count
                sig.objects_changed = sig.object_labels != committed_object_labels

        # 4) Optical flow vs the previous frame.
        sig.flow_magnitude = self._optical_flow(frame)

        # 5) Embedding distance vs committed.
        sig._embedding = self._embedding(frame)
        if sig._embedding is not None and committed_embedding is not None:
            cos = float(np.dot(sig._embedding, committed_embedding))
            sig.embedding_distance = max(0.0, 1.0 - cos)
        elif sig._embedding is not None:
            # First frame with an embedding — treat as "very different".
            sig.embedding_distance = self.embedding_max

        # Combine. Each component is clipped to [0, 1] before weighting.
        score = (
            self.w_dhash      * min(1.0, sig.dhash_distance / self.dhash_max_bits)
            + self.w_face_count * (1.0 if abs(sig.face_count_delta) > 0 else 0.0)
            + self.w_identity   * (1.0 if sig.identity_changed else 0.0)
            + self.w_flow       * min(1.0, sig.flow_magnitude / self.flow_max)
            + self.w_embedding  * min(1.0, sig.embedding_distance / self.embedding_max)
            + self.w_objects    * (1.0 if sig.objects_changed else 0.0)
        )
        sig.score = float(score)
        sig.triggered = sig.score >= self.trigger_score
        return sig

    def commit(self, signals: GateSignals) -> None:
        """Promote ``signals``' frame to be the new committed reference.

        Called by the main loop when it confirms it actually submitted the
        frame for inference. Uses the cached intermediates from
        ``signals._dhash`` / ``_embedding`` / ``_faces`` so we don't redo
        the heavy compute.
        """
        with self._lock:
            if signals._dhash is not None:
                self._committed_dhash = signals._dhash
            self._committed_face_count = signals.face_count
            self._committed_identity_set = signals.identity_set
            if signals._embedding is not None:
                self._committed_embedding = signals._embedding
            self._committed_object_count = signals.object_count
            self._committed_object_labels = signals.object_labels

    # ------------------------------------------------------------------
    # Internal: per-signal helpers
    # ------------------------------------------------------------------

    def _dhash(self, frame: np.ndarray, size: int = 16) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
        return (small[:, 1:] > small[:, :-1]).astype(np.uint8).flatten()

    def _optical_flow(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        prev = self._prev_flow_gray
        self._prev_flow_gray = small
        if prev is None:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            prev, small, None,
            0.5, 3, 15, 3, 5, 1.2, 0,
        )
        # Mean L2 norm of the flow vectors — global "how much is moving".
        return float(np.linalg.norm(flow, axis=2).mean())

    def _embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self._net is None:
            return None
        try:
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 127.5,
                size=(224, 224),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
                crop=False,
            )
            self._net.setInput(blob)
            emb = self._net.forward().flatten().astype(np.float32)
            n = float(np.linalg.norm(emb))
            if n > 0:
                emb /= n
            return emb
        except Exception as e:
            log.warning("MobileNet forward failed: %s", e)
            return None
