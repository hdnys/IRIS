"""SFace face-recognition adapter using OpenCV's YuNet + SFace pipeline.

Detection:    YuNet  (face_detection_yunet_2023mar.onnx)
Recognition:  SFace  (face_recognition_sface_2021dec.onnx)

Both models ship with OpenCV; no extra runtime is required.

Reference gallery
-----------------
Place one clear frontal photo per person in ``<gallery_dir>/``.
The filename stem (without extension) becomes the person's display name.
Supported extensions: .jpg / .jpeg / .png / .bmp

    gallery_dir: models/../data      # default

Tuning notes (from Mohamed's notebook)
---------------------------------------
* COS_THR = 0.3 — cosine similarity threshold.  Faces below this score are
  labeled "Unknown".  Lower → stricter (fewer false positives).
* Detection confidence 0.6 and NMS 0.3 work well for single-camera indoor use.
* 15-frame warm-up is handled by the capture layer; the adapter is stateless.

YAML wiring::

    face_recognition:
      module: Moteur.adapters.face_sface
      class:  SFaceAdapter
      params:
        yunet_path:   models/face_detection_yunet_2023mar.onnx
        sface_path:   models/face_recognition_sface_2021dec.onnx
        gallery_dir:  data
        cos_threshold: 0.3
        det_confidence: 0.6
        det_nms: 0.3
        top_k: 5000

Output written to ``dynamic.faces`` — a list of dicts, one per detected face::

    {
        "person_id":   "Elie" | "Unknown",
        "confidence":  0.82,          # cosine similarity score
        "bounding_box": {"x": 50, "y": 40, "width": 80, "height": 100}
    }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from Moteur.adapters.base import ModelAdapter

log = logging.getLogger(__name__)

_GALLERY_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class SFaceAdapter(ModelAdapter):
    """YuNet detection + SFace recognition, fully OpenCV-native."""

    version = "sface-yunet-0.1"
    writes = ["dynamic.faces"]

    def __init__(
        self,
        role: str,
        yunet_path: str = "models/face_detection_yunet_2023mar.onnx",
        sface_path: str = "models/face_recognition_sface_2021dec.onnx",
        gallery_dir: str = "data",
        cos_threshold: float = 0.3,
        det_confidence: float = 0.6,
        det_nms: float = 0.3,
        top_k: int = 5000,
        **kwargs: Any,
    ) -> None:
        super().__init__(role, **kwargs)
        self.yunet_path = str(yunet_path)
        self.sface_path = str(sface_path)
        self.gallery_dir = Path(gallery_dir)
        self.cos_threshold = cos_threshold
        self.det_confidence = det_confidence
        self.det_nms = det_nms
        self.top_k = top_k

        # Populated in load()
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        # name → list of embedding ndarrays. Multiple references per person
        # (different angles, lighting) markedly improves recognition under
        # head turns and partial occlusion. _match() takes the max similarity
        # across all of a person's embeddings.
        self._gallery: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load both ONNX models and build the reference gallery."""
        if not Path(self.yunet_path).exists():
            raise RuntimeError(
                f"YuNet model not found at '{self.yunet_path}'. "
                "Copy it from Mohamed's branch: models/face_detection_yunet_2023mar.onnx"
            )
        if not Path(self.sface_path).exists():
            raise RuntimeError(
                f"SFace model not found at '{self.sface_path}'. "
                "Copy it from Mohamed's branch: models/face_recognition_sface_2021dec.onnx"
            )

        # Input size (320, 320) is the model's native resolution; YuNet
        # re-scales internally, so we override per frame in run().
        self._detector = cv2.FaceDetectorYN.create(
            self.yunet_path, "", (320, 320),
            self.det_confidence, self.det_nms, self.top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")

        self._build_gallery()
        log.info(
            "SFaceAdapter loaded — %d person(s) in gallery: %s",
            len(self._gallery), list(self._gallery),
        )

    def unload(self) -> None:
        self._detector = None
        self._recognizer = None
        self._gallery.clear()

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, frame: Any, pool_snapshot: dict, **_: Any) -> list[dict]:
        """Detect all faces and return recognition results.

        Returns a list of face dicts conforming to the schema expected by
        ``dynamic.faces``. Returns an empty list if no faces are detected or
        the adapter is not loaded.
        """
        if self._detector is None or self._recognizer is None:
            return []
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)

        if faces is None:
            return []

        results: list[dict] = []
        for face in faces:
            feat = self._recognizer.feature(
                self._recognizer.alignCrop(frame, face)
            )
            name, score = self._match(feat)

            x, y, fw, fh = (int(v) for v in face[:4])
            results.append({
                "person_id": name,
                "confidence": round(float(score), 3),
                "bounding_box": {"x": x, "y": y, "width": fw, "height": fh},
            })

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match(self, feat: np.ndarray) -> tuple[str, float]:
        """Compare feat against the gallery and return (name, score).

        For each person we take the *max* cosine similarity across all of
        their reference embeddings — recognition succeeds if ANY reference
        view matches well. This is what makes multi-image galleries useful:
        a head-turn that wouldn't match the frontal reference can still
        match the left- or right-profile reference.
        """
        best_name, best_score = "Unknown", -1.0
        for name, refs in self._gallery.items():
            for ref in refs:
                score = float(self._recognizer.match(
                    ref, feat, cv2.FaceRecognizerSF_FR_COSINE
                ))
                if score > best_score:
                    best_name, best_score = name, score
        if best_score < self.cos_threshold:
            best_name = "Unknown"
        return best_name, best_score

    def _build_gallery(self) -> None:
        """Embed every reference image in gallery_dir into self._gallery.

        Two layouts are supported simultaneously:

        * Single-image (legacy)::

              data/Mohamad.jpeg          → 1 reference for "Mohamad"

        * Multi-image (per-person folder, used by FaceLearner)::

              data/Mohamad/1.jpg
              data/Mohamad/2.jpg         → N references for "Mohamad"
              data/Mohamad/3.jpg

        If a folder ``data/<name>/`` exists, the legacy ``data/<name>.<ext>``
        file is ignored — the folder wins so a freshly-learned multi-shot
        gallery isn't diluted by the older single shot.
        """
        new_gallery: dict[str, list[np.ndarray]] = {}

        if not self.gallery_dir.is_dir():
            log.warning("Gallery dir '%s' not found — no known faces loaded.", self.gallery_dir)
            self._gallery = new_gallery
            return

        # Pre-collect folder names so we can shadow legacy single files of the
        # same stem.
        folder_names = {
            entry.name for entry in self.gallery_dir.iterdir() if entry.is_dir()
        }

        for entry in sorted(self.gallery_dir.iterdir()):
            if entry.is_dir():
                # Multi-image: average / store every reference image as its
                # own embedding.
                name = entry.name
                embs: list[np.ndarray] = []
                for img_path in sorted(entry.iterdir()):
                    if img_path.suffix.lower() not in _GALLERY_EXTS:
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        log.warning("Cannot read gallery image: %s", img_path)
                        continue
                    emb = self._embed_best_face(img)
                    if emb is None:
                        log.warning("No face in %s — skipping.", img_path.name)
                        continue
                    embs.append(emb)
                if embs:
                    new_gallery[name] = embs
                continue

            # Legacy single image at the gallery root.
            if entry.suffix.lower() not in _GALLERY_EXTS:
                continue
            if entry.stem in folder_names:
                # A multi-image folder for this person already exists;
                # don't pollute their gallery with the older single shot.
                continue
            img = cv2.imread(str(entry))
            if img is None:
                log.warning("Cannot read gallery image: %s", entry)
                continue
            emb = self._embed_best_face(img)
            if emb is None:
                log.warning("No face found in gallery image: %s", entry.name)
                continue
            new_gallery[entry.stem] = [emb]

        self._gallery = new_gallery

    def reload_gallery(self) -> None:
        """Re-scan ``gallery_dir`` and rebuild the gallery in-place.

        Public so the live pipeline's face-learner can refresh after
        capturing a new person without restarting the whole adapter.
        """
        self._build_gallery()
        log.info(
            "SFaceAdapter gallery reloaded — %d people: %s",
            len(self._gallery), list(self._gallery),
        )

    def _embed_best_face(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect the most-confident face in img and return its embedding."""
        h, w = img.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(img)
        if faces is None:
            return None
        best = faces[np.argmax(faces[:, -1])]
        return self._recognizer.feature(self._recognizer.alignCrop(img, best))
