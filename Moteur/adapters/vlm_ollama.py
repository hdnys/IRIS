"""Single-pass VLM adapter served by a local Ollama daemon.

Architecture:

* ``mode="static"`` — sends exactly one multimodal request per frame:
  image + scene prompt + system prompt. The model returns the user-facing
  narration string directly.

Pull the model tag once::

    ollama pull llava-phi3

YAML wiring::

    vlm:
      module: Moteur.adapters.vlm_ollama
      class:  OllamaVLM
      params:
        model:           llava-phi3
        host:  http://127.0.0.1:11434
        keep_alive: 30m
        timeout_s: 120
        max_image_side: 512
        num_ctx: 2048
        num_predict_static: 120

For extra latency / VRAM savings, set these on the **Ollama server**
(env vars, before ``ollama serve``)::

    OLLAMA_FLASH_ATTENTION=1
    OLLAMA_KV_CACHE_TYPE=q8_0

Any multimodal Ollama tag works for ``model``.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np

from Moteur.adapters.base import ModelAdapter

log = logging.getLogger(__name__)


# Single short prompt — llava-phi3 follows plain English better than
# schema-style instructions for this task.
SCENE_PROMPT = (
    "Describe the scene in 1-2 short, factual sentences for a blind user. "
    "Use the text labels in the image to identify people and objects, the overlay is there for you to understand but you should explain the extracted information to the user and not mention them. "
    "Treat the names as your inherent knowledge of who and what is present. "
    "Identify the number of people, name them, and make educated logical deductions to describe their actions and interactions with the environment. "
    "Note key objects and precise spatial layout (left, right, center, foreground). "
    "Do not give advice. Do not greet. Output only the exact scene description. You are describing the scene in natural language for a user that CANNOT  know that this is a frame from a camera with preprocessing "
)


# Per-profile and verbosity guidance for the single-pass narration call.
PROFILE_GUIDANCE = {
    "total_blindness": (
        "User cannot see. Be spatially explicit (left/right/ahead/distance). "
        "Lead with hazards and obstacles. Avoid colors and visual aesthetics."
    ),
    "low_vision": (
        "User has some residual vision. Mention high-contrast cues and key "
        "shapes. Keep language concise."
    ),
    "tunnel_vision": (
        "User sees only the center. Emphasize peripheral context the user "
        "would miss — what is to the sides, approaching from the edges."
    ),
    "color_blindness": (
        "User cannot rely on color. Describe by shape, position, and text. "
        "Avoid color-only references."
    ),
    "peripheral_loss": (
        "User has reduced peripheral vision. Describe central focus and "
        "explicitly mention things outside their likely field of view."
    ),
}

VERBOSITY_GUIDANCE = {
    "minimal": "ONE clause, max 8 words. Only the single most important fact.",
    "standard": "ONE sentence, max 18 words. Direct and factual.",
    "detailed": "Up to two sentences, 35 words total max. No filler.",
}

# Cap on the number of YOLO detections we forward to the VLM.
_MAX_OBJECTS_FOR_VLM = 8


# Cap on raw scene_description length — defends downstream consumers from
# a runaway VLM that ignores the "1-2 sentences" instruction.
SCENE_DESC_MAX_CHARS = 600


def _build_personas(objects: list, faces: list) -> tuple[list[dict], list[dict]]:
    """Match YOLO person boxes with SFace face boxes into unified Persona dicts.

    Matching criterion: face bbox center falls inside the YOLO person bbox.
    O(P × F) — negligible for typical P, F < 20.

    Returns (personas, non_person_objects).

    Each persona dict:
        person_id         str            "Elie" | "Unknown"
        face_confidence   float | None   SFace cosine score; None if unmatched
        position          str | None     YOLO grid label ("center-middle", …)
        size              str | None     "small" | "medium" | "large"
        area_fraction     float | None   person-box area / frame area
        bounding_box      dict | None    YOLO person box {x,y,width,height}
        face_bounding_box dict | None    SFace face box; None if unmatched

    Unmatched persons → persona with person_id "Unknown", no face fields.
    Unmatched faces   → persona built from face box only (YOLO missed the body).
    Headcount authority remains YOLO (bounding_box is not None) — unmatched
    faces are included for identity but must not inflate the person count.
    """
    persons     = [o for o in objects if isinstance(o, dict) and o.get("label") == "person"]
    non_persons = [o for o in objects if isinstance(o, dict) and o.get("label") != "person"]

    matched = [False] * len(faces)
    personas: list[dict] = []

    for person in persons:
        pbb = person.get("bounding_box", {})
        px, py = float(pbb.get("x", 0)), float(pbb.get("y", 0))
        pw, ph = float(pbb.get("width", 0)), float(pbb.get("height", 0))

        best_idx, best_conf = -1, -1.0
        for i, face in enumerate(faces):
            if matched[i]:
                continue
            fbb = face.get("bounding_box", {})
            cx = float(fbb.get("x", 0)) + float(fbb.get("width", 0)) / 2.0
            cy = float(fbb.get("y", 0)) + float(fbb.get("height", 0)) / 2.0
            if px <= cx <= px + pw and py <= cy <= py + ph:
                conf = float(face.get("confidence", 0.0))
                if conf > best_conf:
                    best_conf, best_idx = conf, i

        face = faces[best_idx] if best_idx >= 0 else None
        if best_idx >= 0:
            matched[best_idx] = True

        personas.append({
            "person_id":         face["person_id"] if face else "Unknown",
            "face_confidence":   round(best_conf, 3) if best_idx >= 0 else None,
            "position":          person.get("position"),
            "size":              person.get("size"),
            "area_fraction":     person.get("area_fraction"),
            "bounding_box":      pbb,
            "face_bounding_box": face.get("bounding_box") if face else None,
        })

    # Faces whose center wasn't inside any person box (YOLO body miss).
    for i, face in enumerate(faces):
        if not matched[i]:
            personas.append({
                "person_id":         face["person_id"],
                "face_confidence":   round(float(face.get("confidence", 0.0)), 3),
                "position":          None,
                "size":              None,
                "area_fraction":     None,
                "bounding_box":      None,
                "face_bounding_box": face.get("bounding_box"),
            })

    return personas, non_persons


def _build_grounding(personas: list[dict], non_person_objects: list[dict]) -> str:
    """Compact sensor-grounding block appended to SCENE_PROMPT for llava-phi3.

    Tells the vision model exactly what YOLO and SFace already confirmed so it
    can anchor its description to ground truth rather than hallucinating.
    """
    parts: list[str] = []

    if personas:
        named   = [p["person_id"] for p in personas if p["person_id"] != "Unknown"]
        unknown = sum(1 for p in personas if p["person_id"] == "Unknown")
        people: list[str] = list(named)
        if unknown:
            people.append(f"{unknown} unrecognized {'person' if unknown == 1 else 'people'}")
        if people:
            parts.append("People detected: " + ", ".join(people))

    if non_person_objects:
        obj_strs = [
            f"{o['label']} ({o.get('size','?')}, {o.get('position','?')})"
            for o in non_person_objects[:_MAX_OBJECTS_FOR_VLM]
        ]
        parts.append("Objects detected: " + ", ".join(obj_strs))

    if not parts:
        return ""

    return (
        "Sensor data for this frame:\n"
        + "\n".join(f"  {p}" for p in parts) + "\n"
        "Only describe people and objects that appear in the sensor data above. "
        "Do not mention anything not listed there."
    )


class OllamaVLM(ModelAdapter):
    """Single-pass multimodal narrator via Ollama."""

    version = "ollama-single-pass-0.4"
    writes = ["dynamic.scene_description", "dynamic.vlm_description"]

    def __init__(
        self,
        role: str,
        model: str,
        host: str,
        keep_alive: str,
        timeout_s: float,
        max_image_side: int,
        temperature: float,
        num_ctx: int,
        num_predict_static: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(role, **kwargs)
        # Single multimodal model for one-pass narration.
        self.model = model
        # Strip trailing slash so urljoin-style concatenation below stays clean.
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        # llava-phi3's vision encoder works efficiently at ≤512 px per side;
        # larger inputs cost more base64 without improving description quality.
        self.max_image_side = max_image_side
        self.temperature = temperature
        # 2k context fits one image (~256 tokens) + the combined instruction
        # plenty of slack. The DataPool feeds a fresh prompt every call so KV
        # reuse across calls is not relied on.
        self.num_ctx = num_ctx
        # 120 is well above a 1-2 sentence factual description; prevents the
        # rare run where the model rambles into commentary.
        self.num_predict_static = num_predict_static

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Verify Ollama is reachable and the configured model tag exists.

        Fails loudly at startup so a bad config / missing pull surfaces before
        the first frame, matching the eager-load contract in registry.py.
        """
        try:
            tags = self._http_get("/api/tags")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`. "
                f"Underlying error: {e}"
            ) from e

        names = {m.get("name", "") for m in tags.get("models", [])}
        # Ollama appends ":latest" to bare names in the listing — accept either form.
        if self.model not in names and f"{self.model}:latest" not in names:
            raise RuntimeError(
                f"Model '{self.model}' not pulled. Run: ollama pull {self.model}"
            )

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, frame: Any, pool_snapshot: dict, **kwargs: Any) -> Any:
        """Single mode: one multimodal request returns the narration string."""
        return self._run_static(
            frame,
            pool_snapshot,
            objects=kwargs.get("objects"),
            faces=kwargs.get("faces"),
        )

    # ------------------------------------------------------------------
    # Scene-description pass (vision)
    # ------------------------------------------------------------------

    def _run_static(self, frame: np.ndarray, pool_snapshot: dict,
                    objects: Optional[list] = None,
                    faces: Optional[list] = None) -> str:
        image_b64 = self._encode_frame(frame)

        personas, non_person_objects = _build_personas(objects or [], faces or [])
        grounding = _build_grounding(personas, non_person_objects)
        prompt = SCENE_PROMPT + ("\n\n" + grounding if grounding else "")
        system = self._build_system_prompt(pool_snapshot)

        # ── model (self.model, e.g. llava-phi3) ─────────────────────────
        # Receives: SCENE_PROMPT + system prompt + optional sensor grounding
        #           + the JPEG frame as base64.
        # Returns:  a short narration string.
        body = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "images": [image_b64],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict_static,
            },
        }

        try:
            resp = self._http_post("/api/generate", body)
        except Exception as e:
            log.warning("Scene-description call failed: %s", e)
            return ""

        text = (resp.get("response") or "").strip()
        # Defensive trim — a runaway model would otherwise blow up the
        # combined prompt.
        if len(text) > SCENE_DESC_MAX_CHARS:
            text = text[:SCENE_DESC_MAX_CHARS].rstrip() + "…"
        return text

    def _build_system_prompt(self, snap: dict) -> str:
        profile = snap.get("user_profile", {}) if isinstance(snap, dict) else {}
        vision = profile.get("vision_profile", "low_vision")
        verbosity = profile.get("verbosity", "standard")
        language = profile.get("preferred_language", "en-US")
        profile_guidance = PROFILE_GUIDANCE.get(vision, PROFILE_GUIDANCE["low_vision"])
        length_guidance = VERBOSITY_GUIDANCE.get(verbosity, VERBOSITY_GUIDANCE["standard"])

        return (
            "You are IRIS, narrating a live camera frame for a blind or low-vision user.\n"
            f"Language: {language}. Second person ('you').\n"
            f"Length: {length_guidance}\n"
            f"{profile_guidance}\n"
            "Be factual and spatially precise. Do not greet. Do not give advice. "
            "Output only the final narration."
        )

    # ------------------------------------------------------------------
    # Frame encoding
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: np.ndarray) -> str:
        """BGR ndarray → base64-encoded JPEG suitable for Ollama's ``images`` field.

        JPEG (not PNG) because the payload travels as base64 in JSON and a
        full-resolution PNG would inflate the request well past Ollama's body
        limit. JPEG quality 85 is visually transparent for VLM inputs.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray (BGR)")

        h, w = frame.shape[:2]
        if self.max_image_side and max(h, w) > self.max_image_side:
            scale = self.max_image_side / float(max(h, w))
            frame = cv2.resize(
                frame,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("cv2.imencode failed on frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ------------------------------------------------------------------
    # HTTP helpers (stdlib only to avoid adding a requests dep)
    # ------------------------------------------------------------------

    def _http_post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.host + path,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {e.code} on {path}: {body}") from e

    def _http_get(self, path: str) -> dict:
        req = urllib_request.Request(self.host + path, method="GET")
        with urllib_request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
