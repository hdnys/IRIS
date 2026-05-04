"""Two-model VLM adapter served by a local Ollama daemon.

Architecture (post-refactor — simplified for latency):

* ``mode="static"``  — sends the frame to a fast multimodal model and returns
  a free-form scene description string (1–2 sentences). No JSON, no run_*
  flags, no orchestration metadata. The orchestrator routes the string into
  ``dynamic.scene_description`` so it lands alongside every other adapter's
  output. Default model: ``moondream`` (~1.9 B params, built for fast VQA).

* ``mode="describe"`` — text-only narration pass. Reads the assembled pool
  (scene description + detector outputs) and emits the user-facing string.
  Does not see the image. Default model: ``gemma3:1b`` (~0.7 GB).

Why this shape: in the previous design the static pass produced a 14-field
JSON of orchestration flags. Generating those tokens cost ~2 s of latency
per frame, and the flags themselves only gated stub adapters. We dropped
the flags, dropped JSON mode, and now the static pass is a single short
description (~50 generated tokens, ~250–500 ms warm).

Pull both tags once::

    ollama pull moondream
    ollama pull gemma3:1b

YAML wiring::

    vlm:
      module: Moteur.adapters.vlm_gemma3_ollama
      class:  Gemma3OllamaVLM
      params:
        model:           moondream
        model_describe:  gemma3:1b
        host:  http://127.0.0.1:11434
        keep_alive: 30m
        timeout_s: 120
        max_image_side: 512
        num_ctx: 2048
        num_predict_static: 120       # ~50-token descriptions; 120 is headroom
        num_predict_describe: null    # null = derive from user_profile.verbosity

For extra latency / VRAM savings, set these on the **Ollama server**
(env vars, before ``ollama serve``)::

    OLLAMA_FLASH_ATTENTION=1
    OLLAMA_KV_CACHE_TYPE=q8_0

The class name is historical (the adapter started as Gemma-only). It's
generic Ollama now — any multimodal tag for ``model`` and any text tag
for ``model_describe`` will work.
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


# Single short prompt — Moondream and Gemma both follow plain English better
# than schema-style instructions for this kind of task.
SCENE_PROMPT = (
    "Describe what you see in 1-2 short factual sentences. "
    "Mention people (count and what they appear to be doing), key objects, "
    "spatial layout (left/right/center/foreground), and "
    "(steps, traffic, obstacles). Do not give advice. Do not greet. "
    "Output only the description."
)


# Per-profile and verbosity guidance for the narrator pass.
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
    "minimal": "One short sentence. Only the most important fact.",
    "standard": "Two to three sentences. Focus on what matters now.",
    "detailed": "Up to five sentences. Include relevant context.",
}

VERBOSITY_NUM_PREDICT = {
    "minimal": 60,
    "standard": 160,
    "detailed": 350,
}

# Cap on raw scene_description length — defends the narrator's prompt from
# a runaway VLM that ignores the "1-2 sentences" instruction.
SCENE_DESC_MAX_CHARS = 600


class Gemma3OllamaVLM(ModelAdapter):
    """Multimodal scene-description model + small text narrator, both via Ollama."""

    version = "ollama-scene-narrator-0.3"
    writes = ["dynamic.scene_description", "dynamic.vlm_description"]

    def __init__(
        self,
        role: str,
        model: str = "moondream",
        model_describe: Optional[str] = "gemma3:1b",
        host: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        timeout_s: float = 120.0,
        max_image_side: int = 512,
        temperature: float = 0.2,
        num_ctx: int = 2048,
        num_predict_static: int = 120,
        num_predict_describe: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(role, **kwargs)
        # Multimodal model for the scene-description pass.
        self.model = model
        # Text-only model for narration. None means "reuse the static model"
        # (useful if you only want to manage one tag).
        self.model_describe = model_describe or model
        # Strip trailing slash so urljoin-style concatenation below stays clean.
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        # Moondream's vision tower works at ~378×378 internally; sending a
        # 512-px-side image is plenty and cheaper to base64 than larger.
        self.max_image_side = max_image_side
        self.temperature = temperature
        # 2k context fits one image (~256 tokens) + the narrator prompt with
        # plenty of slack. The DataPool feeds a fresh prompt every call so KV
        # reuse across calls is not relied on.
        self.num_ctx = num_ctx
        # 120 is well above a 1-2 sentence factual description; prevents the
        # rare run where the model rambles into commentary.
        self.num_predict_static = num_predict_static
        # Describe ceiling: None means "derive from the user's verbosity
        # setting at call time"; an integer forces an explicit cap.
        self.num_predict_describe = num_predict_describe

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Verify Ollama is reachable and both model tags exist.

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
        # De-dupe in case static and describe share a tag.
        for tag in {self.model, self.model_describe}:
            # Ollama appends ":latest" to bare names in the listing — accept either form.
            if tag not in names and f"{tag}:latest" not in names:
                raise RuntimeError(
                    f"Model '{tag}' not pulled. Run: ollama pull {tag}"
                )

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, frame: Any, pool_snapshot: dict, **kwargs: Any) -> Any:
        """Two modes:

        * ``mode="static"``   → str (the scene description). Orchestrator
          writes the return value into ``dynamic.scene_description``.
        * ``mode="describe"`` → str | None (the user-facing narration).
        """
        mode = kwargs.get("mode", "static")
        if mode == "static":
            # Static pass does not need the snapshot — see _run_static comment.
            return self._run_static(frame)
        return self._run_describe(pool_snapshot)

    # ------------------------------------------------------------------
    # Scene-description pass (vision)
    # ------------------------------------------------------------------

    def _run_static(self, frame: np.ndarray) -> str:
        # Pool snapshot is intentionally not consumed: the simplified prompt
        # does not condition on prior context. Scene-change detection is the
        # capture layer's job (phash similarity), not the VLM's.
        image_b64 = self._encode_frame(frame)

        body = {
            "model": self.model,
            "prompt": SCENE_PROMPT,
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
        # Defensive trim — moondream is well-behaved but a runaway model
        # would otherwise blow up the narrator prompt.
        if len(text) > SCENE_DESC_MAX_CHARS:
            text = text[:SCENE_DESC_MAX_CHARS].rstrip() + "…"
        return text

    # ------------------------------------------------------------------
    # Narrator pass (text-only)
    # ------------------------------------------------------------------

    def _run_describe(self, snap: dict) -> Optional[str]:
        profile = snap.get("user_profile", {})
        vision = profile.get("vision_profile", "low_vision")
        verbosity = profile.get("verbosity", "standard")
        language = profile.get("preferred_language", "en-US")

        system = (
            f"You are IRIS, speaking aloud to a user.\n"
            f"User profile: {PROFILE_GUIDANCE.get(vision, PROFILE_GUIDANCE['low_vision'])}\n"
            f"Verbosity: {VERBOSITY_GUIDANCE.get(verbosity, VERBOSITY_GUIDANCE['standard'])}\n"
            f"Reply in {language}. Speak in second person ('you see…', 'in front of you…'). "
            f"If the scene mentions a hazard, lead with it. Do not list raw fields; "
            f"speak naturally. Output the description text only — no preamble, no JSON."
        )

        # Lead with the multimodal model's scene_description (the only thing
        # in the snapshot that resembles "looking at the frame" from the
        # narrator's POV); detector outputs are supporting evidence.
        dyn = dict(snap.get("dynamic", {}))
        scene_desc = (dyn.pop("scene_description", "") or "").strip()

        sections: list[str] = []
        if scene_desc:
            sections.append(f"What the camera sees:\n{scene_desc}")
        else:
            sections.append("What the camera sees: (no scene description available)")

        if dyn:
            sections.append(
                "Detector outputs (objects, faces, text, ...):\n"
                f"{json.dumps(dyn, ensure_ascii=False, default=str)}"
            )

        sections.append("Write the description now.")
        prompt = "\n\n".join(sections)

        # Cap describe length: explicit override > verbosity preset.
        max_tokens = (
            self.num_predict_describe
            if self.num_predict_describe is not None
            else VERBOSITY_NUM_PREDICT.get(verbosity, VERBOSITY_NUM_PREDICT["standard"])
        )

        body = {
            # Describe runs on the smaller text-only model — no vision tower
            # to load, far less compute per generated token.
            "model": self.model_describe,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = self._http_post("/api/generate", body)
        except Exception as e:
            log.warning("Narrator call failed: %s", e)
            return None

        text = (resp.get("response") or "").strip()
        return text or None

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
