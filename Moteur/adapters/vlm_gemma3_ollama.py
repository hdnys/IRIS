"""Two-model Gemma adapter served by a local Ollama daemon.

Each VLM pass uses the model that fits its actual job:

* ``mode="static"``   — needs vision and structured output. Default model:
  ``gemma3:4b`` (multimodal, ~3.4 GB Q4_K_M). Ollama's JSON-mode forces
  parseable output.
* ``mode="describe"`` — text-in / text-out: the DataPool already carries every
  dynamic adapter's output, so the describe pass does NOT see the image.
  Default model: ``gemma3:1b`` (text-only, ~0.7 GB) — much faster per token
  and no vision tower to load.

Why split? On a single 12B VLM, the static pass is ~3.5 s and describe ~1.1 s
on a mid-range GPU. With the 4B/1B split, expect ~0.7–1.1 s static and
~0.15–0.3 s describe at <5 GB combined VRAM, with quality acceptable for
this task (static is mostly classification; describe is short narration).

Pull both tags once::

    ollama pull gemma3:4b
    ollama pull gemma3:1b

YAML wiring::

    vlm:
      module: Moteur.adapters.vlm_gemma3_ollama
      class:  Gemma3OllamaVLM
      params:
        model:           gemma3:4b   # static (multimodal)
        model_describe:  gemma3:1b   # describe (text-only); null = reuse model
        host:  http://127.0.0.1:11434
        keep_alive: 30m
        timeout_s: 120
        max_image_side: 896
        num_ctx: 2048                # small — pool feeds full prompt each call
        num_predict_static: 200
        num_predict_describe: null   # null = derive from user_profile.verbosity

If describe quality with gemma3:1b feels too shallow, swap in a stronger
small text LLM (still well under 400 ms): ``qwen2.5:1.5b``, ``llama3.2:3b``,
``phi3.5``.

For extra latency / VRAM savings, also set these on the **Ollama server**
(env vars, before ``ollama serve``)::

    OLLAMA_FLASH_ATTENTION=1   # faster prefill, smaller KV
    OLLAMA_KV_CACHE_TYPE=q8_0  # near-lossless KV-cache quantization
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


# Schema-aligned defaults. Used when the model returns malformed JSON or a
# field is missing — keeps the pipeline moving instead of dropping the frame.
STATIC_DEFAULTS: dict[str, Any] = {
    "is_new_scene": False,
    "scene_type": "unknown",
    "lighting_condition": "unknown",
    "motion_level": "unknown",
    "run_object_detection": True,
    "run_face_recognition": False,
    "run_emotion_detection": False,
    "run_ocr": False,
    "run_depth_estimation": False,
    "people_present": False,
    "people_count": 0,
    "text_visible": False,
    "hazard_detected": False,
    "priority_context": "general",
    # Free-form scene description written by the multimodal model. Routed into
    # dynamic.scene_description by the orchestrator so the (text-only)
    # describe pass has actual visual content to work from.
    "scene_description": "",
}

# Cap on scene_description length to keep the describe-pass prompt bounded
# even if the model ignores the "1-2 sentences" instruction.
SCENE_DESC_MAX_CHARS = 600

# Allowed enum values, mirrored from schema.json. Anything outside these sets is
# coerced back to "unknown" / "general" so jsonschema validation never fails on
# a creative model output.
ENUMS = {
    "scene_type": {"indoor", "outdoor", "vehicle", "unknown"},
    "lighting_condition": {"bright", "normal", "dim", "dark", "unknown"},
    "motion_level": {"still", "slow", "fast", "unknown"},
    "priority_context": {"navigation", "social", "reading", "general"},
}

STATIC_SYSTEM = (
    "You are the visual scene analyzer for IRIS, an assistive vision system "
    "for blind and low-vision users. Examine the image and return a JSON "
    "object describing the scene. Set run_* flags true ONLY when that "
    "downstream analysis would meaningfully help the user (e.g. run_ocr only "
    "if there is readable text, run_face_recognition only if faces are "
    "visible). Decide is_new_scene by comparing against the previous context "
    "you are given — flip it true on a clear change of place or activity. "
    "Also write a short factual scene_description (1–2 sentences) covering "
    "what is visible: people and what they appear to be doing, key objects, "
    "spatial layout (left/right/center/foreground), and any hazards. This "
    "description is the primary source for the downstream narrator, so be "
    "concrete and avoid filler — no opinions, no advice, no greetings."
)

STATIC_USER_TEMPLATE = (
    "Previous static context (may be empty on first frame):\n{previous}\n\n"
    "Return ONLY a JSON object with EXACTLY these keys:\n"
    "  is_new_scene (bool)\n"
    "  scene_type (one of: indoor, outdoor, vehicle, unknown)\n"
    "  lighting_condition (one of: bright, normal, dim, dark, unknown)\n"
    "  motion_level (one of: still, slow, fast, unknown)\n"
    "  run_object_detection (bool)\n"
    "  run_face_recognition (bool)\n"
    "  run_emotion_detection (bool)\n"
    "  run_ocr (bool)\n"
    "  run_depth_estimation (bool)\n"
    "  people_present (bool)\n"
    "  people_count (non-negative integer)\n"
    "  text_visible (bool)\n"
    "  hazard_detected (bool)\n"
    "  priority_context (one of: navigation, social, reading, general)\n"
    "  scene_description (string, 1–2 plain sentences, factual: people, "
    "objects, spatial layout, hazards. No advice, no greetings.)"
)

# Per-profile guidance baked into the describe-pass system prompt. Keeps the
# adapter the single owner of "what does each vision profile expect" so we can
# tune phrasing without touching the orchestrator.
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

# Hard ceilings on generated tokens per verbosity. Without these, the model
# tends to keep elaborating well past the requested length, costing latency
# for output the user will never hear. Sized with ~25% headroom over the
# token count of a well-behaved response at each level.
VERBOSITY_NUM_PREDICT = {
    "minimal": 60,
    "standard": 160,
    "detailed": 350,
}


class Gemma3OllamaVLM(ModelAdapter):
    """Two-model Gemma adapter served by a local Ollama HTTP endpoint.

    Static and describe passes use independent model tags so each can be sized
    for its job. The static pass needs vision and structured output → small
    multimodal model. The describe pass is text-in/text-out (the DataPool
    already carries every dynamic adapter's result) → small text-only LLM,
    no vision tower to load. With ``model="gemma3:4b"`` and
    ``model_describe="gemma3:1b"`` both fit in <5 GB VRAM together.
    """

    version = "gemma3-ollama-split-0.2"
    writes = ["static.*", "dynamic.vlm_description"]

    def __init__(
        self,
        role: str,
        model: str = "gemma3:4b",
        model_describe: Optional[str] = "gemma3:1b",
        host: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        timeout_s: float = 120.0,
        max_image_side: int = 896,
        temperature: float = 0.2,
        num_ctx: int = 2048,
        num_predict_static: int = 400,
        num_predict_describe: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(role, **kwargs)
        # ``model`` handles the static (multimodal) pass.
        self.model = model
        # ``model_describe`` handles the describe pass. None means "reuse the
        # static model" — used when you only want to manage one tag.
        self.model_describe = model_describe or model
        # Strip trailing slash so urljoin-style concatenation below stays clean.
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        # Gemma 3's vision tower expects 896×896. Sending larger images wastes
        # bandwidth and compute on resizes Ollama would do anyway. 0 disables.
        self.max_image_side = max_image_side
        self.temperature = temperature
        # 2k is enough: one image (~256 tokens) + previous-context JSON +
        # instructions ≈ 700 tokens worst case. The DataPool feeds a fresh
        # prompt every call, so KV-cache reuse across calls is not relied on.
        self.num_ctx = num_ctx
        # Static JSON has ~14 short fields plus a 1–2 sentence scene_description.
        # 400 tokens fits worst case (~250 typical) while still capping any
        # rambler that ignores the brevity instruction.
        self.num_predict_static = num_predict_static
        # Describe ceiling: None means "derive from the user's verbosity
        # setting at call time"; a number forces an explicit cap.
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
        mode = kwargs.get("mode", "static")
        if mode == "static":
            return self._run_static(frame, pool_snapshot)
        return self._run_describe(pool_snapshot)

    # ------------------------------------------------------------------
    # Static pass: image → structured JSON
    # ------------------------------------------------------------------

    def _run_static(self, frame: np.ndarray, snap: dict) -> dict:
        previous = snap.get("static", {}) or {}
        # Strip orchestrator-managed fields from the "previous" context — they
        # are noise for the model and would just take up tokens.
        previous_for_prompt = {
            k: v for k, v in previous.items()
            if k not in {"frame_id", "timestamp"}
        }

        image_b64 = self._encode_frame(frame)
        prompt = STATIC_USER_TEMPLATE.format(
            previous=json.dumps(previous_for_prompt, ensure_ascii=False)
        )

        body = {
            "model": self.model,
            "prompt": prompt,
            "system": STATIC_SYSTEM,
            "images": [image_b64],
            # Ollama JSON mode: the server constrains decoding so the response
            # is always parseable. Without this Gemma occasionally wraps JSON
            # in ```json fences or adds prose.
            "format": "json",
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
            log.warning("Gemma3 static call failed: %s", e)
            return dict(STATIC_DEFAULTS)

        raw = resp.get("response", "")
        return self._coerce_static(raw)

    @staticmethod
    def _coerce_static(raw: str) -> dict:
        """Parse the model's JSON output and clamp every field to schema bounds.

        Defensive on purpose: even with ``format: "json"`` we have seen models
        emit a key the schema does not allow. Coercing here means the pool's
        jsonschema check still passes, so downstream stages run instead of the
        whole frame being dropped.
        """
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            log.warning("Gemma3 static returned non-JSON despite format=json: %r", raw[:200])
            parsed = {}

        out = dict(STATIC_DEFAULTS)
        for key, default in STATIC_DEFAULTS.items():
            if key not in parsed:
                continue
            value = parsed[key]
            if key in ENUMS:
                # Enums: lower-case + membership check, otherwise fall back.
                value = str(value).lower()
                if value not in ENUMS[key]:
                    value = default
            elif isinstance(default, bool):
                # Accept "true"/"false"/0/1 in addition to real booleans.
                if isinstance(value, str):
                    value = value.strip().lower() == "true"
                else:
                    value = bool(value)
            elif isinstance(default, int):
                try:
                    value = max(0, int(value))
                except (TypeError, ValueError):
                    value = default
            elif isinstance(default, str):
                # Free-form strings (scene_description, ...): coerce, trim,
                # cap length so a runaway model can't bloat the describe prompt.
                value = str(value).strip()
                if len(value) > SCENE_DESC_MAX_CHARS:
                    value = value[:SCENE_DESC_MAX_CHARS].rstrip() + "…"
            out[key] = value
        return out

    # ------------------------------------------------------------------
    # Describe pass: pool → user-facing string
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
            f"If hazard_detected is true, lead with the hazard. Do not list raw fields; "
            f"speak naturally. Output the description text only — no preamble, no JSON."
        )

        # The describe model is text-only — it never sees the image. Lead with
        # the multimodal model's scene_description (the closest thing it has to
        # "looking at the frame") and treat detector outputs as supporting
        # evidence. Scene metadata (run_* flags, frame_id, etc.) is mostly
        # orchestration noise so it goes last.
        dyn = dict(snap.get("dynamic", {}))
        scene_desc = dyn.pop("scene_description", "") or ""
        static_block = snap.get("static", {})

        prompt_sections: list[str] = []
        if scene_desc:
            prompt_sections.append(f"What the camera sees:\n{scene_desc}")
        else:
            # Fallback for older configs that didn't generate scene_description.
            prompt_sections.append("What the camera sees: (no scene description available)")

        if dyn:
            prompt_sections.append(
                "Detector outputs (objects, faces, text, ...):\n"
                f"{json.dumps(dyn, ensure_ascii=False, default=str)}"
            )

        # Only forward the static fields that actually affect phrasing —
        # everything else is orchestration metadata the narrator does not need.
        narrator_static = {
            k: static_block.get(k)
            for k in ("scene_type", "lighting_condition", "motion_level",
                      "people_count", "hazard_detected", "priority_context")
            if k in static_block
        }
        if narrator_static:
            prompt_sections.append(
                f"Scene context: {json.dumps(narrator_static, ensure_ascii=False)}"
            )

        prompt_sections.append("Write the description now.")
        prompt = "\n\n".join(prompt_sections)

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
            log.warning("Gemma3 describe call failed: %s", e)
            return None

        text = (resp.get("response") or "").strip()
        return text or None

    # ------------------------------------------------------------------
    # Frame encoding
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: np.ndarray) -> str:
        """BGR ndarray → base64-encoded JPEG suitable for Ollama's ``images`` field.

        JPEG (not PNG) because the payload travels as base64 in JSON and a
        12-megapixel PNG inflates the request well past Ollama's default body
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
