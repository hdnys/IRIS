"""Local VLM adapter — Moondream2 (vision) + small instruct LLM (narrator).

No Ollama, no HTTP. Two HuggingFace transformers models load directly into the
process and stay resident in VRAM for the session. The split mirrors the one
we settled on for the Ollama version, only faster:

* **Static pass** uses ``vikhyatk/moondream2`` — a 1.9B-param VLM tuned for
  fast on-device visual Q&A. Sees the frame, returns the structured JSON dict
  and a short scene_description in one shot. ~2 GB VRAM at fp16, ~1 GB at int8.
* **Describe pass** uses a small instruct LLM (default
  ``Qwen/Qwen2.5-1.5B-Instruct``) — text-only, multilingual (matters for
  ``user_profile.preferred_language``). ~3 GB at fp16, ~1.5 GB at int8.

Combined VRAM at fp16: ~5 GB. Per-frame target on consumer GPUs: 300–700 ms warm.

Required deps::

    pip install torch transformers accelerate pillow

Optional, for int8 quantization (saves ~50% VRAM, slight latency cost)::

    pip install bitsandbytes

YAML wiring (drop into ``config/iris.yaml``)::

    vlm:
      module: Moteur.adapters.vlm_local
      class:  LocalVLM
      params:
        vision_model_id:    vikhyatk/moondream2
        vision_revision:    "2025-04-14"
        narrator_model_id:  Qwen/Qwen2.5-1.5B-Instruct
        device:             cuda
        dtype:              float16
        load_in_8bit:       false
        max_image_side:     768
        num_predict_static: 400
        num_predict_describe: null

If narrator quality is insufficient, swap to ``Qwen/Qwen2.5-3B-Instruct``
(~5 GB at fp16) or ``meta-llama/Llama-3.2-3B-Instruct``.
"""
from __future__ import annotations

import gc
import json
import logging
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from Moteur.adapters.base import ModelAdapter

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Static-pass schema definitions (mirrors the Ollama adapter)
# ----------------------------------------------------------------------

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
    # Routed into dynamic.scene_description by the orchestrator. Primary
    # input for the (text-only) narrator pass.
    "scene_description": "",
}

ENUMS = {
    "scene_type": {"indoor", "outdoor", "vehicle", "unknown"},
    "lighting_condition": {"bright", "normal", "dim", "dark", "unknown"},
    "motion_level": {"still", "slow", "fast", "unknown"},
    "priority_context": {"navigation", "social", "reading", "general"},
}

SCENE_DESC_MAX_CHARS = 600


# Moondream takes a single prompt (no system/user split). We bake the full
# instruction set into one block so the model has every constraint in one
# place. Concise on purpose — Moondream follows short structured prompts
# better than verbose ones.
STATIC_PROMPT_TEMPLATE = (
    "Examine the image. Output ONLY a single JSON object — no prose, no markdown "
    "fences. Required keys with their value types:\n"
    "  is_new_scene (bool, true if scene differs significantly from previous)\n"
    "  scene_type (indoor|outdoor|vehicle|unknown)\n"
    "  lighting_condition (bright|normal|dim|dark|unknown)\n"
    "  motion_level (still|slow|fast|unknown)\n"
    "  run_object_detection (bool)\n"
    "  run_face_recognition (bool, only if faces visible)\n"
    "  run_emotion_detection (bool, only if faces visible and expressions readable)\n"
    "  run_ocr (bool, only if readable text in scene)\n"
    "  run_depth_estimation (bool)\n"
    "  people_present (bool)\n"
    "  people_count (non-negative integer)\n"
    "  text_visible (bool)\n"
    "  hazard_detected (bool, true for steps/obstacles/traffic/etc.)\n"
    "  priority_context (navigation|social|reading|general)\n"
    "  scene_description (string, 1-2 factual sentences: people, objects, "
    "spatial layout, hazards. No advice, no greetings.)\n\n"
    "Previous static context: {previous}\n\n"
    "Respond with the JSON object now."
)


# Same describe-pass guidance as the Ollama adapter — the narrator's job is
# unchanged regardless of which engine drives it.
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


_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


class LocalVLM(ModelAdapter):
    """In-process VLM (Moondream2) + text narrator (small instruct LLM)."""

    version = "local-moondream-qwen-0.1"
    writes = ["static.*", "dynamic.scene_description", "dynamic.vlm_description"]

    def __init__(
        self,
        role: str,
        vision_model_id: str = "vikhyatk/moondream2",
        # Pin a known-good revision so a HF Hub update can't silently change
        # the API. Override in YAML if you want a newer release.
        vision_revision: Optional[str] = "2025-04-14",
        narrator_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        narrator_revision: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_8bit: bool = False,
        max_image_side: int = 768,
        num_predict_static: int = 400,
        num_predict_describe: Optional[int] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(role, **kwargs)
        self.vision_model_id = vision_model_id
        self.vision_revision = vision_revision
        self.narrator_model_id = narrator_model_id
        self.narrator_revision = narrator_revision
        # Resolve "cuda" → fall back to CPU automatically if no GPU. Lets the
        # adapter run on a workstation with the GPU in use elsewhere without
        # a config edit.
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable — falling back to CPU")
            device = "cpu"
        self.device = device
        self.dtype = _DTYPES.get(dtype.lower(), torch.float16)
        self.load_in_8bit = load_in_8bit
        # Moondream's vision tower handles up to 768px efficiently. Larger
        # inputs spend extra compute on resizes the tower would do anyway.
        self.max_image_side = max_image_side
        self.num_predict_static = num_predict_static
        self.num_predict_describe = num_predict_describe
        self.temperature = temperature
        # Lazily filled by load(). __init__ runs at registry construction
        # time and we don't want to download weights as a side effect of
        # importing the YAML.
        self._vision_model = None
        self._vision_tokenizer = None
        self._narrator_model = None
        self._narrator_tokenizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load both models into VRAM. Called once by the registry at startup."""
        log.info("LocalVLM: loading vision model %s ...", self.vision_model_id)

        vision_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.vision_revision:
            vision_kwargs["revision"] = self.vision_revision
        if self.load_in_8bit:
            # bitsandbytes places the model itself; do NOT call .to(device).
            vision_kwargs["load_in_8bit"] = True
            vision_kwargs["device_map"] = self.device
        else:
            vision_kwargs["torch_dtype"] = self.dtype

        self._vision_model = AutoModelForCausalLM.from_pretrained(
            self.vision_model_id, **vision_kwargs
        )
        if not self.load_in_8bit:
            self._vision_model = self._vision_model.to(self.device)
        self._vision_model.eval()

        # Moondream ships a custom tokenizer at the same revision.
        tok_kwargs = {"revision": self.vision_revision} if self.vision_revision else {}
        self._vision_tokenizer = AutoTokenizer.from_pretrained(
            self.vision_model_id, **tok_kwargs
        )

        log.info("LocalVLM: loading narrator model %s ...", self.narrator_model_id)
        narrator_kwargs: dict[str, Any] = {}
        if self.narrator_revision:
            narrator_kwargs["revision"] = self.narrator_revision
        if self.load_in_8bit:
            narrator_kwargs["load_in_8bit"] = True
            narrator_kwargs["device_map"] = self.device
        else:
            narrator_kwargs["torch_dtype"] = self.dtype

        self._narrator_tokenizer = AutoTokenizer.from_pretrained(
            self.narrator_model_id,
            revision=self.narrator_revision,
        )
        self._narrator_model = AutoModelForCausalLM.from_pretrained(
            self.narrator_model_id, **narrator_kwargs
        )
        if not self.load_in_8bit:
            self._narrator_model = self._narrator_model.to(self.device)
        self._narrator_model.eval()

    def unload(self) -> None:
        """Drop refs and free GPU memory. Called by the orchestrator on shutdown."""
        self._vision_model = None
        self._vision_tokenizer = None
        self._narrator_model = None
        self._narrator_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, frame: Any, pool_snapshot: dict, **kwargs: Any) -> Any:
        mode = kwargs.get("mode", "static")
        if mode == "static":
            return self._run_static(frame, pool_snapshot)
        return self._run_describe(pool_snapshot)

    # ------------------------------------------------------------------
    # Static pass (vision)
    # ------------------------------------------------------------------

    def _run_static(self, frame: np.ndarray, snap: dict) -> dict:
        image = self._frame_to_pil(frame)
        previous = snap.get("static", {}) or {}
        previous_for_prompt = {
            k: v for k, v in previous.items()
            if k not in {"frame_id", "timestamp"}
        }

        prompt = STATIC_PROMPT_TEMPLATE.format(
            previous=json.dumps(previous_for_prompt, ensure_ascii=False)
        )

        try:
            with torch.inference_mode():
                # encode_image is the heavy step — runs the vision tower once.
                # answer_question then decodes from the encoded image.
                enc = self._vision_model.encode_image(image)
                raw = self._vision_model.answer_question(
                    enc, prompt, self._vision_tokenizer,
                    max_new_tokens=self.num_predict_static,
                )
        except Exception as e:
            log.warning("LocalVLM static call failed: %s", e)
            return dict(STATIC_DEFAULTS)

        return self._coerce_static(raw)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the first balanced JSON object out of free-form model output.

        Moondream isn't constrained to JSON, so its answer may wrap the object
        in fences or prose. We scan for the first ``{``, walk forward tracking
        brace depth, and try to parse the slice. Failure → empty dict, which
        STATIC_DEFAULTS will cover.
        """
        if not text:
            return {}
        start = text.find("{")
        if start == -1:
            return {}
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return {}
        return {}

    def _coerce_static(self, raw: str) -> dict:
        """Parse the model's JSON, clamp every field to schema bounds."""
        parsed = self._extract_json(raw)
        if not parsed:
            log.warning("LocalVLM static: no parseable JSON in output: %r", raw[:200])

        out = dict(STATIC_DEFAULTS)
        for key, default in STATIC_DEFAULTS.items():
            if key not in parsed:
                continue
            value = parsed[key]
            if key in ENUMS:
                value = str(value).lower()
                if value not in ENUMS[key]:
                    value = default
            elif isinstance(default, bool):
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
                value = str(value).strip()
                if len(value) > SCENE_DESC_MAX_CHARS:
                    value = value[:SCENE_DESC_MAX_CHARS].rstrip() + "…"
            out[key] = value
        return out

    # ------------------------------------------------------------------
    # Describe pass (text-only narrator)
    # ------------------------------------------------------------------

    def _run_describe(self, snap: dict) -> Optional[str]:
        profile = snap.get("user_profile", {})
        vision = profile.get("vision_profile", "low_vision")
        verbosity = profile.get("verbosity", "standard")
        language = profile.get("preferred_language", "en-US")

        system_prompt = (
            f"You are IRIS, speaking aloud to a user.\n"
            f"User profile: {PROFILE_GUIDANCE.get(vision, PROFILE_GUIDANCE['low_vision'])}\n"
            f"Verbosity: {VERBOSITY_GUIDANCE.get(verbosity, VERBOSITY_GUIDANCE['standard'])}\n"
            f"Reply in {language}. Speak in second person ('you see…', 'in front of you…'). "
            f"If hazard_detected is true, lead with the hazard. Do not list raw fields; "
            f"speak naturally. Output only the description text — no preamble, no JSON."
        )

        # Lead with the multimodal model's scene_description (the only thing
        # that resembles "looking at the frame" from the narrator's POV);
        # detector outputs are supporting evidence; orchestration metadata is
        # filtered down to what affects phrasing.
        dyn = dict(snap.get("dynamic", {}))
        scene_desc = (dyn.pop("scene_description", "") or "").strip()
        static_block = snap.get("static", {})

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

        narrator_static = {
            k: static_block.get(k)
            for k in ("scene_type", "lighting_condition", "motion_level",
                      "people_count", "hazard_detected", "priority_context")
            if k in static_block
        }
        if narrator_static:
            sections.append(
                f"Scene context: {json.dumps(narrator_static, ensure_ascii=False)}"
            )

        sections.append("Write the description now.")
        user_prompt = "\n\n".join(sections)

        max_tokens = (
            self.num_predict_describe
            if self.num_predict_describe is not None
            else VERBOSITY_NUM_PREDICT.get(verbosity, VERBOSITY_NUM_PREDICT["standard"])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = self._narrator_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._narrator_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self._narrator_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    pad_token_id=(
                        self._narrator_tokenizer.pad_token_id
                        or self._narrator_tokenizer.eos_token_id
                    ),
                )
            # Strip the prompt tokens — generate() returns the full sequence.
            response = self._narrator_tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()
        except Exception as e:
            log.warning("LocalVLM describe call failed: %s", e)
            return None

        return response or None

    # ------------------------------------------------------------------
    # Frame conversion
    # ------------------------------------------------------------------

    def _frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Capture-layer BGR ndarray → PIL RGB at most ``max_image_side`` per side."""
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
