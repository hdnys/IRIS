"""Generic two-model VLM adapter served by a local Ollama daemon.

Architecture:

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
      module: Moteur.adapters.vlm_ollama
      class:  OllamaVLM
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

Any multimodal tag works for ``model`` and any text tag for ``model_describe``.
"""
from __future__ import annotations

import base64
import json
import logging
import time
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
    "People may have colored face-detection boxes and labels over them; "
    "ignore these overlays and never describe them. "
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
    "minimal": "ONE clause, max 8 words. Only the single most important fact.",
    "standard": "ONE sentence, max 18 words. Direct and factual.",
    "detailed": "Up to two sentences, 35 words total max. No filler.",
}

VERBOSITY_NUM_PREDICT = {
    "minimal": 25,
    "standard": 50,
    "detailed": 110,
}

# Cap on the number of YOLO detections we forward to the narrator. Beyond this
# the prompt bloats faster than gemma3:1b can absorb the structure.
_MAX_OBJECTS_FOR_NARRATOR = 8


def _format_objects_line(objects: list[dict]) -> str:
    """Compact, narrator-friendly summary of YOLO detections.

    Groups duplicates by label so the prompt stays short:

        [{label:'chair',position:'center',size:'large'},
         {label:'chair',position:'left-middle',size:'medium'},
         {label:'bottle',position:'top-right',size:'small'}]

    becomes::

        "2 chairs (large center, medium left-middle), bottle (small top-right)"

    Caller passes the adapter's already-sorted list (most-confident first), so
    the truncation below preserves the most prominent detections.
    """
    if not objects:
        return ""

    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for obj in objects[:_MAX_OBJECTS_FOR_NARRATOR]:
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("label", "")).strip()
        if not label:
            continue
        if label not in grouped:
            order.append(label)
            grouped[label] = []
        grouped[label].append(obj)

    parts: list[str] = []
    for label in order:
        items = grouped[label]
        descs = [
            f"{o.get('size','?')} {o.get('position','?')}"
            for o in items
        ]
        if len(items) == 1:
            parts.append(f"{label} ({descs[0]})")
        else:
            # Pluralize the simple way — "chairs" is fine for the small model;
            # we don't need linguistic correctness for cases like "mouse".
            parts.append(f"{len(items)} {label}s ({', '.join(descs)})")
    return ", ".join(parts)


# Cap on raw scene_description length — defends the narrator's prompt from
# a runaway VLM that ignores the "1-2 sentences" instruction.
SCENE_DESC_MAX_CHARS = 600


class OllamaVLM(ModelAdapter):
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
        temperature: float = 0.3,
        num_ctx: int = 2048,
        num_predict_static: int = 120,
        num_predict_describe: Optional[int] = None,
        departure_grace_s: float = 300.0,
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
        # A recognized person must be continuously absent this long before we
        # narrate that they left.
        self.departure_grace_s = max(0.0, float(departure_grace_s))

        # ---- Cross-frame memory for change-aware narration -----------------
        # The narrator describes a moving scene to a user who can't see — what
        # they actually need to hear is *what changed* ("Elie just left",
        # "someone walked in"), not a re-description of state they already
        # have. We keep a tiny state machine of who/what was visible at the
        # last describe call and surface the diff in the next prompt.
        #
        # Lives for the whole session. The orchestrator runs only one
        # describe call at a time, so no locking is required.
        self._prev_recognized: set[str] = set()
        self._last_seen_recognized: dict[str, float] = {}
        self._prev_unknown_count: int = 0
        self._prev_described: bool = False

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

        # System prompt — kept short on purpose. gemma3:1b can only reliably
        # hold a handful of instructions; long bullet lists get partly ignored.
        # Few-shot examples anchor behaviour far more reliably than verbal
        # rules. The CHANGES examples train the model to lead with diffs
        # ("Elie just left") rather than re-describing static state.
        system = (
            f"You are IRIS, narrating a moving camera frame to a blind user.\n"
            f"Length: {VERBOSITY_GUIDANCE.get(verbosity, VERBOSITY_GUIDANCE['standard'])}\n"
            f"Language: {language}. Second person ('you'). No greetings, no advice.\n"
            "Use exact names from 'Recognized people'. Never invent names. "
            "Anyone else is 'someone' or 'a person'.\n"
            "If 'Changes since last narration' is provided, lead with the change — "
            "the user already heard the previous state.\n\n"
            "Examples:\n"
            "  Changes: Mohamad just appeared           → 'Mohamad has just stepped in front of you.'\n"
            "  Changes: Elie left the view              → 'Elie is no longer in front of you.'\n"
            "  Changes: Mohamad in, Elie out            → 'Elie has left and Mohamad is now in front of you.'\n"
            "  Changes: 1 unrecognized person appeared  → 'Someone has just walked up to you.'\n"
            "  No changes, Recognized: Mohamad          → 'Mohamad is still in front of you.'\n"
            "  No changes, no people                    → 'A chair sits ahead on your right.'\n"
            "  Recognized: Mohamad. Objects: chair (large center), bottle (small top-right) "
            "→ 'Mohamad is in front of you, with a chair just ahead and a bottle to your upper right.'\n"
            "  Recognized: Mohamad. Objects: 2 chairs (large center, medium left-middle) "
            "→ 'Mohamad is in front of you, with chairs around him.'\n\n"
            "When 'Visible objects' is provided, treat it as ground truth — those items "
            "ARE in the scene. If a recognized person is also present, weave them together "
            "into ONE sentence (the person AND the most prominent 1-2 objects with their "
            "position). Do not invent objects that are not listed."
        )

        dyn = dict(snap.get("dynamic", {}))
        scene_desc = (dyn.pop("scene_description", "") or "").strip()
        faces = dyn.pop("face_recognition", []) or []
        # Object detections get formatted into a compact human-readable line
        # below — popping here keeps the raw JSON out of "Other detections".
        objects = dyn.pop("object_detection", []) or []
        objects_line = _format_objects_line(objects)

        recognized_list = [
            f.get("person_id", "") for f in faces
            if isinstance(f, dict)
            and f.get("person_id")
            and f["person_id"] != "Unknown"
        ]
        recognized = sorted(set(recognized_list))   # de-dupe + stable order
        recognized_set = set(recognized)
        unknown_count = sum(
            1 for f in faces
            if isinstance(f, dict) and f.get("person_id") == "Unknown"
        )

        # ---- Diff against the previous narration ---------------------------
        # Capture is_first BEFORE _prev_described is updated below so the
        # cue branch can tell "first frame of session" from "scene unchanged".
        is_first = not self._prev_described
        now_s = time.monotonic()
        for name in recognized_set:
            self._last_seen_recognized[name] = now_s
        if not is_first:
            appeared = sorted(recognized_set - self._prev_recognized)
            departed = [
                name for name in sorted(self._prev_recognized - recognized_set)
                if (now_s - self._last_seen_recognized.get(name, now_s))
                >= self.departure_grace_s
            ]
            unknown_delta = unknown_count - self._prev_unknown_count
        else:
            # First call of the session — there is no "previous" frame to
            # diff against. Treat everything as the initial state.
            appeared = []
            departed = []
            unknown_delta = 0

        change_lines: list[str] = []
        for name in appeared:
            change_lines.append(f"{name} just entered the view.")
        for name in departed:
            change_lines.append(f"{name} is no longer in view.")
        if unknown_delta > 0:
            noun = "person" if unknown_delta == 1 else "people"
            change_lines.append(
                f"{unknown_delta} unrecognized {noun} just appeared."
            )
        elif unknown_delta < 0:
            n = -unknown_delta
            noun = "person" if n == 1 else "people"
            change_lines.append(f"{n} unrecognized {noun} left the view.")

        # Build the user prompt with the most attention-worthy content LAST.
        # Small models weight the freshest prompt content most heavily, so
        # the change diff (when present) sits right next to the generation
        # cue; otherwise the recognized-name list does.
        sections: list[str] = []
        if scene_desc:
            sections.append(f"Scene: {scene_desc}")
        if objects_line:
            sections.append(f"Visible objects: {objects_line}")
        if dyn:
            sections.append(
                "Other detections: "
                f"{json.dumps(dyn, ensure_ascii=False, default=str)}"
            )

        if recognized:
            sections.append("People currently in view: " + ", ".join(recognized))
        if unknown_count and not recognized:
            sections.append(f"Unrecognized people in view: {unknown_count}")
        elif unknown_count:
            sections.append(f"Plus {unknown_count} unrecognized.")

        if change_lines:
            sections.append(
                "Changes since last narration:\n"
                + "\n".join(f"- {c}" for c in change_lines)
            )
            cue_names = sorted(set(appeared + departed))
            if cue_names:
                sections.append(
                    "Now narrate the change above in ONE short sentence. "
                    f"You MUST mention {', '.join(cue_names)} by name."
                )
            else:
                sections.append(
                    "Now narrate the change above in ONE short sentence:"
                )
        elif recognized:
            preamble = (
                "" if is_first
                else "Nothing has changed since the last narration. "
            )
            obj_cue = (
                " AND the most prominent visible object with its position"
                if objects_line else ""
            )
            sections.append(
                f"{preamble}Now narrate the current scene in ONE short sentence. "
                f"You MUST mention {', '.join(recognized)} by name{obj_cue}."
            )
        else:
            sections.append("Now narrate the scene in ONE short sentence:")

        prompt = "\n\n".join(sections)

        # Save state for the next describe call. Done BEFORE the HTTP call
        # so a transient Ollama failure doesn't leave us re-announcing the
        # same diff next frame.
        self._prev_recognized = (self._prev_recognized | recognized_set) - set(departed)
        for name in departed:
            self._last_seen_recognized.pop(name, None)
        self._prev_unknown_count = unknown_count
        self._prev_described = True

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
        if not text:
            return None

        # Safety net: small models sometimes drop the most important names
        # despite the prompt. Prepend the missing piece — a stilted line is
        # still better than a narration that silently omits a person event.
        # Departed names take priority because they're the most surprising
        # for the user to miss ("wait, where did Elie go?").
        prefixes: list[str] = []
        lower = text.lower()
        for name in departed:
            if name.lower() not in lower:
                prefixes.append(f"{name} is no longer in front of you.")
        for name in recognized:
            if name.lower() not in lower:
                prefixes.append(f"{name} is in front of you.")
        if prefixes:
            text = " ".join(prefixes) + " " + text

        return text

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
