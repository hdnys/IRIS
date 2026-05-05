"""Generic two-model VLM adapter served by a local Ollama daemon.

Architecture:

* ``mode="static"``  — sends the frame to a fast multimodal model and returns
  a free-form scene description string (1–2 sentences). No JSON, no run_*
  flags, no orchestration metadata. The orchestrator routes the string into
  ``dynamic.scene_description`` so it lands alongside every other adapter's
  output. Default model: ``llava-phi3`` (~2.9 B params, fast multimodal VQA).

* ``mode="describe"`` — text-only narration pass. Reads the assembled pool
  (scene description + detector outputs) and emits the user-facing string.
  Does not see the image. Default model: ``gemma3:1b`` (~0.7 GB).

Why this shape: in the previous design the static pass produced a 14-field
JSON of orchestration flags. Generating those tokens cost ~2 s of latency
per frame, and the flags themselves only gated stub adapters. We dropped
the flags, dropped JSON mode, and now the static pass is a single short
description (~50 generated tokens, ~250–500 ms warm).

Pull both tags once::

    ollama pull llava-phi3
    ollama pull gemma3:1b

YAML wiring::

    vlm:
      module: Moteur.adapters.vlm_ollama
      class:  OllamaVLM
      params:
        model:           llava-phi3
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
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np

from Moteur.adapters.base import ModelAdapter

log = logging.getLogger(__name__)


# Structured outline prompt for llava-phi3. Produces labeled keyword sections
# that gemma3:1b can reliably interpret — avoids the hallucination and
# over-poetic prose that natural-language scene prompts produce.
SCENE_PROMPT = (
    "Analyze this image and output a structured scene outline. "
    "Use short keywords and phrases only — NO full sentences. "
    "Be thorough: list everything you observe in each category.\n\n"
    "SPATIAL_LEFT: <objects, people, features on the left side>\n"
    "SPATIAL_CENTER: <objects, people, features in the center>\n"
    "SPATIAL_RIGHT: <objects, people, features on the right side>\n"
    "SPATIAL_FOREGROUND: <what is close to the camera>\n"
    "SPATIAL_BACKGROUND: <what is far from the camera or behind other elements>\n"
    "ENVIRONMENT: <type of space, e.g. office, kitchen, corridor, outdoor street, living room>\n"
    "LIGHTING: <quality and source, e.g. bright natural from left, dim overhead, shadowed, backlit>\n"
    "SURFACES: <visible floors, walls, ceiling, tables, counters, furniture surfaces>\n"
    "ACTIONS: <what people or things are actively doing, e.g. person typing, walking toward camera, arm raised>\n"
    "INTERACTIONS: <how people or objects relate to each other or the space, e.g. person facing desk, two people talking>\n"
    "CONTEXT: <any other notable details, e.g. door open, screen on, window showing outside, clutter, empty space>\n\n"
    "If a category has nothing notable, write NONE. "
    "Ignore colored detection boxes and text overlays on the image — "
    "treat any labeled names as your own knowledge of who is present."
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


# Cap on the structured outline length forwarded to the narrator prompt.
# Structured outlines are longer than prose — 1200 gives 10-12 populated
# categories without bloating gemma3:1b's context.
SCENE_DESC_MAX_CHARS = 1200


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
            for o in non_person_objects[:_MAX_OBJECTS_FOR_NARRATOR]
        ]
        parts.append("Objects detected: " + ", ".join(obj_strs))

    if not parts:
        return ""

    return (
        "Sensor data for this frame (ground truth — do not contradict):\n"
        + "\n".join(f"  {p}" for p in parts) + "\n"
        "Use the above as authoritative facts for identities and object labels. "
        "In your outline, add spatial positions, environment, lighting, actions, "
        "and context that the sensors cannot provide."
    )


class OllamaVLM(ModelAdapter):
    """Multimodal scene-description model + small text narrator, both via Ollama."""

    version = "ollama-scene-narrator-0.3"
    writes = ["dynamic.scene_description", "dynamic.vlm_description"]

    def __init__(
        self,
        role: str,
        model: str,
        model_describe: str,
        host: str,
        keep_alive: str,
        timeout_s: float,
        max_image_side: int,
        temperature: float,
        num_ctx: int,
        num_predict_static: int,
        num_predict_describe: Optional[int],
        departure_grace_s: float,
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
        # llava-phi3's vision encoder works efficiently at ≤512 px per side;
        # larger inputs cost more base64 without improving description quality.
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
        # Person count is authoritative from YOLO; SFace identities are used
        # for names only. When YOLO count is stable, identities are frozen to
        # the last count-change event to suppress SFace flicker (same physical
        # person getting a different label frame-to-frame).
        self._stable_recognized: set[str] = set()   # identities locked at last YOLO count change
        self._prev_yolo_person_count: int = -1       # -1 = uninitialized
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
            return self._run_static(frame, pool_snapshot)
        return self._run_describe(pool_snapshot)

    # ------------------------------------------------------------------
    # Scene-description pass (vision)
    # ------------------------------------------------------------------

    def _run_static(self, frame: np.ndarray, snap: dict) -> str:
        image_b64 = self._encode_frame(frame)

        # Read personas and non-person objects from the pool — already built
        # by the orchestrator's persona synthesis step before this call.
        dyn = snap.get("dynamic", {})
        personas = dyn.get("personas", []) or []
        objects  = dyn.get("object_detection", []) or []
        grounding = _build_grounding(personas, objects)
        prompt = SCENE_PROMPT + ("\n\n" + grounding if grounding else "")

        # ── model (self.model, e.g. llava-phi3) ─────────────────────────
        # Receives: SCENE_PROMPT + sensor-grounding block (personas + YOLO
        #           non-person objects) + the JPEG frame as base64.
        # Returns:  a raw scene-description string (no system prompt, no JSON).
        body = {
            "model": self.model,
            "prompt": prompt,
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
        # narrator prompt.
        if len(text) > SCENE_DESC_MAX_CHARS:
            text = text[:SCENE_DESC_MAX_CHARS].rstrip() + "…"
        return text

    # ------------------------------------------------------------------
    # Narrator pass (text-only)
    # ------------------------------------------------------------------

    def _run_describe(self, snap: dict) -> Optional[str]:
        profile = snap.get("user_profile", {})
        verbosity = profile.get("verbosity", "standard")
        language = profile.get("preferred_language", "en-US")

        # System prompt — kept short on purpose. gemma3:1b can only reliably
        # hold a handful of instructions; long bullet lists get partly ignored.
        # Few-shot examples anchor behaviour far more reliably than verbal
        # rules. The CHANGES examples train the model to lead with diffs
        # ("Elie just left") rather than re-describing static state.
        system = (
            f"You are IRIS, narrating a scene to a blind user.\n"
            f"Length: {VERBOSITY_GUIDANCE.get(verbosity, VERBOSITY_GUIDANCE['standard'])}\n"
            f"Language: {language}. Second person ('you'). No greetings, no advice.\n"
            "Use exact names from 'People in view'. Never invent names. "
            "Anyone unnamed is 'someone' or 'a person'.\n"
            "'Scene outline' is a structured keyword summary from a vision model — "
            "interpret it to write natural, spoken narration. Do NOT read it aloud literally.\n"
            "If 'Changes since last narration' is provided, lead with the change — "
            "the user already heard the previous state.\n\n"
            "Examples:\n"
            "  Changes: Mohamad just appeared           → 'Mohamad has just stepped in front of you.'\n"
            "  Changes: Elie left the view              → 'Elie is no longer in front of you.'\n"
            "  Changes: Mohamad in, Elie out            → 'Elie has left and Mohamad is now in front of you.'\n"
            "  Changes: 1 unrecognized person appeared  → 'Someone has just walked up to you.'\n"
            "  No changes, Mohamad center, desk foreground, office, typing "
            "→ 'Mohamad is at a desk directly in front of you, appearing to type.'\n"
            "  No changes, no people, chair left, bottle right, bright office "
            "→ 'There is a chair to your left and a bottle to your right in a bright room.'\n"
            "  Mohamad center foreground, Elie right background, both standing, corridor "
            "→ 'Mohamad is right in front of you with Elie further back to your right, both standing in a corridor.'\n"
            "  No changes, no people, kitchen, kettle foreground center "
            "→ 'A kettle is directly in front of you in what looks like a kitchen.'\n"
            "  Nour IS the person in the outline: large figure left, resting chin on hand, gazing at window, bright light "
            "→ 'Nour is to your left, resting his chin on his hand and looking toward a bright window.'\n\n"
            "When a name is given, that person and every description of a figure/man/woman in the outline are ONE entity. "
            "Describe them once, fluently. Never split one person into two references in the same sentence."
        )

        dyn = dict(snap.get("dynamic", {}))
        scene_desc = (dyn.pop("scene_description", "") or "").strip()
        # Personas were built by the orchestrator (Phase 2.5) from face_recognition
        # + object_detection. object_detection now contains non-person objects only.
        personas = dyn.pop("personas", []) or []
        objects  = dyn.pop("object_detection", []) or []
        dyn.pop("face_recognition", None)  # cleared by orchestrator; drop from "Other"

        # Headcount is YOLO-only — unmatched SFace faces (bounding_box is None)
        # must not inflate it.
        yolo_person_count = sum(
            1 for p in personas
            if p.get("bounding_box") is not None
        )
        objects_line = _format_objects_line(objects)

        # Current raw SFace identities drawn from personas — noisy frame-to-frame.
        sface_recognized_set = {
            p["person_id"] for p in personas
            if p["person_id"] and p["person_id"] != "Unknown"
        }

        # ---- Diff against the previous narration ---------------------------
        # YOLO person count is the authoritative headcount signal.
        # Identity update rules (asymmetric by design):
        #   • count changed   → reset stable to current SFace output.
        #   • count stable, SFace has known names → accept them (known overrides
        #     unknown AND other known — a confident recognition is always trusted).
        #   • count stable, SFace returns only Unknown → keep existing known
        #     identities (Unknown cannot evict a known name).
        is_first = not self._prev_described
        count_changed = yolo_person_count != self._prev_yolo_person_count

        appeared: list[str] = []
        departed: list[str] = []
        unknown_delta = 0

        if is_first:
            self._stable_recognized = sface_recognized_set
        elif count_changed:
            count_delta = yolo_person_count - self._prev_yolo_person_count
            if count_delta > 0:
                appeared = sorted(sface_recognized_set - self._stable_recognized)
            else:
                departed = sorted(self._stable_recognized - sface_recognized_set)
            prev_unknown = max(0, self._prev_yolo_person_count - len(self._stable_recognized))
            new_unknown  = max(0, yolo_person_count - len(sface_recognized_set))
            unknown_delta = new_unknown - prev_unknown
            self._stable_recognized = sface_recognized_set
        elif sface_recognized_set:
            # Count stable and SFace returned at least one known name:
            # trust the recognition (overrides both unknown slots and stale
            # known names). Unknown-only frames leave stable untouched.
            self._stable_recognized = sface_recognized_set

        # Narration uses the stable (YOLO-consistent) identity state.
        recognized_set = self._stable_recognized
        recognized     = sorted(recognized_set)

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
            sections.append(f"Scene outline (keywords from vision model):\n{scene_desc}")
        if objects_line:
            sections.append(f"Visible objects: {objects_line}")
        if dyn:
            sections.append(
                "Other detections: "
                f"{json.dumps(dyn, ensure_ascii=False, default=str)}"
            )

        # Build a spatially-rich people line from personas.
        # Names come from the stable recognized_set; YOLO-backed personas supply
        # position + size. Names not matched to any persona are listed without
        # position (YOLO missed the body but identity is known from stability).
        if yolo_person_count > 0:
            yolo_personas = [p for p in personas if p.get("bounding_box") is not None]
            covered_names: set[str] = set()
            person_descs: list[str] = []
            for p in yolo_personas:
                name = p["person_id"] if p["person_id"] in recognized_set else "Unknown"
                if name in recognized_set:
                    covered_names.add(name)
                pos  = p.get("position") or ""
                size = p.get("size") or ""
                info = ", ".join(x for x in [size, pos] if x)
                person_descs.append(f"{name} ({info})" if info else name)
            for name in sorted(recognized_set - covered_names):
                person_descs.append(name)
            sections.append("People in view: " + ", ".join(person_descs))
            # Explicit bridge: the scene outline describes the same people listed
            # above. Without this, small models narrate the name and the outline
            # description as two separate entities ("Nour is here. A man is typing").
            if recognized:
                noun = "person" if len(recognized) == 1 else "people"
                verb = "is" if len(recognized) == 1 else "are"
                sections.append(
                    f"Note: {', '.join(recognized)} {verb} the same {noun} described "
                    "in the scene outline above. Merge all their details (position, "
                    "action, appearance) into ONE description per person. "
                    "Do not refer to the same person twice."
                )

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

        # Save state before the HTTP call so a transient Ollama failure
        # doesn't leave us re-announcing the same diff next frame.
        self._prev_yolo_person_count = yolo_person_count
        self._prev_described = True

        # Cap describe length: explicit override > verbosity preset.
        max_tokens = (
            self.num_predict_describe
            if self.num_predict_describe is not None
            else VERBOSITY_NUM_PREDICT.get(verbosity, VERBOSITY_NUM_PREDICT["standard"])
        )

        # ── model_describe (self.model_describe, e.g. gemma3:1b) ────────
        # Receives: system = IRIS narrator persona + verbosity/language rules
        #           prompt = assembled pool sections (scene desc, YOLO objects,
        #                    recognized names, changes diff, generation cue)
        # No image — text-only; the scene description from self.model feeds in
        # via the "Scene: …" section of prompt.
        body = {
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
