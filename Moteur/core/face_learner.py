"""Guided multi-image face capture for the live pipeline.

State machine driven one frame at a time by the main loop. Walks the user
through a fixed sequence of head poses (look straight, turn left, turn
right, look up, look down), speaks each instruction through the supplied
TTS callback, waits for the user to settle, verifies a face is visible,
and saves one image per pose under ``<gallery_dir>/<name>/<n>.jpg``.

Why pose-guided capture: SFace cosine similarity drops sharply with head
yaw / pitch beyond ~15 degrees. A single frontal reference fails on a
profile view. Five well-spread poses give the multi-image gallery enough
coverage that recognition holds across head turns.

When ``done`` flips to True the learner has already saved every captured
frame and asked the supplied SFaceAdapter to reload its gallery — the
caller only needs to drop the learner reference and resume.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Long form is spoken via TTS; short form is shown on the HUD.
_PROMPTS: list[tuple[str, str]] = [
    ("Look straight at the camera.",                "Look straight"),
    ("Turn your head slightly to the left.",        "Turn left"),
    ("Turn your head slightly to the right.",       "Turn right"),
    ("Tilt your head a little upwards.",            "Look up"),
    ("Tilt your head a little downwards.",          "Look down"),
]


class FaceLearner:
    """One pose at a time: speak → wait → verify face → capture → advance."""

    SETTLE_S = 2.5            # delay between speaking the prompt and capturing
    POSE_TIMEOUT_S = 20.0     # give up on a single pose after this many seconds

    def __init__(
        self,
        name: str,
        sface_adapter: Any,
        speak: Callable[[str], None],
        gallery_dir: Path,
        prompts: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        self.name = name.strip()
        self._sface = sface_adapter
        self._speak = speak
        self._target_dir = Path(gallery_dir) / self.name
        self._prompts = prompts if prompts is not None else _PROMPTS

        self._idx: int = 0
        self._captured: list[np.ndarray] = []
        self._prompt_started_at: Optional[float] = None
        self._pose_started_at: Optional[float] = None

        self.done: bool = False
        self.failed: bool = False
        self._intro_spoken = False

    # ------------------------------------------------------------------
    # Per-frame entrypoint
    # ------------------------------------------------------------------

    def step(self, frame: np.ndarray) -> tuple[str, bool]:
        """Advance the state machine by one frame.

        Returns ``(message, done)``. ``message`` is meant for the on-screen
        overlay; ``done`` flips to True exactly once when the learner is
        finished (success or failure) and the caller should drop us.
        """
        if self.done:
            return ("", True)

        # Spoken intro fires once on the first call so we don't talk over
        # the previous narration.
        if not self._intro_spoken:
            self._intro_spoken = True
            self._speak(
                f"Starting face capture for {self.name}. "
                f"I will guide you through {len(self._prompts)} poses."
            )

        if self._idx >= len(self._prompts):
            return self._finish()

        long_prompt, short_prompt = self._prompts[self._idx]
        now = time.time()

        # First time we land on this pose — speak it and start the clock.
        if self._prompt_started_at is None:
            self._speak(long_prompt)
            self._prompt_started_at = now
            self._pose_started_at = now

        # Per-pose timeout. If the user can't get a face on camera in time
        # we skip the pose rather than stalling forever.
        if (self._pose_started_at is not None
                and now - self._pose_started_at > self.POSE_TIMEOUT_S):
            log.warning("Pose %d timed out — skipping.", self._idx + 1)
            self._idx += 1
            self._prompt_started_at = None
            self._pose_started_at = None
            return (
                f"[{self._idx}/{len(self._prompts)}] {short_prompt} — TIMED OUT",
                False,
            )

        elapsed = now - self._prompt_started_at
        remaining = max(0.0, self.SETTLE_S - elapsed)

        # Phase 1: settle. Show countdown so the user can pose.
        if remaining > 0:
            return (
                f"[{self._idx + 1}/{len(self._prompts)}] {short_prompt}  "
                f"(hold... {remaining:.1f}s)",
                False,
            )

        # Phase 2: capture. Verify a face is actually visible — if not,
        # keep waiting (don't reset the prompt timer, just hold).
        try:
            faces = self._sface.run(frame, {}) or []
        except Exception as e:
            log.warning("SFace failed during learn step: %s", e)
            faces = []

        if not faces:
            return (
                f"[{self._idx + 1}/{len(self._prompts)}] {short_prompt}  "
                f"(no face detected — hold still)",
                False,
            )

        # Got one — store the raw frame (saved later).
        self._captured.append(frame.copy())
        self._idx += 1
        self._prompt_started_at = None
        self._pose_started_at = None
        return (
            f"[{self._idx}/{len(self._prompts)}] Captured!",
            False,
        )

    # ------------------------------------------------------------------
    # Internal: finalize
    # ------------------------------------------------------------------

    def _finish(self) -> tuple[str, bool]:
        if not self._captured:
            self.failed = True
            self.done = True
            self._speak("No frames were captured. Aborting.")
            return ("No frames captured.", True)

        self._target_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self._captured, start=1):
            cv2.imwrite(str(self._target_dir / f"{i}.jpg"), frame)

        # Ask the live adapter to re-embed its gallery so the new person
        # is recognized on the very next frame.
        try:
            if hasattr(self._sface, "reload_gallery"):
                self._sface.reload_gallery()
            elif hasattr(self._sface, "_build_gallery"):
                self._sface._build_gallery()
        except Exception as e:
            log.warning("Gallery reload after learn failed: %s", e)

        self.done = True
        self._speak(f"Done. {self.name} has been added to the gallery.")
        return (
            f"Saved {len(self._captured)} images to {self._target_dir}",
            True,
        )
