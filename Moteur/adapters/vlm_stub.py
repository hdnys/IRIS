"""Hardcoded VLM stub.

Lets the pipeline run end-to-end before any real VLM is wired in. Replace by
pointing the ``vlm`` role in ``config/iris.yaml`` at a real adapter (Ollama,
llama.cpp, transformers, ...).
"""
from __future__ import annotations

from typing import Any

from Moteur.adapters.base import ModelAdapter


class VLMStub(ModelAdapter):
    version = "stub-0.1"
    writes = ["static.*", "dynamic.vlm_description"]

    def run(self, frame: Any, pool_snapshot: dict, **kwargs: Any) -> Any:
        mode = kwargs.get("mode", "static")
        if mode == "static":
            return {
                "is_new_scene": True,
                "scene_type": "indoor",
                "lighting_condition": "normal",
                "motion_level": "still",
                "run_object_detection": True,
                "run_face_recognition": True,
                "run_emotion_detection": False,
                "run_ocr": False,
                "run_depth_estimation": False,
                "people_present": True,
                "people_count": 1,
                "text_visible": False,
                "hazard_detected": False,
                "priority_context": "social",
            }

        profile = pool_snapshot.get("user_profile", {}).get("vision_profile", "low_vision")
        dyn = pool_snapshot.get("dynamic", {})
        objs = dyn.get("objects", []) or []
        faces = dyn.get("faces", []) or []
        return f"[stub:{profile}] Scene with {len(objs)} object(s) and {len(faces)} face(s)."
