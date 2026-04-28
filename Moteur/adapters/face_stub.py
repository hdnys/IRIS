"""Face-recognition stub. Replace with the existing face_recognition wrapper from main.py."""
from __future__ import annotations

from typing import Any

from Moteur.adapters.base import ModelAdapter


class FaceStub(ModelAdapter):
    version = "stub-0.1"
    writes = ["dynamic.faces"]

    def run(self, frame: Any, pool_snapshot: dict, **_: Any) -> Any:
        return [
            {
                "person_id": None,
                "confidence": 0.7,
                "emotion": None,
                "bounding_box": {"x": 50.0, "y": 50.0, "width": 80.0, "height": 80.0},
            }
        ]
