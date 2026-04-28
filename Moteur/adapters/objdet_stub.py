"""Object-detection stub. Returns one fake bounding box."""
from __future__ import annotations

from typing import Any

from Moteur.adapters.base import ModelAdapter


class ObjDetStub(ModelAdapter):
    version = "stub-0.1"
    writes = ["dynamic.objects"]

    def run(self, frame: Any, pool_snapshot: dict, **_: Any) -> Any:
        return [
            {
                "label": "chair",
                "confidence": 0.9,
                "bounding_box": {"x": 0.0, "y": 0.0, "width": 100.0, "height": 200.0},
                "distance_m": None,
            }
        ]
