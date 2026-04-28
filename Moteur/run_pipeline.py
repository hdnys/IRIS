"""Smoke test for the engine wiring.

Loads the YAML config, builds pool + registry + orchestrator, and runs a
single fake frame through the full pipeline. Useful as a sanity check
before plugging in real models or the capture layer.

Run from the project root::

    python -m Moteur.run_pipeline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from Moteur.core.orchestrator import Orchestrator
from Moteur.core.pool import DataPool
from Moteur.core.registry import AdapterRegistry

CONFIG_PATH = Path(__file__).parent / "config" / "iris.yaml"


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    pool = DataPool(user_profile=cfg["user_profile"])
    registry = AdapterRegistry.from_config(cfg)
    orch = Orchestrator(
        pool, registry,
        max_workers=cfg.get("pipeline", {}).get("max_workers", 4),
    )

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    description = orch.process_frame(fake_frame)

    print("DESCRIPTION:", description)
    print("STATE:", json.dumps(pool.snapshot(), indent=2, default=str))

    orch.shutdown()


if __name__ == "__main__":
    main()
