"""Logging schema. Everything the analysis layer needs comes from here.

We log generously and structurally (JSONL), because the entire experiment depends on
post-hoc analysis of trajectories, and because the confound defenses (DESIGN.md §8)
require being able to reconstruct exactly what prompt/perception each agent saw.

Three streams per run:
  self_models.jsonl  — every self-model version, per agent (for trajectory/divergence)
  ticks.jsonl        — per-agent per-tick state (drives, position, action, alive)
  events.jsonl       — notable world events (deaths, shocks, rescues) for causal traces
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any


@dataclasses.dataclass
class TickRecord:
    run_id: str
    tick: int
    agent_id: str
    alive: bool
    drives: dict[str, float]
    position: str
    chosen_action: str
    perceived: str          # exactly what the agent perceived (post-salience)
    self_model_version: int
    cause_of_death: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


@dataclasses.dataclass
class EventRecord:
    run_id: str
    tick: int
    kind: str               # "death" | "shock" | "rescue" | "resource_depleted" | ...
    agent_id: str | None
    detail: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


class RunLogger:
    """Writes the three JSONL streams for one run into runs/<run_id>/."""

    def __init__(self, run_id: str, root: str = "runs"):
        self.run_id = run_id
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self._self_models = open(
            os.path.join(self.dir, "self_models.jsonl"), "a", encoding="utf-8")
        self._ticks = open(
            os.path.join(self.dir, "ticks.jsonl"), "a", encoding="utf-8")
        self._events = open(
            os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8")
        self.log_meta({"run_id": run_id, "created": time.time()})

    def log_meta(self, meta: dict[str, Any]) -> None:
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def log_self_model(self, agent_id: str, record: dict[str, Any]) -> None:
        record = {**record, "agent_id": agent_id, "run_id": self.run_id}
        self._self_models.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._self_models.flush()

    def log_tick(self, record: TickRecord) -> None:
        self._ticks.write(record.to_json() + "\n")
        self._ticks.flush()

    def log_event(self, record: EventRecord) -> None:
        self._events.write(record.to_json() + "\n")
        self._events.flush()

    def close(self) -> None:
        for f in (self._self_models, self._ticks, self._events):
            f.close()
