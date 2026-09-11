"""Load experiment config from YAML (configs/*.yaml).

Single source of truth for world/drive/reflection parameters, so scripts never
hardcode values that can silently drift from the config file. CLI flags in the
scripts override individual config values for one-off runs; the YAML is the
default.
"""

from __future__ import annotations

import random
from typing import Any

import yaml

from .drives import DriveKind, DriveSpec


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def drive_specs_from_config(cfg: dict[str, Any]) -> dict[DriveKind, DriveSpec]:
    specs = {}
    for name, params in cfg["drives"].items():
        kind = DriveKind(name)
        specs[kind] = DriveSpec(kind=kind, **params)
    return specs


def resources_from_config(cfg: dict[str, Any]) -> list:
    from .sim.world import ResourceNode  # local import: avoid a cycle with sim/

    return [
        ResourceNode(location=r["location"], drive=DriveKind(r["drive"]),
                    quantity=r["quantity"])
        for r in cfg["world"]["resources"]
    ]


def initial_locations(rng: random.Random, locations: list[str], n_agents: int) -> list[str]:
    """Starting location per agent, in agent order.

    When there are at least as many locations as agents, place them one-per-
    location -- no accidental co-location at t=0. This matters for isolation
    conditions (configs/experiment_null_baseline.yaml) where NOT co-locating
    agents is the entire point; without it, independent rng.choice() per agent
    can collide by chance even when the config intends full isolation.

    Otherwise (fewer locations than agents, e.g. the default 5-location/6-agent
    experiment_minimal.yaml) falls back to random-with-replacement, which is what
    allows agents to start co-located and able to perceive each other.
    """
    if len(locations) >= n_agents:
        return rng.sample(locations, n_agents)
    return [rng.choice(locations) for _ in range(n_agents)]
